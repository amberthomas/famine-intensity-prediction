#where bad code goes to die... or has already been killed and is buried for posterity 

# dropped histogram function helper (I wouldn't ever use this, since it misunderstands 
# the data compression power the histograms are supposed to offer)

def modis_monthly_hist(mframe, year, month, one_month = False):
    avg_df = pd.DataFrame([])
    agg_hdr = '{}-{}'.format(year, month)
    month_df = mframe.filter(like=agg_hdr)
    val_sum = pd.Series([])
    count = 0
    for c in month_df:
        if count == 0:
            val_sum = month_df[c]
        else:
            val_sum += month_df[c]
        count += 1
        if one_month: # histogram will be first of month instead of avg
            break
    month_avg = val_sum/count
    return month_avg

# dropped histogram function helper (I wouldn't ever use this, since it misunderstands 
# the data compression power the histograms are supposed to offer)
def get_hist_timeseries(mframe, date, n_ts, months_back = 1, flat=True):
    start = date - pd.DateOffset(months=months_back)
    timeseries = pd.Series([])
    for step in range(1, n_ts + 1):
        date_step = start - pd.DateOffset(months=step)
        ys, ms = [date_step.year, '{:02}'.format(date_step.month)]
        month_hist = modis_monthly_hist(mframe, ys, ms)
        #def concat(): 
        if step == 1:
            for r in range(0, month_hist.shape[0]):
                timeseries[r] = month_hist[r]
        else:
            for r in range(0, month_hist.shape[0]):
                timeseries[r] = np.concatenate((timeseries[r], month_hist[r]), axis = 1)
    if flat:
        for r in range(0, month_hist.shape[0]):
            timeseries[r] = np.ravel(timeseries[r])
    return  timeseries

# dropped histogram function helper (I wouldn't ever use this, since it misunderstands 
# the data compression power the histograms are supposed to offer)

# should average each feature over month
#def get_month_avg(mdf):
def modis_monthly_avg(mframe, year, month, feat_list, year_hdr = False):
    avg_df = pd.DataFrame([])
    for feat in feat_list:
        agg_hdr = '{}_{}-{}'.format(feat, year, month)
        col = feat
        if year_hdr: #adds date info to col name, not good for concat
            col = agg_hdr
        avg_df[col] = mframe.filter(like=agg_hdr).mean(axis=1)
    return avg_df



# a version of label / feature generation, has since been updated 
def label_feature_tables(bframe, mframe):
    phases = get_cols_by_datelist('1/1/2010', pd.datetime.today(), 'IPC_Phase', bframe)
    bframe = bframe.fillna(0)
    mframe = mframe.fillna(0)

    
    feature_table = pd.DataFrame(columns=['score'])
    region_num = len(bframe.index)
    num_scores = len(phases[2])
    print('Number of regions: ', region_num)
    print('Number of samples: ', num_scores)

    # for each label
    for i in range(0, num_scores):
        cur = pd.DataFrame([])
        #print(phases[1][0])
        dates =[phases[2][i]] * region_num
        cur['month'] = [m.month for m in dates]
        cur['time_delta'] = [(d - pd.datetime(2010,1,1)).days for d in dates]
        cur['month_sin'] = np.sin((cur.month-1)*(2.*np.pi/12))
        cur['month_cos'] = np.cos((cur.month-1)*(2.*np.pi/12))
        cur['year'] = [y.year for y in dates]
        for step in range(1, 4):
            suffix = '.' + str(step) + '_m_back'
            date_step = phases[2][i] - pd.DateOffset(months=step)
            ys, ms = [date_step.year, '{:02}'.format(date_step.month)]
            col_names_dates = [f + suffix for f in feats]
            all_feat_avgs = modis_monthly_avg(mframe, ys, ms, feats)
            
            cur[col_names_dates] = all_feat_avgs
            
            

        cur['score'] = bframe[phases[0][i]] # classifier
        feature_table = pd.concat([feature_table, cur])
    

    
    #display(feature_table[(feature_table.drop(drop_cols, axis=1).T == 0).any()])
    feature_table = feature_table[feature_table.score != 0] #classifier
    feature_table = feature_table.replace([np.inf, -np.inf], np.nan)
    feature_table = feature_table.fillna(0)
    drop_cols = ['month', 'score', 'time_delta', 'month_sin', 'month_cos', 'year']
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #display(feature_table[(feature_table.drop(drop_cols, axis=1).T == 0).any()])
    feature_table = feature_table[(feature_table.drop(drop_cols, axis=1).T != 0).any()]
    feature_table = feature_table.sort_index()
    label_table = pd.DataFrame(feature_table['score'], columns=['score'])
    feature_table.drop(['score', 'month'], axis=1, inplace=True)
    return label_table, feature_table

# I can't even remeber if I got this working. Twas once again, very dumb. 

def label_feature_hist_split(bframe, mframe, num_test):

    phases = get_cols_by_datelist('1/1/2010', '3/22/2017', 'IPC_Phase', bframe)
    bframe = bframe.fillna(0)
    mframe = mframe.fillna(0)
    region_num = len(bframe.index)
    num_scores = len(phases[2])
    feature_table = pd.DataFrame(columns=['hist','prev_score','score'])
    #feature_table = np.zeros((region_num*num_scores, 32, 3, 7))
    print('Total number of labels: ', num_scores)
    
    #NOTE THAT WE ARE NOT USING FAKE LABEL
    for i in range(1, num_scores):
        cur = pd.DataFrame([])
        date = phases[2][i]
        cur['score'] = bframe[phases[0][i]] # classifier
        if i == 0:
            cur['prev_score'] = fake_label(cur['score'], .2)
        else:
            cur['prev_score'] = bframe[phases[0][i-1]]

        cur['hist'] = get_hist_timeseries(mframe, date, 3)
        feature_table = pd.concat([feature_table, cur])
        print(feature_table.shape)

    
    feature_table = feature_table.replace([np.inf, -np.inf], np.nan)
    feature_table = feature_table.fillna(0)

    print('pre drop total rows: ', (feature_table.shape[0]))
    
    feature_table = feature_table[feature_table.score != 0] #classifier
    label_table = pd.DataFrame(feature_table['score'], columns=['score'])
    feature_table.drop(['score'], axis=1, inplace=True)
    
    train_feature = feature_table.iloc[0:region_num*(num_scores - num_test-1)]
    train_label = label_table.iloc[0:region_num*(num_scores - num_test-1)]
    
    test_feature = feature_table.iloc[-1*region_num*num_test:]
    test_label = label_table.iloc[-1*region_num*num_test:]
 
    print('Number of test score periods: ', np.ceil(test_label.shape[0]/region_num))
    print('Number of train score periods: ', np.ceil(train_label.shape[0]/region_num))
    print('Number of test score rows: ', (test_label.shape[0]))
    print('Number of train score row: ', (train_label.shape[0]))
    
    print('Post drop total rows: ', (feature_table.shape[0]))

    return train_label, train_feature, test_label, test_feature


"""wtf = train_feature['hist']
print(train_feature['prev_score'].iloc[0])
print(wtf.shape)
print(wtf[0])"""
# god even knows.
def format_hist_feature(features):
    feat_arr = np.zeros((features.shape[0], features['hist'].iloc[0].shape[0] + 1))
    for index, row in df.iterrows():
        print(row.shape)
        print(row)
        feat_arr[f]= np.append(row['hist'], np.array(row['prev_score']))
    return feat_arr
print(format_hist_feature(train_feature).shape)
