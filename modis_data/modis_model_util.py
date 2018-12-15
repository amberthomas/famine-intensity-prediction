#Here are the stupid number of packages that I use 
import pandas as pd
import itertools
from datetime import date
import numpy as np
import gdal
import datetime
from math import isnan
from scipy import stats
import matplotlib.pyplot as plt
import pydot
import seaborn as sns
from IPython.display import display

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


"""
get_dates(filename)

pulls the dates that should correspond to the modis images, 
the filename is the .txt file where the date list lives.
the date list is generated along side the images. 
"""
def get_dates(filename):
    list_dates = []
    with open(filename, 'r') as f:
        list_dates = [line.rstrip('\n') for line in f]
    return list_dates

"""
format_feature_df(df1, inplace)

df1 - the pd dataframe that holds modis data (already paired with dates) 
inplace - exactly as it sounds. If no argument given it will make a copy 
of the df and leave the original intact 

The input dataframe as the index of each region and date stored as a list.
This reformats so that each index is given a column
"""
def format_feature_df(df1, inplace = False):
    # there are 14 features 
    if not inplace:
        df1 = df1.copy()
    feats = ['fR', 'fNIR', 'fB', 'fG', 'fSR', 'fNDVI', 'fGNVDI', 'fTVI', 'fSAVI', 'fOSAVI', 'fNLI', 'fRVDI', 'fCARI', 'fPSRI']
    orig_hdrs = df1.columns.values
    df2 = pd.DataFrame(df1)
    for col in orig_hdrs:
        combo_hdr = [('{}_{}'.format(i[0], i[1])) for i in (itertools.product(feats, [col]))]
        df2[combo_hdr] = pd.DataFrame(df2[col].values.tolist(), index= df2.index)
    df2 = df2.drop(list(orig_hdrs), axis = 1)
    return df2


"""
get_cols_by_datelist(start, end, prefix, bucket_frame)

start- start date 
end - end date 
which Artemis csv prefix you want to aggregate (ie IPC score, acled_fatalities, Rainfall etc)
Which Artemis dataframe you want to pull from (ie afg_bframe)
"""
def get_cols_by_datelist(start, end, prefix, bucket_frame):
    datelist = pd.date_range(start, end, freq='M').tolist()
    col_list = []
    suffix = []
    dates = []
    for d in datelist:
        y, m = [d.year, '{:02}'.format(d.month)]
        col_name ='{}.{}_{}'.format(prefix, y, m)
        if col_name in bucket_frame:
            if (bucket_frame[col_name].count() > 0):
                col_list.append(col_name)
                suffix.append('.{}_{}'.format(y, m))
                dates.append(d)
    return col_list, suffix, dates


"""
format_df_graph(df3, start, end, shift)

df3 - df already formatted by /format_feature_df/ that you want to prep to compare 
against artemis data
start, end - just lets you scale the x - time axis to artemis data
shift - shifts the the data by x number of months 
"""
def format_df_graph(df3, start=None, end=None, shift=None):
    ndvidf = pd.DataFrame([])
    fdf = df3.filter(like='fNDVI')
    for col in fdf:
        temp = pd.DataFrame([])
        temp['fNDVI'] = fdf[col]
        temp['date'] = col.split('_')[1] 
        temp['region'] = fdf[col].index.values
        ndvidf = pd.concat([ndvidf, temp])
    ndvidf['date'] = pd.to_datetime(ndvidf['date'], format='%Y-%m-%d', utc=True)
    if shift:
        ndvidf['date'] = ndvidf['date'] + pd.DateOffset(months=shift)
    ndvidf.index = (ndvidf.date)
    ndvidf = ndvidf.loc[start:end]
    return ndvidf

"""
compare_frames(bframe, mframe, start, end, country, shift)

bframe- artemis data 
mframe -modis data 
start, end - time scale for modis data  
country - which country are you running comparisons on? for title of graph 
shift - shift for modis data 

plots a graph to compare Average NDVI across the entire country for modis and 
Artemis data 
prints RSME
"""
def compare_frames(bframe, mframe, start, end, country, shift = None):
    #(acled_frame, bucket_frame, country):
    #plt.figure(figsize=(20,15))
    #plt.rcParams.update({'font.size': 22})
    plt.xticks(rotation=70)
    plt.title('Compare MODIS and FAM NDVI readings for {}'.format(country), fontweight ='bold')
    ndvidf = format_df_graph(mframe, start, end, shift)
    plt.plot(ndvidf['fNDVI'].resample('M').mean(), '-', label = 'MODIS', linewidth='4')
    bfy = []
    bfx = []
    col, month, dates = get_cols_by_datelist(start, end, 'NdviAll', bframe)
    for i in range(0, len(col)):
        bfy.append(bframe[col[i]].mean())
        bfx.append(dates[i])

    plt.plot(bfx, bfy, '-', label = 'FAM NDVI', linewidth='4')
    plt.legend()
    """leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              fancybox=True, shadow=True, ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(8.0)"""
    print('{} RSME: '.format(country), (((ndvidf['fNDVI'].resample('M').mean() - bfy) ** 2).mean() ** .5))
    plt.show();
    
    
    
"""
possible_region_seasons(bframe, date, seasons, start)

bframe- the artemis dataframe
date - the date of the IPC score you are trying to extract 
start - where does the b-frame start, to prevent key error 
seasons - the dictionay that contains regional growing seasons (from FAO) 

For a given IPC score date, returns list of possible growing seasons for each region
(warning, this def inefficient)
"""
def possible_region_seasons(bframe, date, seasons, start):
    possible_seasons = []
    for index, row in bframe.iterrows():
        max_date = start
        #iter through dict, and months in zone list
        for s in seasons:
            if row[s] == 1:#check if in zone
                for m in seasons[s]:
                    tdate = datetime.date(date.year, int(m), 1)
                    if tdate > max_date and tdate < date:
                        max_date = tdate
                    elif datetime.date(tdate.year-1, tdate.month, tdate.day) > max_date and datetime.date(tdate.year-1, tdate.month, tdate.day) < date:
                        max_date = datetime.date(tdate.year-1, tdate.month, tdate.day)
        possible_seasons.append(max_date)
    return pd.to_datetime(possible_seasons)


"""
 nearest_grow_period(bframe, mframe, cur, date, season_len, seasons, start)

bframe- the artemis dataframe
cur - the temporary df that is being appended to the total feature/ label break down
date - the date of the IPC score you are trying to extract 
start - where does the b-frame start, to prevent key error 
seasons - the dictionay that contains regional growing seasons (from FAO) 
season_len - (how far back from the start point you want to grab ag info from)

For a given IPC score date, returns list of possible growing seasons for each region
(warning, this def inefficient)
"""
def nearest_grow_period(bframe, mframe, cur, date, season_len, seasons, start):
    feats = ['fR', 'fNIR', 'fB', 'fG', 'fSR', 'fNDVI', 'fGNVDI', 'fTVI', 'fSAVI', 'fOSAVI', 'fNLI', 'fRVDI', 'fCARI', 'fPSRI']
    print(date)
    dt_date =  datetime.date(date.year, date.month, date.day)
    start_dates = possible_region_seasons(bframe, dt_date, seasons, start)
    for step in range(0, season_len):
        date_step = start_dates - pd.DateOffset(months=step)
        season = [[ds.year, '{:02}'.format(ds.month)] for ds in date_step] 
        suffix = '.' + str(step+1) + '_m_back'
        
        #need to iterate over regions as well
        for f in feats: cur[f + suffix] = np.nan
        all_feat_avgs = []
        cur['NDVIanom'+suffix] = np.nan
        cur['NDVI'+suffix] = np.nan
        for index, row in bframe.iterrows():
            avgs = modis_monthly_avg(mframe, season[index][0], season[index][1], feats)
            for f in feats: cur.loc[index, f + suffix] = avgs.loc[index,f]
            cur.loc[index, 'NDVIanom'+suffix] = bframe.loc[index, 'NdviAllAnom' + '.{}_{}'.format(season[index][0], season[index][1])]
            cur.loc[index, 'NDVI'+suffix] = bframe.loc[index, 'NdviAll' + '.{}_{}'.format(season[index][0], season[index][1])]
    return cur


# should average each feature over month
#def get_month_avg(mdf):
# currently the modis df might have multiple index entries per month, so averages it down to 
# one number per region 
def modis_monthly_avg(mframe, year, month, feat_list, year_hdr = False):
    avg_df = pd.DataFrame([])
    for feat in feat_list:
        agg_hdr = '{}_{}-{}'.format(feat, year, month)
        col = feat
        if year_hdr: #adds date info to col name, not good for concat
            col = agg_hdr
        avg_df[col] = mframe.filter(like=agg_hdr).mean(axis=1)
    return avg_df


"""
label_season_feature_tables(bframe, mframe, seasons, start, binary, regressor)

bframe - artemis data
mframe - modis data
seasons - season dictionary 
start - when you want to start your searches (index errors thrown by niger and south sudan patch)
binary - 0 if you want all 5 classes, other wise a number 1-4, which basically tells function
where it should be splitting the scores. 0 is not critical, 1 is critical. if binary = 2, 
scores of 1, 2 will be labelled as 0 (not critical) and 3, 4, 5 will be given a class of 1 (critical)
** 2 is recommended if doing a binary split ** 

regressor - if true predicts pct population of 3 and above (no way to adjust class break sorry),
would return a number [0, 1) instead of a class for label 


basically makes the df that will feed the feature label pairs that go into the models... it's gross 
"""
def label_season_feature_tables(bframe, mframe, seasons, start, binary = 0, regressor = False):
    feats = ['fR', 'fNIR', 'fB', 'fG', 'fSR', 'fNDVI', 'fGNVDI', 'fTVI', 'fSAVI', 'fOSAVI', 'fNLI', 'fRVDI', 'fCARI', 'fPSRI']
    phases = get_cols_by_datelist(start, pd.datetime.today(), 'IPC_Phase', bframe)
    #########
    if regressor:
        new_label = pd.DataFrame([])
        for y in phases[1]:
            col1 = ['IPC3_pct' + y, 'IPC4_pct' + y, 'IPC5_pct' + y]
            new_label['IPCfam_pct' + y] = bframe['IPC3_pct' + y] + bframe['IPC4_pct' + y] + bframe['IPC5_pct' + y]
        #temp = bframe[phases[0]]
        bframe[['IPCfam_pct' + y for y in phases[1]]] = new_label
        #display(temp)
    #########
    name = ['acled_fatalities' + d for d in phases[1]]
    bframe[name] = bframe[name].fillna(0)
    name = ['NdviAllAnom' + d for d in phases[1]]
    bframe[name] = bframe[name].fillna(1)#just assume null values are on the average
    bframe = bframe.fillna(0)
    mframe = mframe.fillna(0)
    
    feature_table = pd.DataFrame(columns=['score'])
    region_num = len(bframe.index)
    num_scores = len(phases[2])
    print('Number of regions: ', region_num)
    print('Number of samples: ', num_scores)

    # for each label
    for i in range(1, num_scores): # when 1 and not 0, not using fake_label()
        cur = pd.DataFrame([])
        dates =[phases[2][i]] * region_num
        cur['month'] = [m.month for m in dates]
        cur['month_sin'] = np.sin((cur.month-1)*(2.*np.pi/12))
        cur['month_cos'] = np.cos((cur.month-1)*(2.*np.pi/12))
        cur['score'] = bframe[phases[0][i]] # classifier
        if regressor: cur['score_pct'] = bframe['IPCfam_pct' + phases[1][i]]
        # if choose to genereate a 'fake' label, if range of i starts at 1, it does not happen
        if i == 0:
            cur['prev_score'] = fake_label(cur['score'], .2)
        else:
            cur['prev_score'] = bframe[phases[0][i-1]]
            if regressor: cur['prev_score_pct'] = bframe['IPCfam_pct' + phases[1][i-1]]
            
        col_names_dates = []    
        for step in range(1, 4):
            suffix = '.' + str(step) + '_m_back'
            date_step = phases[2][i] - pd.DateOffset(months=step)
            ys, ms = [date_step.year, '{:02}'.format(date_step.month)]
            cur['acled_fatalities'+suffix] = bframe['acled_fatalities' + '.{}_{}'.format(ys, ms)]
            col_names_dates.extend([f + suffix for f in feats])
            col_names_dates.append('NDVIanom'+suffix)
            col_names_dates.append('NDVI'+suffix)
            
        cur[col_names_dates] = nearest_grow_period(bframe, mframe, pd.DataFrame([]), phases[2][i], 3, seasons, start)
            
        #nearest_grow_period(bframe, mframe, pd.DataFrame([]), phases[2][i], 3)
        feature_table = pd.concat([feature_table, cur])
    
    #get rid of entries with IPC score 0 (or not found) and then 0 out any inf 
    feature_table = feature_table[feature_table.score != 0] #classifier
    feature_table = feature_table.replace([np.inf, -np.inf], np.nan)
    feature_table = feature_table.fillna(0)
    
    #If all visual data from col is 0, then drop col
    drop_cols = ['month', 'score', 'month_sin', 'month_cos', 'prev_score']
    drop_cols.extend(['NDVI.1_m_back', 'NDVI.2_m_back', 'NDVI.3_m_back','NDVIanom.1_m_back', 'NDVIanom.2_m_back', 'NDVIanom.3_m_back', 'acled_fatalities.1_m_back', 'acled_fatalities.2_m_back', 'acled_fatalities.3_m_back'])
    if regressor: drop_cols.extend(['score_pct', 'prev_score_pct'])
    
    feature_table = feature_table[(feature_table.drop(drop_cols, axis=1).T != 0).any()]
    feature_table = feature_table.sort_index()
    
    #split off labels and features 
    label_col = 'score'
    feat_drop = ['score', 'month']
    if regressor: 
        label_col = 'score_pct'
        feat_drop.append(label_col)
    label_table = pd.DataFrame(feature_table[label_col], columns=[label_col])
    
    #if want binary classification 
    if binary != 0 and not regressor:
        label_table = label_table.where(label_table > binary, 0)
        label_table = label_table.where(label_table < binary, 1)
    feature_table.drop(feat_drop, axis=1, inplace=True)

    return label_table, feature_table

# this isn't really used in any main code at this point. just condenses method of dropping rows, 
# used when dividing test and train set by last n years 
# can't just use number of regions * n b/c some data gets dropped due to poor quality so it has to get a bit convoluted 
def row_dropper(table):
    feature_table = table.copy()
    """ display(feature_table.iloc[0])
    print('first elem pre proc: ', feature_table['fB.1_m_back'].iloc[0])
    print('Row Dropper feature table slice: ', feature_table.shape[0])"""
    #drop_cols = ['month', 'score', 'time_delta', 'month_sin', 'month_cos', 'year']
    drop_cols = ['month', 'score', 'month_sin', 'month_cos', 'prev_score']
    feature_table = feature_table[feature_table.score != 0] #classifier
    feature_table = feature_table[(feature_table.drop(drop_cols, axis=1).T != 0).any()]
    label_table = pd.DataFrame(feature_table['score'], columns=['score'])
    feature_table.drop(['score', 'month'], axis=1, inplace=True)
    """ print('feature table post proc', feature_table.shape[0])
    print('first elem post proc: ', feature_table['fB.1_m_back'].iloc[0])
    display(feature_table.iloc[0])"""
    return feature_table, label_table

#basically NEVER used, if you want to make a fake label for prev_ipc score so you can use 
# the first set of IPC scores in your dataset you can, but I never really opted to use it
# because previous score is such an important feature and i didn't want to artificially 
# influence it 
def fake_label(cur_label, prob):
    np.random.seed(904)
    shift = np.random.rand(cur_label.shape[0])
    shift[shift < prob] = -1
    shift[shift > (1-prob*.8)] = 1
    shift[np.abs(shift) < 1 ] = 0
    high_mask = np.where(cur_label > 3, 0, 1)
    
    fake_prev = cur_label + shift*high_mask
    
    fake_prev = np.clip(fake_prev, 1, 5)
    return fake_prev


# functionally similar to label_season_feature_tables, except it doesn't grab modis data by season 
# and it pre divides the data by train and test based on the last few years 
def label_feature_tables_split(bframe, mframe, num_test):
    feats = ['fR', 'fNIR', 'fB', 'fG', 'fSR', 'fNDVI', 'fGNVDI', 'fTVI', 'fSAVI', 'fOSAVI', 'fNLI', 'fRVDI', 'fCARI', 'fPSRI']
    
    phases = get_cols_by_datelist('1/1/2010', '3/22/2017', 'IPC_Phase', bframe)
    bframe = bframe.fillna(0)
    mframe = mframe.fillna(0)
    region_num = len(bframe.index)
    num_scores = len(phases[2])
    feature_table = pd.DataFrame(columns=['score'])
    print('Total number of labels: ', num_scores)
    
    for i in range(1, num_scores):
        print(i)
        cur = pd.DataFrame([])
        dates =[phases[2][i]] * region_num
        cur['score'] = bframe[phases[0][i]] # classifier
        if i == 0:
            cur['prev_score'] = fake_label(cur['score'], .2)
            #display(cur[['score','prev_score']])
        else:
            cur['prev_score'] = bframe[phases[0][i-1]]
        cur['month'] = [m.month for m in dates]
        #cur['time_delta'] = [(d - pd.datetime(2010,1,1)).days for d in dates]
        cur['month_sin'] = np.sin((cur.month-1)*(2.*np.pi/12))
        cur['month_cos'] = np.cos((cur.month-1)*(2.*np.pi/12))
        #cur['year'] = [y.year for y in dates]

        
        for step in range(1, 4):
            suffix = '.' + str(step) + '_m_back'
            date_step = phases[2][i] - pd.DateOffset(months=step)
            ys, ms = [date_step.year, '{:02}'.format(date_step.month)]
            
            col_names_dates = [f + suffix for f in feats]
            all_feat_avgs = modis_monthly_avg(mframe, ys, ms, feats)
            cur[col_names_dates] = all_feat_avgs

        
        feature_table = pd.concat([feature_table, cur])
        
    region_num = len(bframe.index)
    num_scores = len(phases[2])
    
    
    feature_table = feature_table.replace([np.inf, -np.inf], np.nan)
    feature_table = feature_table.fillna(0)

    print('pre drop total rows: ', (feature_table.shape[0]))
     
    train_feature, train_label = row_dropper(feature_table.iloc[0:region_num*(num_scores - num_test-1)])
    test_feature, test_label = row_dropper(feature_table.iloc[-1*region_num*num_test:])
    
   
    """#Truly painstaking verificaiton that the code is correct
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):
        display(feature_table)
        #display(train_feature)
        #display(test_feature)"""
 
    print('Number of test score periods: ', np.ceil(test_label.shape[0]/region_num))
    print('Number of train score periods: ', np.ceil(train_label.shape[0]/region_num))
    print('Number of test score rows: ', (test_label.shape[0]))
    print('Number of train score row: ', (train_label.shape[0]))
    
    
    ft, lt = row_dropper(feature_table)
    #print('table cols: \n{} \nfeat: \n{} \nlabels\n{}'.format(feature_table.columns.values, ft.columns.values, lt.columns.values))
    print('Post drop total rows: ', (ft.shape[0]))
    return train_label, train_feature, test_label, test_feature


# exactly as it sounds, this is how I format the confusion matrix to display 
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, title = '', percentage = True):
    fmt = 'd'
    if (percentage):
        confusion_matrix = np.round(confusion_matrix / confusion_matrix.sum(), 2)
        fmt = '.2f'
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    #fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)
    """try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")"""
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix '+ title)
    plt.show();
    return fig


"""
class_pre_re(class_label, cm)

gives the prec and recall of the given label (IPC score) from a confusion matrix. 
"""
def class_pre_re(class_label, cm):

    class_label = class_label - 1
    #print('pre: ', cm[:, class_label])
    #print('re: ', cm[class_label])
    pre = cm[class_label][class_label]/(cm[:, class_label].sum())
    re = cm[class_label][class_label]/(cm[class_label].sum())
    return pre, re


"""
NOTE IMPORTANT: This is only used assuming that the features used are pre-pickled 
Otherwise compy and paste code in between hashes 
build_super_model_test_train_array(label_code, features_to_drop)

label_code - 
"""
def build_super_model_test_train_array(label_code, features_to_drop = []):
    afg_label_table, afg_feature_table = [None, None]
    somalia_label_table, somalia_feature_table = [None, None]
    niger_label_table, niger_feature_table = [None, None]
    mali_label_table, mali_feature_table = [None, None]
    southsudan_label_table, southsudan_feature_table = [None, None]
    if label_code == 'BIN':
        # binary data set 
        print("LABEL TYPE: Binary")
        afg_label_table = pd.read_pickle('modis_data/afg_bin_labels')
        afg_feature_table = pd.read_pickle('modis_data/afg_bin_features')
        somalia_label_table = pd.read_pickle('modis_data/somalia_bin_labels')
        somalia_feature_table = pd.read_pickle('modis_data/somalia_bin_features')
        niger_label_table = pd.read_pickle('modis_data/niger_bin_labels')
        niger_feature_table = pd.read_pickle('modis_data/niger_bin_features')
        mali_label_table = pd.read_pickle('modis_data/mali_bin_labels')
        mali_feature_table = pd.read_pickle('modis_data/mali_bin_features')
        southsudan_label_table = pd.read_pickle('modis_data/southsudan_bin_labels')
        southsudan_feature_table = pd.read_pickle('modis_data/southsudan_bin_features')
        
    elif label_code == 'REG':
        # regressor data set 
        print("LABEL TYPE: Continuous / regressor")
        afg_label_table = pd.read_pickle('modis_data/afg_reg_labels')
        afg_feature_table = pd.read_pickle('modis_data/afg_reg_features')
        somalia_label_table = pd.read_pickle('modis_data/somalia_reg_labels')
        somalia_feature_table = pd.read_pickle('modis_data/somalia_reg_features')
        niger_label_table = pd.read_pickle('modis_data/niger_reg_labels')
        niger_feature_table = pd.read_pickle('modis_data/niger_reg_features')
        mali_label_table = pd.read_pickle('modis_data/mali_reg_labels')
        mali_feature_table = pd.read_pickle('modis_data/mali_reg_features')
        southsudan_label_table = pd.read_pickle('modis_data/southsudan_reg_labels')
        southsudan_feature_table = pd.read_pickle('modis_data/southsudan_reg_features')
    else:
        # multi class classification dataset 
        if label_code != 'MULT':
            print ("INCORRECT LABEL TYPE GIVEN: \nMulticlass labels chosen as defult")
        else:
            print("LABEL TYPE: Multiclass") 
        afg_label_table = pd.read_pickle('modis_data/afg_mult_labels')
        afg_feature_table = pd.read_pickle('modis_data/afg_mult_features')
        somalia_label_table = pd.read_pickle('modis_data/somalia_mult_labels')
        somalia_feature_table = pd.read_pickle('modis_data/somalia_mult_features')
        niger_label_table = pd.read_pickle('modis_data/niger_mult_labels')
        niger_feature_table = pd.read_pickle('modis_data/niger_mult_features')
        mali_label_table = pd.read_pickle('modis_data/mali_mult_labels')
        mali_feature_table = pd.read_pickle('modis_data/mali_mult_features')
        southsudan_label_table = pd.read_pickle('modis_data/southsudan_mult_labels')
        southsudan_feature_table = pd.read_pickle('modis_data/southsudan_mult_features')
    #####If Not Pickled Copy and Paste This Below#####    
    somalia_feature_table_d = somalia_feature_table.drop(features_to_drop, axis=1)
    mali_feature_table_d = mali_feature_table.drop(features_to_drop, axis=1)
    niger_feature_table_d = niger_feature_table.drop(features_to_drop, axis=1)
    afg_feature_table_d = afg_feature_table.drop(features_to_drop, axis=1)
    southsudan_feature_table_d = southsudan_feature_table.drop(features_to_drop, axis=1)

    somalia_labels = np.array(somalia_label_table)
    somalia_feature_list = list(somalia_feature_table_d.columns)
    somalia_features = np.array(somalia_feature_table_d)
    somalia_train_features, somalia_test_features, somalia_train_labels, somalia_test_labels = train_test_split(somalia_features, somalia_labels, test_size = 0.25, random_state = 42)

    mali_labels = np.array(mali_label_table)
    mali_feature_list = list(mali_feature_table_d.columns)
    mali_features = np.array(mali_feature_table_d)
    mali_train_features, mali_test_features, mali_train_labels, mali_test_labels = train_test_split(mali_features, mali_labels, test_size = 0.25, random_state = 42)

    afg_labels = np.array(afg_label_table)
    afg_feature_list = list(afg_feature_table_d.columns)
    afg_features = np.array(afg_feature_table_d)
    afg_train_features, afg_test_features, afg_train_labels, afg_test_labels = train_test_split(afg_features, afg_labels, test_size = 0.25, random_state = 42)

    niger_labels = np.array(niger_label_table)
    niger_feature_list = list(niger_feature_table_d.columns)
    niger_features = np.array(niger_feature_table_d)
    niger_train_features, niger_test_features, niger_train_labels, niger_test_labels = train_test_split(niger_features, niger_labels, test_size = 0.25, random_state = 42)

    southsudan_labels = np.array(southsudan_label_table)
    southsudan_feature_list = list(southsudan_feature_table_d.columns)
    southsudan_features = np.array(southsudan_feature_table_d)
    southsudan_train_features, southsudan_test_features, southsudan_train_labels, southsudan_test_labels = train_test_split(southsudan_features, southsudan_labels, test_size = 0.25, random_state = 42)

    train_features = np.append(somalia_train_features, niger_train_features, axis=0)
    train_features = np.append(train_features, mali_train_features, axis=0)
    train_features = np.append(train_features, afg_train_features, axis=0)
    train_features = np.append(train_features, southsudan_train_features, axis=0)                          

    test_features = np.append(somalia_test_features, niger_test_features, axis=0)
    test_features = np.append(test_features, mali_test_features,axis=0)
    test_features = np.append(test_features, afg_test_features,axis=0)
    test_features = np.append(test_features, southsudan_test_features,axis=0)

    train_labels = np.append(somalia_train_labels, niger_train_labels, axis=0)
    train_labels = np.append(train_labels, mali_train_labels, axis=0)
    train_labels = np.append(train_labels, afg_train_labels, axis=0)
    train_labels = np.append(train_labels, southsudan_train_labels, axis=0)

    test_labels = np.append(somalia_test_labels, niger_test_labels, axis=0)
    test_labels = np.append(test_labels, mali_test_labels, axis=0)
    test_labels = np.append(test_labels, afg_test_labels, axis=0)
    test_labels = np.append(test_labels, southsudan_test_labels, axis=0)

    feature_list = somalia_feature_list
    #####If Not Pickled Copy and Paste This Above##### 
    
    return train_features, train_labels, test_features, test_labels, feature_list


#for regressor
# Create linear regression object
"""The coefficient R^2 is defined as (1 - u/v), where u is the 
residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the 
total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best 
possible score is 1.0 and it can be negative (because the model can be 
arbitrarily worse)"""
# baseline is a regression that uses the past percent to predict current percent
def LinReg_baseline(train_features, train_labels, test_features, test_labels):
    regr = LinearRegression()
    trf =train_features[:, np.newaxis,train_features.shape[1]-1]
    regr.fit(trf, train_labels)

    # Make predictions using the testing set
    y_pred = regr.predict(test_features[:, np.newaxis,train_features.shape[1]-1])
    y_pred_train = regr.predict(trf)
    #TRAIN EVAL
    print('Coefficients: \n', regr.coef_)
    
    print("Train Mean squared error: %.4f"
          % mean_squared_error(train_labels, y_pred_train))
    # Explained variance score: 1 is perfect prediction
    print('Train Variance score (r2): %.4f' % r2_score(train_labels, y_pred_train))
    
    #TEST EVAL
    print("Test Mean squared error: %.4f"
          % mean_squared_error(test_labels, y_pred))
    
    print("Test Abd squared error: %.4f"
          % mean_absolute_error(test_labels, y_pred))
    
    # Explained variance score: 1 is perfect prediction
    print('Test Variance score (r2): %.4f' % r2_score(test_labels, y_pred))
    # Plot outputs

    plt.scatter(test_features[:, np.newaxis,train_features.shape[1]-1], test_labels,  color='blue')
    plt.plot(test_features[:, np.newaxis,train_features.shape[1]-1], y_pred, color='red', linewidth=3)

    
    plt.xticks(())
    plt.yticks(())

    plt.show()
    
    
#Classifier Baseline 
# The baseline predictions get the mode of the train set and predict that for the test
def Mode_Baseline(train_labels, test_labels, classes):
    baseline_preds = np.ones(test_labels.shape)
    mode = stats.mode(train_labels)[0][0]
    baseline_preds = (baseline_preds*mode).astype(int)
    test_acc = accuracy_score(test_labels.astype(int), baseline_preds.astype(int))
    test_f1 = f1_score(test_labels.astype(int), baseline_preds.astype(int), average='macro')
    test_f1w = f1_score(test_labels.astype(int), baseline_preds.astype(int), average='weighted')

    cm = confusion_matrix(test_labels.astype(int), baseline_preds.astype(int), labels=np.arange(classes[0], classes[1]))
    print_confusion_matrix(cm, np.arange(classes[0], classes[1]), title = 'Mode Baseline', percentage = False)
    baseline_preds = np.ones(train_labels.shape)
    baseline_preds = (baseline_preds*mode).astype(int)
    train_acc = accuracy_score(train_labels.astype(int), baseline_preds.astype(int))
    train_f1 = f1_score(train_labels.astype(int), baseline_preds.astype(int), average='macro')
    train_f1w = f1_score(train_labels.astype(int), baseline_preds.astype(int), average='weighted')

    print('train baseline acc: ', train_acc)
    print('test baseline acc: ', test_acc)
    print('train baseline f1: ', train_f1)
    print('test baseline f1: ', test_f1)
    print('train baseline f1 weighted: ', train_f1w)
    print('test baseline f1: weighted', test_f1w)
    
#Classifier Baseline Persistance
# The baseline predictions use previous IPC score as prediciton for current IPC score
# n can be used if you want to test over the last n years. if n = 0, then test over all years
def pers_baseline_F1(bframe, binary, n=0):
    phase_base = get_cols_by_datelist('1/1/2010', '3/22/2017', 'IPC_Phase', bframe)
    num_scores = len(phase_base[2])
    if n == 0: n = num_scores - 1
    start = num_scores - n - 1
    base_df = bframe[phase_base[0]]
    if binary:
        base_df = base_df.where(base_df > 2, 0)
        base_df = base_df.where(base_df < 2, 1)
    correct = 0
    pred_pers = np.array([])
    label_pers = np.array([])
    for i in range(start, len(phase_base[0]) - 1):
        pred_pers = np.append(pred_pers, base_df[phase_base[0][i]].values)
        label_pers = np.append(label_pers, base_df[phase_base[0][i+1]].values)
    print('persistance test acc: ', accuracy_score(label_pers.astype(int), pred_pers.astype(int)))
    print('persistance test f1: ', f1_score(label_pers.astype(int), pred_pers.astype(int), average='macro'))
    return label_pers.astype(int), pred_pers.astype(int)

# wrapper function that calls the pers baseline for all countries. 
def all_country_persistance_wrapper(afg_bframe, somalia_bframe, mali_bframe, niger_bframe, southsudan_bframe, classes, n = 0):
    binary = True
    if classes[0]:
        binary = False
    c1, b1 = pers_baseline_F1(afg_bframe, binary, n)
    c2, b2 = pers_baseline_F1(somalia_bframe, binary, n)
    c3, b3 = pers_baseline_F1(mali_bframe, binary, n)
    c4, b4 = pers_baseline_F1(niger_bframe, binary, n)
    c5, b5 = pers_baseline_F1(southsudan_bframe, binary, n)
    c = np.append(np.append(np.append(np.append(c1,c2),c3),c4),c5)
    b = np.append(np.append(np.append(np.append(b1,b2),b3),b4),b5)
    cm = confusion_matrix(b, c, labels=np.arange(classes[0], classes[1]))
    print_confusion_matrix(cm, np.arange(classes[0], classes[1]), title = 'Persistence Baseline', percentage = False)
    print('Overall acc: ', accuracy_score(c, b))
    print('Overall f1: ', f1_score(c, b, average='macro'))
    print('Overall f1 w: ', f1_score(c, b, average='weighted'))
    #good output below for quick copy and paste to spread sheet
    print('{},{},{}'.format('accuracy','f1','f1_weighted'))
    print('{},{},{}'.format(accuracy_score(c, b),f1_score(c, b, average='macro'),f1_score(c, b, average='weighted')))


"""
regressor_train_test(regressor, title = '')
title - The name of the regressor, makes print statements and copying and 
pasting into spread sheets cleaner
regressor - Enter in the type of regressor, with its hyperparameters alread set

Use: Fits and runs predictions, gets all the evaluation metrics 
Currently it plots nonsense, scatter of labels vs preds and a line of labels v labels
"""
def regressor_train_test(regressor, train_features, train_labels, test_features, test_labels, title = ''):
    # Train the model on training data
    regressor.fit(train_features, train_labels)

    # Use the forest's predict method on the test and train data
    test_pred = regressor.predict(test_features)
    train_pred = regressor.predict(train_features)
    
    #TRAIN EVAL
    train_eval = [mean_squared_error(train_labels, train_pred),
                  mean_absolute_error(train_labels, train_pred),
                  r2_score(train_labels, train_pred),
                  explained_variance_score(train_labels, train_pred)]
    print("Train Mean squared error: %.4f" % train_eval[0])
    # Explained variance score: 1 is perfect prediction
    print('Train Absolute error: %.4f' % train_eval[1])
    print('Train Variance score (r2): %.4f' % train_eval[2])
    print('Train Explained Variance score (r2): %.4f' % train_eval[3])
    
    #TEST EVAL
    test_eval = [mean_squared_error(test_labels, test_pred),
                  mean_absolute_error(test_labels, test_pred),
                  r2_score(test_labels, test_pred),
                  explained_variance_score(test_labels, test_pred)]
    print("Test Mean squared error: %.4f" % test_eval[0])
    # Explained variance score: 1 is perfect prediction
    print('Test Absolute error: %.4f' % test_eval[1])
    print('Test Variance score (r2): %.4f' % test_eval[2])
    print('Test Explained Variance score (r2): %.4f' % test_eval[3])
    
    
    # Plot outputs
    print('{},{},{},{},{}'.format(title+' Best', 'MSE','MAE', 'R2', 'Exp Var'))
    print('{},{},{},{},{}'.format(title+' Train',train_eval[0],train_eval[1],train_eval[2],train_eval[3]))
    print('{},{},{},{},{}'.format(title+' Test', test_eval[0],test_eval[1],test_eval[2],test_eval[3]))
    plt.scatter(test_labels, test_pred,  color='blue')
    plt.plot(test_labels, test_labels, color='red', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.show();
    

"""
classifier_train_test(classifier, title = '')
title - The name of the classifier, makes print statements and copying and 
pasting into spread sheets cleaner
classifier - Enter in the type of classifier, with its hyperparameters alread set
classes - the range number of classes (0, 2) for binary and (1, 6) for multiclass 

Use: Fits and runs predictions, gets all the evaluation metrics, prints confusion matrix
"""
def classifier_train_test(classifier, train_features, train_labels, test_features, test_labels, classes, title = ''):
    # Train the model on training data
    classifier.fit(train_features, train_labels.astype(int));

    # Use the forest's predict method on the test data
    test_pred = classifier.predict(test_features)
    # Calculate the absolute errors

    cm = confusion_matrix(test_labels.astype(int), test_pred.astype(int), labels=np.arange(classes[0],classes[1]))
    test_acc = accuracy_score(test_labels.astype(int), test_pred.astype(int))
    test_f1 = f1_score(test_labels.astype(int), test_pred.astype(int), average='macro')
    test_f1w = f1_score(test_labels.astype(int), test_pred.astype(int), average='weighted')

    print_confusion_matrix(cm, np.arange(classes[0],classes[1]), title=title, percentage = False)
    #print('test pr, re 4: ',class_pre_re(4, cm))
    #print('test pr, re 5: ', class_pre_re(5, cm))
    #print('test acc: ', test_acc)

    # Use the forest's predict method on the train data
    train_pred = classifier.predict(train_features)
    # Calculate the absolute errors

    #cm_train = confusion_matrix(train_labels.astype(int), train_pred_rf.astype(int), labels=np.arange(classes[0],classes[1]))
    train_acc = accuracy_score(train_labels.astype(int), train_pred.astype(int))
    train_f1 = f1_score(train_labels.astype(int), train_pred.astype(int), average='macro')
    train_f1w = f1_score(train_labels.astype(int), train_pred.astype(int), average='weighted')
    #print_confusion_matrix(cm_train, np.arange(classes[0],classes[1]))
    print('')
    #print('train pr, re 4: ',class_pre_re(4, cm))
    #print('train pr, re 5: ', class_pre_re(5, cm))
    print('train acc: ', train_acc)
    print('train f1: ', train_f1)
    print('train f1 weighted: ', train_f1w)
    print('')
    print('test acc: ', test_acc)
    print('test f1: ', test_f1)
    print('test f1: weighted', test_f1w)

    print('{},{},{},{}'.format(title+' Best', 'F1','F1 Weighted', 'Accuracy'))
    print('{},{},{},{}'.format(title+' Train',train_acc,train_f1,train_f1w))
    print('{},{},{},{}'.format(title+' Test', test_acc,test_f1,test_f1w))
    

    
"""
hyperparam_search(parameter_candidates, scoring, classifier, print_full)

parameter_candidates - a dictionary of the hyperparemeters that you want to test, 
with the key being the hyperparameter and the value being and array of the values 
to be tested in grid search 
scoring - an ARRAY (must be iterable array even if only testing 1 scoring function) 
the search will get the values that each combonation of parameters give based 
on these scoring functions. 
IMPORTANT: the first element in the array is the primary scoring that the 
hyperparameter seach orders on. Therefore put the metric you are most interested in there.
The "Best Model" will be based on this score. 
classifier - the initialized classifier enter in an hyperparameters that you are not testing
(like random seed) prior to inputting in function
print_full - None prints full table, any other value restricts the number of rows to 
that given value.

Use: runs cross validation hyper parameter tuning grid search 
"""  
def hyperparam_search(train_features, train_labels, test_features, test_labels, parameter_candidates, scoring, classifier, rows_disp = None):
    print('parameter_candidates: ',parameter_candidates)
    print('scoring: ', scoring)

    gs = GridSearchCV(estimator=classifier, param_grid=parameter_candidates, n_jobs=-1, scoring=scoring, refit=scoring[0] )
    gs.fit(train_features, train_labels.astype(int)) 
    results = pd.DataFrame(gs.cv_results_ )
    print('Best {} score for Train:'.format(scoring[0]), gs.best_score_) 
    for pc in gs.best_params_:
        print('Best {} for Train:'.format(pc), gs.best_params_[pc])

    metrics = ['rank_test_' + s for s in scoring] 
    metrics.extend(['mean_test_'+ s for s in scoring])
    for pc in parameter_candidates:
        metrics.extend(['param_' + p for p in list(pc.keys())])
    metrics.append('params')
    # Streamlined table to easily compare performances accross different metrics
    with pd.option_context('display.max_rows', rows_disp, 'display.max_columns', None):
        display(results[metrics].sort_values(by=['rank_test_' + scoring[0]]).drop('params', axis=1))
  
        
    return results


"""
get_RF_variable_importance(rf, feature_list, title)

rf - the random forest that you want to perform feature analysis on, 
can be classifier or regressor
feature_list - list of feature names, should be returned by 
build_super_model_test_train_array() if used, otherwise must be taken from column names
output by label_season_feature_tables() feature table.

Use: get the variable importance (or average order of tree splits) from random forest
"""  
def get_RF_variable_importance(rf, feature_list, title = ''):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    #plt.figure(figsize=(20,10))
    #plt.rcParams.update({'font.size': 22})
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances '+title);
    plt.show();
