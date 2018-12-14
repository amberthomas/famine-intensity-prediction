"""
Helper functions to visualize data from version 8 of the acled website.

"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np
import seaborn as sns
import copy

#index by time stamp, needed to resample the data into month long chunks, only for data from website
def index_by_event_date(acled_frame):
    print(acled_frame.event_date[0:5])
    acled_frame['event_date'] = pd.to_datetime(acled_frame['event_date'], format='%d-%b-%y', utc=True)
    print('month code: {}'.format(acled_frame.event_date[0].month))
    acled_frame.index = (acled_frame.event_date)
    
    
    
def plot_fatalities_by_events(acled_frame, country):
    event_grps = acled_frame[acled_frame['fatalities'] > 0].groupby('event_type')
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 22})
    plt.xticks(rotation=70)
    title_str = 'Fatalities by Event Type in ' + country
    plt.title(title_str, fontweight ='bold')
    for name, data in event_grps:
        plt.plot(data.fatalities.resample('M').sum(), '-', label = name, linewidth='4')
        leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, shadow=True, ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(8.0)
        
        
        
def event_counts(acled_frame, title_str, group_name = '', min_fatalities = 0):
    event_frame = acled_frame[acled_frame['fatalities'] > min_fatalities]
    if group_name == '':
        event_grps = event_frame.groupby('country')
    else:
        event_grps = event_frame.groupby(group_name)
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 22})
    plt.xticks(rotation=70)
    plt.title(title_str, fontweight ='bold')
    for name, data in event_grps:
        if group_name == '':
            name = 'Events'
        plt.plot(data.year.resample('M').count(), '-', label = name, linewidth='4')
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            fancybox=True, shadow=True, ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(8.0)
        
        
def actor_type(acled_frame, country):
    inter1_grps = acled_frame.groupby('inter1')
    inter2_grps = acled_frame.groupby('inter2')
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 22})
    plt.xticks(rotation=70)
    plt.title('Actor Type in Event: ' + country, fontweight ='bold')
    actor_codes = ['Governments and State Security Services',
              'Rebel Groups',
              'Political Militias',
              'Identity Militias',
              'Rioters',
              'Protesters',
              'Civilians',
              'External/Other Forces']
    keys1 = acled_frame['inter1'].unique()
    keys2 = acled_frame['inter2'].unique()
    for code in range(0,8):
        if (code + 1) in keys1:
            actor_data1 = inter1_grps['inter1'].get_group(code + 1)
        if (code + 1) in keys2:
            actor_data2 = inter2_grps['inter2'].get_group(code + 1)
        plt.plot(actor_data1.resample('M').count() + actor_data2.resample('M').count(), '-', label = actor_codes[code], linewidth='4')
    plt.plot(acled_frame.data_id.resample('M').count(), '-', label = 'Total Num Events', linewidth='4')
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, shadow=True, ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(8.0)
        
        
        
        
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


def website_vs_bucket_fatalities(acled_frame, bucket_frame, country):
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 22})
    plt.xticks(rotation=70)
    plt.title('Fatality Totals in {} both data set'.format(country), fontweight ='bold')
    plt.plot(acled_frame['fatalities'].resample('M').sum(), '-', label = 'acled_website', linewidth='4')
    bfy = []
    bfx = []
    col, month, dates = get_cols_by_datelist('1/1/2012', pd.datetime.today(), 'acled_fatalities', bucket_frame,)
    for i in range(0, len(col)):
        bfy.append(bucket_frame[col[i]].sum())
        bfx.append(dates[i])

    plt.plot(bfx, bfy, '-', label = 'acled_bucket', linewidth='4')
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              fancybox=True, shadow=True, ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(8.0)
        
        
        
def correlation_cols(bucket_frame, pre1, pre2, log=False):
    col_phase, month, dates = get_cols_by_datelist('1/1/2012', pd.datetime.today(), pre1, bucket_frame)
    bPx = pd.DataFrame([])
    bPy = pd.DataFrame([])
    new_cols = copy.deepcopy(month)
    for m in month:
        if pre2 + m not in bucket_frame:
            new_cols.remove(m)
            continue
        if (bucket_frame[pre2 + m].count() > 0):
            bPx[pre1 + m] = bucket_frame[pre1 + m]
            bPy[pre2 + m] = bucket_frame[pre2 + m]
        else:
            new_cols.remove(m)
    if not new_cols: return
    display(bPy.describe())
    
    print("number of Nan in " + pre2)
    display(bPy.isna().sum())
    bPx.columns = new_cols
    bPy.columns = new_cols

    if log:
        bPy = bPy.apply(np.log)
        
    month_corr = bPx.corrwith(bPy)
    
    #print(type(month_corr))
    print("Correlation between {} and {} by month".format(pre1, pre2))
    df_month_corr = (pd.DataFrame(data=month_corr, index = month_corr.index, columns=["Pearson's"]))
    x = pd.concat([bPx[col] for col in bPx])
    y = pd.concat([bPy[col] for col in bPy])
    total_corr = x.corr(y)
    print("Correlation between {} and {} all months: \n{}".format(pre1, pre2, total_corr))
    df_total_corr = (pd.DataFrame(data=total_corr, index = ['All Months'], columns=["Pearson's"]))
    display(pd.concat([df_month_corr, df_total_corr]))
    

    max_month = month_corr.abs().idxmax(axis=0)
    print(max_month)
    
    #ax = sns.regplot(norm_x[max_month], norm_y[max_month], label=pre2 + 'max_month')
    ax = sns.regplot(x, y, label=pre2)
    ax.set(xlabel=pre1)
    leg = plt.legend(loc='upper center', bbox_to_anchor=(1.4, 1),
    fancybox=True, shadow=True, ncol=1)
    
def IPC_pct_corr_plots(bucket_frame):
    pre2 = 'acled_fatalities'
    pre3 = 'RainAll'
    pre4 = 'NdviCrop'
    for i in range(1, 6):
        pre1 = 'IPC{}_pct'.format(i)
        fig = plt.figure(i)
        #correlation_cols(bucket_frame, pre1, pre2)
        #correlation_cols(bucket_frame, pre1, pre3)
        correlation_cols(bucket_frame, pre1, pre4)
        #fig.savefig('plots/'+pre1+'-'+pre2+'-bestfit.png')
        
        
def score_freq(bucket_frame, country):
    phase_scores, m, d = get_cols_by_datelist('1/1/10', pd.datetime.today(), 'IPC_Phase', bucket_frame)
    df = pd.DataFrame([])
    fig, ax = plt.subplots()
    for c in phase_scores:
        df[c] = bucket_frame[c]
    s = pd.concat([df[col] for col in df])
    scores = s.value_counts()
    scores.plot(ax=ax, kind='bar')
    ax.set(title='IPC Score freq. ' + country)
    print(scores)


