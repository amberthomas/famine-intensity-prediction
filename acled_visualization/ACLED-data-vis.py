#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import numpy as np
import seaborn as sns
import copy
from IPython.display import display

# In[10]:


from acled_vis import *


# In[11]:


frame = pd.read_csv("acled_visualization/01-2017-10-2018-afghanistan.csv")
b_frame = pd.read_csv('acled_visualization/afghanistan.csv')


# In[12]:


mali_bframe = pd.read_csv('acled_visualization/mali.csv')
mali_aframe = pd.read_csv('acled_visualization/01-2010-10-2018-mali.csv')
niger_bframe = pd.read_csv('acled_visualization/niger.csv')
niger_aframe = pd.read_csv('acled_visualization/01-2010-10-2018-niger.csv')
somalia_bframe = pd.read_csv('acled_visualization/somalia.csv')
somalia_aframe = pd.read_csv('acled_visualization/01-2010-10-2018-somalia.csv')
southsudan_bframe = pd.read_csv('acled_visualization/southsudan.csv')
southsudan_aframe = pd.read_csv('acled_visualization/01-2010-10-2018-southsudan.csv')


# In[13]:



#get groups by admin zone and list of zone names
admin1_acled_g = frame.groupby('admin1')
admin1_acled = list(admin1_acled_g.groups.keys())
admin1_bucket_g = b_frame.groupby('admin1_name')
admin1_bucket = list(admin1_bucket_g.groups.keys())

#print the suspected typo in the website download
print('num bucket: {}, num acled: {}'.format(admin1_bucket_g.ngroups, admin1_acled_g.ngroups))
print('count of Ghanzi entries: {}, count of Ghazni entries: {}'.format(admin1_acled_g.event_id_cnty.get_group('Ghanzi').count(), admin1_acled_g.event_id_cnty.get_group('Ghazni').count()))
admin1_acled.remove("Ghanzi")
#print comparison list without typo, note the spelling discrepancies
admin_codes = []
for i in range(0, 34):
    print("acled: {}, bucket: {}".format(admin1_acled[i], admin1_bucket[i]))
    code = list(admin1_bucket_g.admin1_code.get_group(admin1_bucket[i]))
    admin_codes.extend(code)


#fix the spellings (change website to match bucket)
frame.admin1[(frame['admin1'] == 'Helmand')] = 'Hilmand'
frame.admin1[(frame['admin1'] == 'Herat')] = 'Hirat'
frame.admin1[(frame['admin1'] == 'Jowzjan')] = 'Jawzjan'
frame.admin1[(frame['admin1'] == 'Nimruz')] = 'Nimroz'
frame.admin1[(frame['admin1'] == 'Paktia')] = 'Paktya'
frame.admin1[(frame['admin1'] == 'Panjshir')] = 'Panjsher'
frame.admin1[(frame['admin1'] == 'Sar-e Pol')] = 'Sari pul'
frame.admin1[(frame['admin1'] == 'Urozgan')] = 'Uruzgan'
frame.admin1[(frame['admin1'] == 'Ghanzi')] = 'Ghazni'

# need to regroup frames after adjustment 
admin1_acled_g = frame.groupby('admin1')
admin1_acled = list(admin1_acled_g.groups.keys())
admin1_bucket_g = b_frame.groupby('admin1_name')
admin1_bucket = list(admin1_bucket_g.groups.keys())

#verify that the regions have been fixed
#print('num bucket: {}, num acled: {}'.format(admin1_bucket_g.ngroups, admin1_acled_g.ngroups))
admin_codes = []
"""for i in range(0, 34):
    print("acled: {}, bucket: {}, matches: {}".format(admin1_acled[i], admin1_bucket[i], admin1_acled[i] == admin1_bucket[i]))"""


# In[14]:


print('afghanistan')
index_by_event_date(frame)
print('mali')
index_by_event_date(mali_aframe)
print('niger')
index_by_event_date(niger_aframe)
print('somalia')
index_by_event_date(somalia_aframe)
print('southsudan')
index_by_event_date(southsudan_aframe)


# In[15]:


#display(frame)


# In[16]:


print('Afghanistan: Num events with no fatalities.')
display((frame['fatalities']==0).astype(int).sum(axis=0))
print('Mali: Num events with no fatalities.')
display((mali_aframe['fatalities']==0).astype(int).sum(axis=0))
print('Niger: Num events with no fatalities.')
display((niger_aframe['fatalities']==0).astype(int).sum(axis=0))
print('Somalia: Num events with no fatalities.')
display((somalia_aframe['fatalities']==0).astype(int).sum(axis=0))
print('South Sudan: Num events with no fatalities.')
display((southsudan_aframe['fatalities']==0).astype(int).sum(axis=0))


# In[17]:


plot_fatalities_by_events(frame, 'Afghanistan')
title_str = 'Number of Events in Afghanistan'
event_counts(frame, title_str, 'event_type')
title_str = 'Number of Events w/ fatalities > 10 in Afghanistan'
event_counts(frame, title_str, 'event_type', 10)
title_str = 'Total Events'
event_counts(frame, title_str)
actor_type(frame, 'Afghanistan')
website_vs_bucket_fatalities(frame, b_frame, 'Afghanistan')


# In[18]:


print('AFGHANISTAN: Correlation between Percent pop in given IPC and NDVI.')
IPC_pct_corr_plots(b_frame)


# In[19]:


plot_fatalities_by_events(mali_aframe, 'Mali')
title_str = 'Number of Events in Mali'
event_counts(mali_aframe, title_str, 'event_type')
title_str = 'Number of Events w/ fatalities > 10 in Mali'
event_counts(mali_aframe, title_str, 'event_type', 10)
title_str = 'Total Events in Mali'
event_counts(mali_aframe, title_str)
actor_type(mali_aframe, 'Mali')
website_vs_bucket_fatalities(mali_aframe, mali_bframe, 'Mali')


# In[20]:


print('MALI: Correlation between Percent pop in given IPC and NDVI.')
IPC_pct_corr_plots(mali_bframe)


# In[21]:


plot_fatalities_by_events(niger_aframe, 'Niger')
title_str = 'Number of Events in Niger'
event_counts(niger_aframe, title_str, 'event_type')
title_str = 'Number of Events w/ fatalities > 10 in Niger'
event_counts(niger_aframe, title_str, 'event_type', 10)
title_str = 'Total Events in Niger'
event_counts(niger_aframe, title_str)
actor_type(niger_aframe, 'Niger')
website_vs_bucket_fatalities(niger_aframe, niger_bframe, 'Niger')


# In[22]:


print('NIGER: Correlation between Percent pop in given IPC and NDVI.')
IPC_pct_corr_plots(niger_bframe)


# In[23]:


plot_fatalities_by_events(somalia_aframe, 'Somalia')
title_str = 'Number of Events in Somalia'
event_counts(somalia_aframe, title_str, 'event_type')
title_str = 'Number of Events w/ fatalities > 10 in Somalia'
event_counts(somalia_aframe, title_str, 'event_type', 10)
title_str = 'Total Events in Somalia'
event_counts(somalia_aframe, title_str)
actor_type(somalia_aframe, 'Somalia')
website_vs_bucket_fatalities(somalia_aframe, somalia_bframe, 'Somalia')


# In[24]:


print('SOMALIA: Correlation between Percent pop in given IPC and NDVI.')
IPC_pct_corr_plots(somalia_bframe)


# In[25]:


plot_fatalities_by_events(southsudan_aframe, 'South Sudan')
title_str = 'Number of Events in South Sudan'
event_counts(southsudan_aframe, title_str, 'event_type')
title_str = 'Number of Events w/ fatalities > 10 in South Sudan'
event_counts(southsudan_aframe, title_str, 'event_type', 10)
title_str = 'Total Events in South Sudan'
event_counts(southsudan_aframe, title_str)
actor_type(southsudan_aframe, 'South Sudan')
website_vs_bucket_fatalities(southsudan_aframe, southsudan_bframe, 'South Sudan')


# In[26]:


print('SOUTH SUDAN: Correlation between Percent pop in given IPC and NDVI.')
IPC_pct_corr_plots(southsudan_bframe)


# In[27]:


country = 'Afghanistan'
score_freq(b_frame, country)
country = 'Mali'
score_freq(mali_bframe, country)
country = 'Niger'
score_freq(niger_bframe, country)
country = 'Somalia'
score_freq(somalia_bframe, country)
country = 'South Sudan'
score_freq(southsudan_bframe, country)


# #### 

# In[28]:


def multi_country_score_freq(scores, by_country=False):
    #plt.rcParams.update({'font.size': 16})
    #plt.rcParams.update({'figure.figsize': (10,6)})
    #plt.figure(figsize=(20,20))
    fig, ax = plt.subplots()
    ax.set(title='IPC Score freq. by Country')
    #ax.set(figure=plt.figure(figsize=(20,20)))
    s = pd.Series([])
    dumb = pd.DataFrame(['scores', 'country'])
    for country in scores:
        print (country)
        bucket_frame = scores[country]
        phase_scores, m, d = get_cols_by_datelist('1/1/10', pd.datetime.today(), 'IPC_Phase', bucket_frame)
        if by_country:
            df = bucket_frame[phase_scores]#.astype(str) +' '+ country
        else:
            df = bucket_frame[phase_scores]
        
        """for c in phase_scores:
            df[c] = bucket_frame[c]"""
        temp = [df[col] for col in df]
        #temp.append(s)
        s = pd.concat(temp)
        score_count = s.value_counts()
        print(score_count.index.values)
        cur = pd.DataFrame({'Scores':s})
        #cur['scores'] = s
        cur['Country'] = country
        dumb = pd.concat([dumb, cur])
    #score_count.sort_index(inplace=True)
    #countries = [c.split(' ')[1] for c in score_count.index.values]
    #scores = [c.split(' ')[0] for c in score_count.index.values]
    #dfx= pd.DataFrame({'counts': score_count.values, 'country': countries, 'scores':scores})
    #dfx = df.groupby(['country'])
    #display(dfx.groups.keys())
    #print(score_count)
    #score_count.plot(ax=ax, kind='bar')
    #df.groupby('year').case_status.value_counts().unstack(0).
    dumb.groupby('Country').Scores.value_counts().unstack(0).plot(ax=ax,kind='bar')
    plt.show()
    


multi_country_score_freq({'Afghanistan': b_frame,'Somalia': somalia_bframe, 'Mali': mali_bframe, 'South Sudan': southsudan_bframe, 'Niger': niger_bframe}, True)


# In[ ]:




