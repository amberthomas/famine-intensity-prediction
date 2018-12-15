#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_utils import ModisFeatureExtractor
# the Histogram Extractor was never finished 
#from data_utils import ModisHistogramExtractor
from IPython.display import display
import cv2
# In[2]:


from modis_model_util import *


# In[3]:
import argparse
parser = argparse.ArgumentParser(description='Doing the data proc demo.')
parser.add_argument('-d', action='store_true', dest='demo', default=False, help='To demo the tif parse or not my friend.')
args = parser.parse_args()
RUN_DEMO = args.demo


# I can use a funciton call get dates to pull the dates corresponding with my Modis images 
if RUN_DEMO:
    dates = get_dates('modis_data/afg_dates.txt')


# In[4]:


# The feature extractor just grabs the veg indices from the modis images
# the important arguments are the first two, which include which directory the images are stored in and the
# number of bands the image contains (the other two are vestigial hold over from other ideas)
# this code takes a few minutes to run so I just ended up making pickled dfs with the info
# feel free to uncomment it below and see it function 
if RUN_DEMO:
    feature_data = ModisFeatureExtractor('modis_data/afg_full_time', 7, True, False)


# In[5]:


# when looking over the plots of the afghanistan ndvi across dates I noticed that they appeared to be ~ 3 months 
# off of the Artemis data set so the commented out code just realigns the images with the correct date 
# this has already been fixed in the pickles, but if you start cold run this 
#AFGHANISTAN DATE ADJUSTMENT
if RUN_DEMO:
    tempt_df = pd.DataFrame(dates, columns=['d'])
    tempt_df = pd.to_datetime(tempt_df['d'], format='%Y-%m-%d', utc=True)
    display(tempt_df[0])
    tempt_df +=  pd.DateOffset(months=3)
    display(tempt_df[0].month)
    new_dates = []
    for d in tempt_df:
        new_dates.append('{}-{}-{}'.format(d.year, '{:02}'.format(d.month), '{:02}'.format(d.day)))
    print(new_dates[0])
    dates = new_dates


# In[6]:


# if you are running this for the first time trying to grab information from the modis image, this pairs 
#the dates of the image with the index info 
# this information has already been pulled and pickled for your convenience 

if RUN_DEMO:
    feature_info = feature_data.get_data()


    if len(feature_info[0][1]) != len(dates):
        print (len(feature_data.get_data()[0][1]))
        print('issue in command zone')
    afg_df = pd.DataFrame(columns=dates)
    for i in range(0, len(feature_info)):
        s = (feature_info[i][1])
        name = feature_info[i][0]
        afg_df.loc[name] = s
    afg_df = afg_df.sort_index()
    afg_df = afg_df.set_index(afg_df.reset_index().index.values)

    print('Here is what the 1st round processed dataframe for Afghanistan Looks like.')
    print('It will require another round of processing before use.')
    display(afg_df)
    #afg_df.to_pickle('feature_index_afg_pickle')


# In[7]:


# extract the modis data pickles 
mali_df = pd.read_pickle('modis_data/feature_index_mali_int_pickle')
mali_df = mali_df.sort_index()
somalia_df = pd.read_pickle('modis_data/feature_index_somalia_int_pickle')
somalia_df = somalia_df.sort_index()
afg_df = pd.read_pickle('modis_data/feature_index_afg_pickle')
afg_df = afg_df.sort_index()
niger_df = pd.read_pickle('modis_data/feature_index_niger_int_pickle')
niger_df = niger_df.sort_index()
southsudan_df = pd.read_pickle('modis_data/feature_index_southsudan_int_pickle')
southsudan_df = southsudan_df.sort_index()


# In[8]:


# Extract the Artemis data into a data frame
afg_bframe = pd.read_csv('acled_visualization/afghanistan.csv')
mali_bframe = pd.read_csv('acled_visualization/mali.csv')
niger_bframe = pd.read_csv('acled_visualization/niger.csv')
somalia_bframe = pd.read_csv('acled_visualization/somalia.csv')
southsudan_bframe = pd.read_csv('acled_visualization/southsudan.csv')

#There were some issues with pulling image data so in some cases a county needed to be dropped from 
# the modis data, this re-aligns the Artemis data
mali_bframe = mali_bframe.drop(mali_bframe.index[[1,3]])
display(len(mali_bframe.index.values))
display(len(mali_df.index.values))

display(len(somalia_bframe.index.values))
display(len(somalia_df.index.values))

display(len(afg_bframe.index.values))
display(len(afg_df.index.values))

niger_bframe = niger_bframe.drop(niger_bframe.index[[13]])
display(len(niger_bframe.index.values))
display(len(niger_df.index.values))
#with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
    #display((niger_bframe))
#there is no 0,23,32,50,57,58,63,

"""with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
    display((southsudan_bframe))
    display(southsudan_df.index.values)"""
southsudan_bframe = southsudan_bframe.drop(southsudan_bframe.index[[18]])
display(len(southsudan_bframe.index.values))
display(len(southsudan_df.index.values))
#display(afg_df)


# In[9]:


# align artemis data indices to merge properly with modis data
mali_bframe = mali_bframe.set_index(mali_bframe.reset_index().index.values)
somalia_bframe = somalia_bframe.set_index(somalia_bframe.reset_index().index.values)
niger_bframe = niger_bframe.set_index(niger_bframe.reset_index().index.values)
afg_bframe = afg_bframe.set_index(afg_bframe.reset_index().index.values)
southsudan_bframe = southsudan_bframe.set_index(southsudan_bframe.reset_index().index.values)


# In[10]:


# align modis data indices to merge properly with artemis data
somalia_df = somalia_df.set_index(somalia_df.reset_index().index.values)
afg_df = afg_df.set_index(afg_df.reset_index().index.values)
niger_df = niger_df.set_index(niger_df.reset_index().index.values)
southsudan_df = southsudan_df.set_index(southsudan_df.reset_index().index.values)
mali_df = mali_df.set_index(mali_df.reset_index().index.values)


# In[11]:


mali_df3 = format_feature_df(mali_df)
somalia_df3 = format_feature_df(somalia_df)
afg_df3 = format_feature_df(afg_df)
niger_df3 = format_feature_df(niger_df)
southsudan_df3 = format_feature_df(southsudan_df)


# In[12]:


print('Sample of feature table (before aggregated into months)\n')
with pd.option_context('display.max_rows', 10, 'display.max_columns', 16):
    display(afg_df3)


# In[13]:


# compare NDVI from Artemis data and Modis data 
print('Run NDVI comparisons between FAM data set and Data set pulled from MODIS\n')
compare_frames(somalia_bframe, somalia_df3,datetime.date(2012, 1, 1), datetime.date(2017, 3, 1), 'Somalia')
compare_frames(mali_bframe, mali_df3,datetime.date(2014, 1, 1), datetime.date(2017, 3, 1), 'Mali')
compare_frames(niger_bframe, niger_df3,datetime.date(2014, 1, 1), datetime.date(2017, 3, 1), 'Niger')
compare_frames(southsudan_bframe, southsudan_df3,datetime.date(2014, 1, 1), datetime.date(2017, 3, 1), 'South Sudan')
compare_frames(afg_bframe, afg_df3,datetime.date(2014, 1, 1), datetime.date(2017, 3, 1), 'Afg')


# In[14]:


# Need some kind of dictionary for seasons  as a part of the inputs for generating model feature and label data
mali_seasons = {'Interior Delta and Lacustral Zone':['04'],
                'Sahelian zone':['07', '09'],
                'Sudanese zone':['11'],
                'Sudano-guinean zone': ['07', '09']
}

somalia_seasons = {'Bay and Bakool':['01', '07']
}

niger_seasons = {'Sahel':['04', '10']
}

southsudan_seasons = {'Equatoria':['01', '08']
}

afg_seasons = {'Cereal':['06', '10']
}


# In[15]:


# displays functionality of the label/feature divider
# however this runs a bit slowly, so all these dfs are pre-pickled for you 
# this call generates a multi-class classification data set 
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


basically makes the df that will feed the feature label pairs that go into the models 
"""
print('Starting sample label feature table split:\n')
afg_label_table, afg_feature_table = label_season_feature_tables(afg_bframe, afg_df3, afg_seasons, datetime.date(2012, 1, 1), 0)
print('Value Counts of Classes for Afghanistan Example\n', afg_label_table.score.value_counts())


# In[16]:



"""somalia_label_table, somalia_feature_table = label_season_feature_tables(somalia_bframe, somalia_df3, somalia_seasons, datetime.date(2012, 1, 1), 0)
#afg_label_table, afg_feature_table = label_season_feature_tables(afg_bframe, afg_df3, afg_seasons, datetime.date(2012, 1, 1), 2, True)
mali_label_table, mali_feature_table = label_season_feature_tables(mali_bframe, mali_df3,mali_seasons, datetime.date(2012, 1, 1), 0)
niger_label_table, niger_feature_table = label_season_feature_tables(niger_bframe, niger_df3, niger_seasons, datetime.date(2010, 1, 1), 0)
southsudan_label_table, southsudan_feature_table = label_season_feature_tables(southsudan_bframe, southsudan_df3,southsudan_seasons, datetime.date(2013, 9, 1), 0)"""


# In[17]:


#Split data SUPER MODEL MULTICLASS
#This only computes the super model, no longer splits up based on country basis 
print('Splitting up Features for Multi-Label Classification Testing')
train_features, train_labels, test_features, test_labels, feature_list = build_super_model_test_train_array('MULT')
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[18]:


#Multi-label Baselines
print('Mode Baseline')
Mode_Baseline(train_labels, test_labels, (1, 6))
print('Persistence Baseline:\n')
all_country_persistance_wrapper(afg_bframe, somalia_bframe, mali_bframe, niger_bframe, southsudan_bframe, (1, 6))



# In[20]:


#Multi-class Tests

######Hyper-param testing below######
parameter_candidates_rf = [
    {'n_estimators': [100, 500, 700, 1000, 1200, 1500, 2000], 'max_depth': [2,3,4]}
]
scoring = ['f1_macro','f1_weighted', 'accuracy', 'recall_weighted','recall_macro', 'precision_weighted','precision_macro']
class_weight = None
rf = RandomForestClassifier(class_weight = class_weight, random_state = 42)

print('Example of Cross Validation grid search for optimal Random Forest hyperparameters in progress.')
print('Please be patient, this will take a minute...')
hstable = hyperparam_search(train_features, train_labels, test_features, test_labels, parameter_candidates_rf, scoring, rf)
paramsrf = hstable.params.values[0]
paramsrf['random_state'] = 42
paramsrf['class_weight'] = class_weight

rf = RandomForestClassifier(**paramsrf)
classifier_train_test(rf, train_features, train_labels, test_features, test_labels, (1,6), title = 'Random Forest')
print("\nRandom Forest Variable importance:")
get_RF_variable_importance(rf, feature_list, 'Multi-class RF')

######Hyper-param testing above######

print('Hyperparameter Testing for other Multi-Label Classification models skipped for your convenience:\n')
print("\nRandom Forest Weighted Penalty")
class_weight = {1:1, 2:1, 3:1.5, 4:2}
rf = RandomForestClassifier(n_estimators = 100, max_depth=3, class_weight = class_weight, random_state = 42)
classifier_train_test(rf, train_features, train_labels, test_features, test_labels, (1,6), title = 'Random Forest Weighted')
print("\nRandom Forest Variable importance:")
get_RF_variable_importance(rf, feature_list, 'Weighted Binary RF')

print("\nLogistic Regression")
lr = LogisticRegression(C= .9, random_state = 42)
classifier_train_test(lr, train_features, train_labels, test_features, test_labels, (1,6), title = 'Logistic Regression')
print("\nLogistic Regression Weighted Penalty")
lr = LogisticRegression(C= .65, class_weight = {1:1, 2:1, 3:1.5, 4:2}, random_state = 42)
classifier_train_test(lr, train_features, train_labels, test_features, test_labels, (1,6), title = 'Logistic Regression Weighted')
print("\nGradient Boosting")
gb = GradientBoostingClassifier(n_estimators= 100, subsample =.21, max_depth = 3, max_features = 10, random_state = 42)
classifier_train_test(gb, train_features, train_labels, test_features, test_labels, (1,6), title = 'Gradient Boosting')


# In[21]:


#Split data SUPER MODEL
#This only computes the super model, no longer splits up based on country basis 
print('Splitting up Features for Binary Testing')
train_features, train_labels, test_features, test_labels, feature_list = build_super_model_test_train_array('BIN')
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[22]:


#Binary Baselines
print('Mode Baseline')
Mode_Baseline(train_labels, test_labels, (0, 2))
print('Persistence Baseline:\n')
all_country_persistance_wrapper(afg_bframe, somalia_bframe, mali_bframe, niger_bframe, southsudan_bframe, (0, 2))



# In[23]:


#Binary Tests
print('Hyperparameter Testing for Binary Classification skipped for your convenience:\n')
print("Random Forest")
rf = RandomForestClassifier(n_estimators = 1500, max_depth=4, random_state = 42)
classifier_train_test(rf, train_features, train_labels, test_features, test_labels, (0,2), title = 'Random Forest')
print("\nRandom Forest Variable importance:")
get_RF_variable_importance(rf, feature_list, 'Binary RF')
print("\nRandom Forest Weighted Penalty")
rf = RandomForestClassifier(n_estimators = 1000, max_depth=2, class_weight = 'balanced', random_state = 42)
classifier_train_test(rf, train_features, train_labels, test_features, test_labels, (0,2), title = 'Random Forest Weighted')
print("\nRandom Forest Variable importance:")
get_RF_variable_importance(rf, feature_list, 'Weighted Binary RF')
print("\nLogistic Regression")
lr = LogisticRegression(C= .7, random_state = 42)
classifier_train_test(lr, train_features, train_labels, test_features, test_labels, (0,2), title = 'Logistic Regression')
print("\nLogistic Regression Weighted Penalty")
lr = LogisticRegression(C= .5, class_weight = 'balanced', random_state = 42)
classifier_train_test(lr, train_features, train_labels, test_features, test_labels, (0,2), title = 'Logistic Regression Weighted')
print("\nGradient Boosting")
gb = GradientBoostingClassifier(n_estimators = 500,random_state = 42, subsample = .15, max_depth=3, max_features = 10) 
classifier_train_test(gb, train_features, train_labels, test_features, test_labels, (0,2), title = 'Gradient Boosting')


# In[24]:


#Split data SUPER MODEL
#This only computes the super model, no longer splits up based on country basis 
print('Splitting up Features for Regressor Testing (Predicting over Percent Population in crisis).')
train_features, train_labels, test_features, test_labels, feature_list = build_super_model_test_train_array('REG')
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[25]:


#Regressor Tests
print('Hyperparameter Testing for Percent Population Regressor skipped for your convenience:\n')
print('Linear Regression Baseline (use the previos percentage to predict current):\n')
LinReg_baseline(train_features, train_labels, test_features, test_labels)
print('Linear Regression, all features:\n')
regr = LinearRegression()
regressor_train_test(regr,train_features, train_labels, test_features, test_labels, 'Linear Regression')

print("Random Forest")
rf = RandomForestRegressor(n_estimators = 1500, max_depth=3, random_state = 42)
regressor_train_test(rf, train_features, train_labels, test_features, test_labels, title = 'Random Forest')
print("\nRandom Forest Variable importance:")
get_RF_variable_importance(rf, feature_list, 'Regressor RF')


# In[26]:


print("\nGradient Boosting")
gb = GradientBoostingRegressor(loss ='huber', n_estimators = 100,random_state = 42, subsample = .21, max_depth=3, max_features = 25) 
regressor_train_test(gb, train_features, train_labels, test_features, test_labels, title = 'Gradient Boosting')


# In[ ]:




