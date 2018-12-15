#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
#from sklearn
from pomegranate import *
import pandas as pd
from collections import Counter


# In[7]:


print("This function detects transitions in IPC scores from 2 to 3 and 3 to 4. Vary the no_components and pseudocount and algorithm for tuning purposes")

##########################################################################
###Afghanistan section######################################

frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_afghanistan.csv')

#print(len(frame.Country))

afg1 = (frame.jaspreet_2013)
afg2 = (frame.jaspreet_2014)
afg3 = (frame.jaspreet_2015)
afg4 = (frame.jaspreet_2016)
afg5 = (frame.jaspreet_2017)

ndvi2013 = (frame.AVG_2013)
ndvi2014 = (frame.AVG_2014)
ndvi2015 = (frame.AVG_2015)
ndvi2016 = (frame.AVG_2016)
ndvi2017 = (frame.AVG_2017)

anom_13_avg = (frame.anom_13_avg)
anom_14_avg = (frame.anom_14_avg)
anom_15_avg = (frame.anom_15_avg)
#anom_15_2_avg = (frame.anom_15_2_avg)
anom_16_avg = (frame.anom_16_avg)
#anom_16_2_avg = (frame.anom_16_2_avg)
anom_17_avg = (frame.anom_17_avg)

yeardata_afg = np.vstack((afg1,afg2,afg3,afg4))
#print(yeardata_afg.shape)

yeardata1_afg = np.vstack((afg1,afg2,afg3,afg4,afg5))

yeardata_afg = yeardata_afg.T - 1
yeardata1_afg = yeardata1_afg.T - 1

anomdata_afg = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg))
anomdata1_afg = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg, anom_17_avg))

ndvidata_afg = np.vstack((ndvi2013,ndvi2014,ndvi2015,ndvi2016))
ndvidata1_afg = np.vstack((ndvi2013,ndvi2014,ndvi2015,ndvi2016,ndvi2017))

ndvidata_afg = ndvidata_afg.T 
ndvidata1_afg = ndvidata1_afg.T 
anomdata_afg = anomdata_afg.T 
anomdata1_afg = anomdata1_afg.T 

###########Somalia########################################################
frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_somalia.csv')

som5 = (frame.jaspreet_2013)

som7 = (frame.jaspreet_2014)

som9 = (frame.jaspreet_2015)

som11 = (frame.jaspreet_2016)

som13 = (frame.jaspreet_2017)

som15 = (frame.jaspreet_2018)

yeardata_som = np.vstack((som5,som7,som11,som13))

yeardata1_som = np.vstack((som5,som7,som11,som13,som15))

yeardata_som = yeardata_som.T - 1
yeardata1_som = yeardata1_som.T - 1

ndvi_13 = (frame.ndvi_13)

ndvi_14 = (frame.ndvi_14)

ndvi_15 = (frame.ndvi_15)

ndvi_16 = (frame.ndvi_16)

ndvi_17 = (frame.ndvi_17)


ndvidata_som = np.vstack((ndvi_13,ndvi_14,ndvi_15,ndvi_16))
ndvidata1_som = np.vstack((ndvi_13,ndvi_14,ndvi_15,ndvi_16,ndvi_17))

anom_13_avg = (frame.anom_13_avg)
anom_14_avg = (frame.anom_14_avg)
anom_15_avg = (frame.anom_15_avg)
anom_16_avg = (frame.anom_16_avg)
anom_17_avg = (frame.anom_17_avg)

anomdata_som = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg))
anomdata1_som = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg, anom_17_avg))

ndvidata_som = ndvidata_som.T 
ndvidata1_som = ndvidata1_som.T 
anomdata_som = anomdata_som.T 
anomdata1_som = anomdata1_som.T 


#################southSudan##################################################
frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_southsudan.csv')


year0 = [1]*len(frame.country)
year7 = (frame.jaspreet_2014)

year9 = (frame.jaspreet_2015)

year11 = (frame.jaspreet_2016)

year13 = (frame.jaspreet_2017)

year15 = (frame.jaspreet_2018)

ndvi_13_avg = (frame.ndvi_13_avg)
ndvi_14 = (frame.ndvi_14)

ndvi_15 = (frame.ndvi_15)
ndvi_15_2 = (frame.ndvi_15_2)
ndvi_16 = (frame.ndvi_16)
ndvi_16_2 = (frame.ndvi_16_2)
ndvi_17 = (frame.ndvi_17)
ndvi_17_2 = (frame.ndvi_17_2)
ndvi_18 = (frame.ndvi_18)




anom_13_avg = (frame.anom_13_avg)
anom_14_avg = (frame.anom_14_avg)
anom_15_avg = (frame.anom_15_avg)

anom_16_avg = (frame.anom_16_avg)

anom_17_avg = (frame.anom_17_avg)

ndvidata_ss = np.vstack((ndvi_13_avg,ndvi_14,ndvi_15,ndvi_16))
ndvidata1_ss = np.vstack((ndvi_13_avg,ndvi_14,ndvi_15,ndvi_16,ndvi_17))

anomdata_ss = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg))
anomdata1_ss = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg, anom_17_avg))

ndvidata_ss = ndvidata_ss.T 
ndvidata1_ss = ndvidata1_ss.T 
anomdata_ss = anomdata_ss.T 
anomdata1_ss = anomdata1_ss.T 



#yeardata = np.empty([1,1])

#print(yeardata)
yeardata_ss = np.vstack((year0,year7,year9,year11))
yeardata1_ss = np.vstack((year0,year7,year9,year11,year13))

yeardata_ss = yeardata_ss.T - 1
yeardata1_ss = yeardata1_ss.T - 1




##################mali########################################################
frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_mali.csv')

#print(len(frame.Country))
year0 = (frame.jaspreet_2013)
year1 = (frame.jaspreet_2014)
year2 = (frame.jaspreet_2015)

year4 = (frame.jaspreet_2016)
year6 = (frame.jaspreet_2017)


ndvi1_max = (frame.ndvi13_2_max)
ndvi2_max = (frame.ndvi14_2_max)
ndvi4_max = (frame.ndvi15_2_max)
ndvi6_max = (frame.ndvi16_2_max)
ndvi8_max = (frame.ndvi_17_2_max)

anom_13_avg = (frame.anom_13_avg)
anom_14_avg = (frame.anom_14_avg)
anom_15_2_avg = (frame.anom_15_2_avg)
anom_16_2_avg = (frame.anom_16_2_avg)
anom_17_2_avg = (frame.anom_17_2_avg)

anomdata_mali = np.vstack((anom_13_avg,anom_14_avg, anom_15_2_avg, anom_16_2_avg))
anomdata1_mali = np.vstack((anom_13_avg,anom_14_avg, anom_15_2_avg, anom_16_2_avg, anom_17_2_avg))
ndvidata_mali = np.vstack((ndvi1_max,ndvi2_max,ndvi4_max,ndvi6_max))
ndvidata1_mali = np.vstack((ndvi1_max,ndvi2_max,ndvi4_max,ndvi6_max,ndvi8_max))

#ndvidata = np.vstack((ndvi1_min,ndvi2_min,ndvi3_min,ndvi4_min,ndvi5_min,ndvi6_min,ndvi7_min,ndvi8_min))
#ndvidata1 = np.vstack((ndvi1_min,ndvi2_min,ndvi3_min,ndvi4_min,ndvi5_min,ndvi6_min,ndvi7_min,ndvi8_min,ndvi9_min))

ndvidata_mali = ndvidata_mali.T 
ndvidata1_mali = ndvidata1_mali.T 
anomdata_mali = anomdata_mali.T 
anomdata1_mali = anomdata1_mali.T 

yeardata = np.empty([1,1])

#print(yeardata)
yeardata_mali = np.vstack((year0,year1,year2,year4))
yeardata1_mali = np.vstack((year0,year1,year2,year4,year6))

yeardata_mali = yeardata_mali.T - 1
yeardata1_mali = yeardata1_mali.T - 1


#################################Niger########################################

frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_niger.csv')

#print(len(frame.Country))
year0 = [1]*len(frame.Country)
year1 = (frame.jaspreet_2014)
year2 = (frame.jaspreet_2015)
#year3 = (frame.jaspreet_2015_2)
year4 = (frame.jaspreet_2016)
#year5 = (frame.jaspreet_2016_2)
year6 = (frame.jaspreet_2017)
#year7 = (frame.jaspreet_2017_2)
#year8 = (frame.jaspreet_2018)

yeardata_niger = np.vstack((year0,year1,year2,year4))

yeardata1_niger = np.vstack((year0,year1,year2,year4,year6))
yeardata_niger = yeardata_niger.T - 1
yeardata1_niger = yeardata1_niger.T - 1

ndvi2013 = (frame.ndvi2013)
ndvi2014 = (frame.ndvi2014)
#print("length of ndvi is")
#print(len(ndvi2014))
#ndvi2015_1 = (frame.ndvi2015_1)
ndvi2015 = (frame.ndvi2015)
#ndvi2016_1 = (frame.ndvi2016_1)
ndvi2016 = (frame.ndvi2016)
#ndvi2017_1 = (frame.ndvi2017_1)
ndvi2017 = (frame.ndvi2017)

#anomaly = (frame.NdviAllAnom.2012_06)
#print("anomaly is")
#print(anomaly)
anom_13_avg = (frame.anom_13_avg)
anom_14_avg = (frame.anom_14_avg)
anom_15_avg = (frame.anom_15_avg)
#anom_15_2_avg = (frame.anom_15_2_avg)
anom_16_avg = (frame.anom_16_avg)
#anom_16_2_avg = (frame.anom_16_2_avg)
anom_17_avg = (frame.anom_17_avg)
#anom_17_2_avg = (frame.anom_17_2_avg)

ndvidata_niger = np.vstack((ndvi2013,ndvi2014,ndvi2015,ndvi2016))
ndvidata1_niger = np.vstack((ndvi2013,ndvi2014,ndvi2015,ndvi2016,ndvi2017))

anomdata_niger = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg))
anomdata1_niger = np.vstack((anom_13_avg, anom_14_avg, anom_15_avg, anom_16_avg, anom_17_avg))

ndvidata_niger = ndvidata_niger.T 
ndvidata1_niger = ndvidata1_niger.T 
anomdata_niger = anomdata_niger.T 
anomdata1_niger = anomdata1_niger.T 
#print(yeardata_niger)



###########################combine all samples##################################### 
#print(yeardata_afg.shape)
#print(yeardata_som.shape)
#print(yeardata_ss.shape)
#print(yeardata_mali.shape)
#print(yeardata_niger.shape)

country_feature = ([1] * yeardata_afg.shape[0]) + ([2]*yeardata_som.shape[0]) + ([3]*yeardata_ss.shape[0]) + ([4]*yeardata_mali.shape[0]) + ([5]*yeardata_niger.shape[0])
#country_feature.append()
country_feature1 = np.vstack((country_feature,country_feature,country_feature,country_feature,country_feature))
country_feature = np.vstack((country_feature,country_feature,country_feature,country_feature))
country_feature = country_feature.T
country_feature1 = country_feature1.T
#print(country_feature.shape)


yeardata = np.vstack((yeardata_afg,yeardata_som,yeardata_ss,yeardata_mali,yeardata_niger))
yeardata1 = np.vstack((yeardata1_afg,yeardata1_som,yeardata1_ss,yeardata1_mali,yeardata1_niger))

transitiondata = np.zeros((yeardata.shape))
transitiondata1 = np.zeros((yeardata1.shape))
#transitiondata = transitiondata.T 
#transitiondata1 = transitiondata1.T 

for i in range(0,yeardata.shape[0]):
    for j in range(0,yeardata.shape[1]-1):
        if (yeardata[i][j] == 1 and yeardata[i][j+1] == 2):
            transitiondata[i][j+1] = 1
        elif (yeardata[i][j] == 2 and yeardata[i][j+1] == 3):
            transitiondata[i][j+1] = 2
        else:
            transitiondata[i][j+1] = 0
            
for i in range(0,yeardata1.shape[0]):
    for j in range(0,yeardata1.shape[1]-1):
        if (yeardata1[i][j] == 1 and yeardata1[i][j+1] == 2):
            transitiondata1[i][j+1] = 1
        elif (yeardata1[i][j] == 2 and yeardata1[i][j+1] == 3):
            transitiondata1[i][j+1] = 2
        else:
            transitiondata1[i][j] = 0

anomdata = np.vstack((anomdata_afg,anomdata_som,anomdata_ss,anomdata_mali,anomdata_niger))
anomdata1 = np.vstack((anomdata1_afg,anomdata1_som,anomdata1_ss,anomdata1_mali,anomdata1_niger))

ndvidata = np.vstack((ndvidata_afg,ndvidata_som,ndvidata_ss,ndvidata_mali,ndvidata_niger))
ndvidata1 = np.vstack((ndvidata1_afg,ndvidata1_som,ndvidata1_ss,ndvidata1_mali,ndvidata1_niger))

#print("final yeardata shape")
#print(yeardata1.shape)
feature = country_feature[yeardata1.min(axis=1)>=0,:]
feature1 = country_feature1[yeardata1.min(axis=1)>=0,:]
anomdata = anomdata[yeardata1.min(axis=1)>=0,:]
anomdata1 = anomdata1[yeardata1.min(axis=1)>=0,:]
ndvidata = ndvidata[yeardata1.min(axis=1)>=0,:]
ndvidata1 = ndvidata1[yeardata1.min(axis=1)>=0,:]
transitiondata = transitiondata[yeardata1.min(axis=1)>=0,:]
transitiondata1 = transitiondata1[yeardata1.min(axis=1)>=0,:]
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]
yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()
ndvidata = ndvidata.tolist()
ndvidata1 = ndvidata1.tolist()
anomdata = anomdata.tolist()
anomdata1 = anomdata1.tolist()
feature = feature.tolist()
feature1 = feature1.tolist()
transitiondata = transitiondata.tolist()
transitiondata1 = transitiondata1.tolist()

zerofreq = 0
onefreq = 0
twofreq = 0

for element in transitiondata1:
    number = element[4]
    if number == 0:
        zerofreq += 1
    elif number == 1:
        onefreq += 1
    elif number == 2:
        twofreq += 1
    

print(zerofreq)
print(onefreq)
print(twofreq)
print(zerofreq+onefreq+twofreq)

transition = np.array((transitiondata))
transition1 = np.array((transitiondata1))

year = np.array((yeardata))
ndvi = np.array((ndvidata))
year1 = np.array((yeardata1))
ndvi1 = np.array((ndvidata1))
anom = np.array((anomdata))
anom1 = np.array((anomdata1))
feature = np.array((feature))
feature1 = np.array((feature1))

inputs = np.dstack((transition,ndvi,anom))
inputs1 = np.dstack((transition1,ndvi1,anom1))

inputs = np.reshape(inputs,(267,4,3))
inputs1 = np.reshape(inputs1,(267,5,3))

class0 = 0
class1 = 0
class2 = 0

count = 0
temps = 0
counter_prob = 0
dictionary = []
model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=50, X=inputs ,algorithm='baum-welch', verbose=False, emission_pseudocount=1, end_state=False)
for j in range(temps,267):
    matrix =[]
    comparison = inputs1[j][4][0]
    for i in range(0,5):
        temph = inputs1[j][:][:]
        temph[4][0] = i
        probability = model.log_probability(temph)
        matrix.append(probability)
    ind = matrix.index(max(matrix))
    if comparison == ind:
        counter_prob +=1
        if ind==0:
            class0 += 1
        elif ind==1:
            class1 += 1
        elif ind==2:
            class2 += 1
    else:
        dictionary.append((comparison,ind))
print("accuracy_prob of Ensemble model is")
print(counter_prob)
print(counter_prob/267)

print("class 0 accuracy" )
print(class0/zerofreq)
print("class1 accuracy")
print(class1/onefreq)
print("class2 accuracy")
print(class2/twofreq)

print("misclassifications are")
print("format is actual, misclassified")
print(Counter(dictionary))


# In[ ]:




