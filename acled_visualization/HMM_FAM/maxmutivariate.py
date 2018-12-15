#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
#from sklearn
from pomegranate import *
import pandas as pd
from collections import Counter


# In[3]:


model = HiddenMarkovModel()


# In[4]:


frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_mali.csv')
print("length is")
print(len(frame.Country))
year0 = (frame.jaspreet_2013)
year1 = (frame.jaspreet_2014)
year2 = (frame.jaspreet_2015)
year3 = (frame.jaspreet_2015_2)
year4 = (frame.jaspreet_2016)
year5 = (frame.jaspreet_2016_2)
year6 = (frame.jaspreet_2017)
year7 = (frame.jaspreet_2017_2)
year8 = (frame.jaspreet_2018)

yeardata = np.empty([1,1])

ndvi1 = (frame.ndvi_13_2)
ndvi2 = (frame.ndvi_14_2)
ndvi3 = (frame.ndvi_15_1)
ndvi4 = (frame.ndvi_15_2)
ndvi5 = (frame.ndvi_16_1)
ndvi6 = (frame.ndvi_16_2)
ndvi7 = (frame.ndvi_17_1)
ndvi8 = (frame.ndvi_17_2)
ndvi9 = (frame.ndvi_18_1)

ndvi1_max = (frame.ndvi13_2_max)
ndvi2_max = (frame.ndvi14_2_max)
ndvi3_max = (frame.ndvi15_max)
ndvi4_max = (frame.ndvi15_2_max)
ndvi5_max = (frame.ndvi_16_max)
ndvi6_max = (frame.ndvi16_2_max)
ndvi7_max = (frame.ndvi17_max)
ndvi8_max = (frame.ndvi_17_2_max)
ndvi9_max = (frame.ndvi_18_max)

ndvi1_min = (frame.ndvi13_2_min)
ndvi2_min = (frame.ndvi14_2_min)
ndvi3_min = (frame.ndvi15_min)
ndvi4_min = (frame.ndvi15_2_min)
ndvi5_min = (frame.ndvi_16_min)
ndvi6_min = (frame.ndvi16_2_min)
ndvi7_min = (frame.ndvi17_min)
ndvi8_min = (frame.ndvi_17_2_min)
ndvi9_min = (frame.ndvi_18_min)

yeardata = np.vstack((year0,year1,year2,year3,year4,year5,year6,year7))
yeardata1 = np.vstack((year0,year1,year2,year3,year4,year5,year6,year7,year8))
yeardata = yeardata.T - 1
yeardata1 = yeardata1.T - 1

yeardata = yeardata[yeardata.min(axis=1)>=0,:]
yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]
#print(yeardata1)

#ndvidata = np.vstack((ndvi1,ndvi2,ndvi3,ndvi4,ndvi5,ndvi6,ndvi7,ndvi8))
#ndvidata1 = np.vstack((ndvi1,ndvi2,ndvi3,ndvi4,ndvi5,ndvi6,ndvi7,ndvi8,ndvi9))

#ndvidata = np.vstack((ndvi1_max,ndvi2_max,ndvi3_max,ndvi4_max,ndvi5_max,ndvi6_max,ndvi7_max,ndvi8_max))
#ndvidata1 = np.vstack((ndvi1_max,ndvi2_max,ndvi3_max,ndvi4_max,ndvi5_max,ndvi6_max,ndvi7_max,ndvi8_max,ndvi9_max))

ndvidata = np.vstack((ndvi1_min,ndvi2_min,ndvi3_min,ndvi4_min,ndvi5_min,ndvi6_min,ndvi7_min,ndvi8_min))
ndvidata1 = np.vstack((ndvi1_min,ndvi2_min,ndvi3_min,ndvi4_min,ndvi5_min,ndvi6_min,ndvi7_min,ndvi8_min,ndvi9_min))

ndvidata = ndvidata.T 
ndvidata1 = ndvidata1.T 

print("ndvi shape is")
print(ndvidata.shape)
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]
yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

print("yeardata1 shape is")
print(yeardata1.shape)
ndvidata = ndvidata[yeardata1.min(axis=1)>=0,:]
ndvidata1 = ndvidata1[yeardata1.min(axis=1)>=0,:]
year = np.array((yeardata))
yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()
ndvidata = ndvidata.tolist()
ndvidata1 = ndvidata1.tolist()

year = np.array((yeardata))
ndvi = np.array((ndvidata))
year1 = np.array((yeardata1))
ndvi1 = np.array((ndvidata1))

inputs = np.dstack((year,ndvi))
inputs1 = np.dstack((year1,ndvi1))

inputs = np.reshape(inputs,(49,8,2))
inputs1 = np.reshape(inputs1,(49,9,2))

model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=12, X=inputs ,algorithm='baum-welch', verbose=False, emission_pseudocount=0.5, end_state=False)
count = 0
output_diagnol = []
dictionary = {}
temps = 0
counter_mat = 0
counter_prob = 0

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0


for j in range(temps,49):
    matrix =[]
    comparison = inputs1[j][8][0]
    
    for i in range(0,5):
        temph = inputs1[j][:][:]
        #print(temph.shape)
        temph[8][0] = i
        probability = model.log_probability(temph)
        matrix.append(probability)
    ind = matrix.index(max(matrix))
    if comparison == ind:
        counter_prob +=1
        if ind==0:
            class0 += 1
            #print("class 0")
            #print(class0)
        elif ind==1:
            class1 += 1
        elif ind==2:
            #print("come here")
            class2 += 1
        elif ind==3:
            class3 += 1
        elif ind==4:
            class4 += 1

print("accuracy_prob is")
print(counter_prob)
print(counter_prob/49)

print("classwise accuracies are")
print("Class 0")
print(class0/49)
print("Class 1")
print(class1/49)
print("Class 2")
print(class2/49)
print("Class 3")
print(class3/49)
print("Class 4")
print(class4/49)


# In[5]:


frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_niger.csv')

print(len(frame.Country))

year1 = (frame.jaspreet_2014)
year2 = (frame.jaspreet_2015)
year3 = (frame.jaspreet_2015_2)
year4 = (frame.jaspreet_2016)
year5 = (frame.jaspreet_2016_2)
year6 = (frame.jaspreet_2017)
year7 = (frame.jaspreet_2017_2) 
#year8 = (frame.jaspreet_2018)



yeardata = np.empty([1,1])


yeardata = np.vstack((year1,year2,year3,year4,year5,year6))

yeardata1 = np.vstack((year1,year2,year3,year4,year5,year6,year7))

print("this part complete")

#ndvi2012 = (frame.ndvi2012)
#ndvi2013 = (frame.ndvi2013)
ndvi2014 = (frame.ndvi2014)
#print("length of ndvi is")
#print(len(ndvi2014))
ndvi2015_1 = (frame.ndvi2015_1)
ndvi2015 = (frame.ndvi2015)
ndvi2016_1 = (frame.ndvi2016_1)
ndvi2016 = (frame.ndvi2016)
ndvi2017_1 = (frame.ndvi2017_1)
ndvi2017 = (frame.ndvi2017)

ndvi14_max = (frame.ndvi14_max)
#print("length of ndvi is")
#print(len(ndvi2014))
ndvi15_max = (frame.ndvi15_max)
ndvi15_2_max = (frame.ndvi15_2_max)
ndvi16_max = (frame.ndvi16_max)
ndvi16_2_max = (frame.ndvi16_2_max)
ndvi17_max = (frame.ndvi17_max)
ndvi17_2_max = (frame.ndvi17_2_max)

ndvi14_min = (frame.ndvi14_min)
#print("length of ndvi is")
#print(len(ndvi2014))
ndvi15_min = (frame.ndvi15_min)
ndvi15_2_min = (frame.ndvi15_2_min)
ndvi16_min = (frame.ndvi16_min)
ndvi16_2_min = (frame.ndvi16_2_min)
ndvi17_min = (frame.ndvi17_min)
ndvi17_2_min = (frame.ndvi17_2_min)


#ndvidata = np.empty([1,1])
#ndvidata = np.vstack((ndvi2014,ndvi2015_1,ndvi2015,ndvi2016_1,ndvi2016,ndvi2017_1))
#ndvidata1 = np.vstack((ndvi2014,ndvi2015_1,ndvi2015,ndvi2016_1,ndvi2016,ndvi2017_1,ndvi2017))

#ndvidata = np.vstack((ndvi14_max,ndvi15_max,ndvi15_2_max,ndvi16_max,ndvi16_2_max,ndvi17_max))
#ndvidata1 = np.vstack((ndvi14_max,ndvi15_max,ndvi15_2_max,ndvi16_max,ndvi16_2_max,ndvi17_max,ndvi17_2_max))

ndvidata = np.vstack((ndvi14_min,ndvi15_min,ndvi15_2_min,ndvi16_min,ndvi16_2_min,ndvi17_min))
ndvidata1 = np.vstack((ndvi14_min,ndvi15_min,ndvi15_2_min,ndvi16_min,ndvi16_2_min,ndvi17_min,ndvi17_2_min))
#ndvidata = np.vstack((ndvi2014,ndvi2015_1,ndvi2015,ndvi2016_1,ndvi2016,ndvi2017_1))
#ndvidata1 = np.vstack((ndvi2014,ndvi2015_1,ndvi2015,ndvi2016_1,ndvi2016,ndvi2017_1,ndvi2017))

#yeardata = np.empty([1,1])
#yeardata = np.vstack((year1,year2,year3,year4))
#yeardata1 = np.vstack((year1,year2,year3,year4,year5))
yeardata = yeardata.T - 1
yeardata1 = yeardata1.T - 1

print(yeardata.shape[0])
ndvidata = ndvidata.T 
ndvidata1 = ndvidata1.T 
print("ndvi shape is")
print(ndvidata.shape[0])
ndvidata = ndvidata[yeardata1.min(axis=1)>=0,:]
ndvidata1 = ndvidata1[yeardata1.min(axis=1)>=0,:]
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]
print("yeardata shape is")
print(yeardata.shape[0])
yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

year = np.array((yeardata))
yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()
ndvidata = ndvidata.tolist()
ndvidata1 = ndvidata1.tolist()

year = np.array((yeardata))
ndvi = np.array((ndvidata))
year1 = np.array((yeardata1))
ndvi1 = np.array((ndvidata1))

inputs = np.dstack((year,ndvi))
inputs1 = np.dstack((year1,ndvi1))

inputs = np.reshape(inputs,(48,6,2))
inputs1 = np.reshape(inputs1,(48,7,2))

model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=2, X=inputs ,algorithm='viterbi', verbose=False, emission_pseudocount=0.5, end_state=False)
count = 0
output_diagnol = []
dictionary = {}
temps = 0
counter_mat = 0
counter_prob = 0
class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0

for j in range(temps,48):
    matrix =[]
    comparison = inputs1[j][6][0]
    
    for i in range(0,5):
        temph = inputs1[j][:][:]
        #print(temph.shape)
        temph[6][0] = i
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
        elif ind==3:
            class3 += 1
        elif ind==4:
            class4 += 1

print("accuracy_prob of Niger is")
print(counter_prob)
print(counter_prob/48)
print("classwise accuracies are")
print("Class 0")
print(class0/48)
print("Class 1")
print(class1/48)
print("Class 2")
print(class2/48)
print("Class 3")
print(class3/48)
print("Class 4")
print(class4/48)


# In[6]:


frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_afghanistan.csv')
print(len(frame.Country))

year1 = (frame.jaspreet_2013)
year2 = (frame.jaspreet_2014)
year3 = (frame.jaspreet_2015)
year4 = (frame.jaspreet_2016)
year5 = (frame.jaspreet_2017)

ndvi2012 = (frame.AVG_2012)
ndvi2013 = (frame.AVG_2013)
ndvi2014 = (frame.AVG_2014)
ndvi2015 = (frame.AVG_2015)
ndvi2016 = (frame.AVG_2016)
ndvi2017 = (frame.AVG_2017)



ndvi_12_max = (frame.ndvi12_max)

ndvi_13_max = (frame.ndvi13_max)

ndvi_14_max = (frame.ndvi14_max)

ndvi_15_max = (frame.ndvi15_max)

ndvi_16_max = (frame.ndvi16_max)

ndvi_17_max = (frame.ndvi17_max)



ndvi_12_min = (frame.ndvi12_min)

ndvi_13_min = (frame.ndvi13_min)

ndvi_14_min = (frame.ndvi14_min)

ndvi_15_min = (frame.ndvi15_min)

ndvi_16_min = (frame.ndvi16_min)

ndvi_17_min = (frame.ndvi17_min)

#yeardata = np.empty([1,1])
yeardata = np.vstack((year1,year2,year3,year4))
yeardata1 = np.vstack((year1,year2,year3,year4,year5))
yeardata = yeardata.T - 1
yeardata1 = yeardata1.T - 1


print(yeardata1.shape[0])
print("this part complete")

ndvidata = np.empty([1,1])
#ndvidata = np.vstack((ndvi2013,ndvi2014,ndvi2015,ndvi2016))
#ndvidata1 = np.vstack((ndvi2013,ndvi2014,ndvi2015,ndvi2016,ndvi2017))

ndvidata = np.vstack((ndvi_13_max,ndvi_14_max,ndvi_15_max,ndvi_16_max))
ndvidata1 = np.vstack((ndvi_13_max,ndvi_14_max,ndvi_15_max,ndvi_16_max,ndvi_17_max))

#ndvidata = np.vstack((ndvi_13_min,ndvi_14_min,ndvi_15_min,ndvi_16_min))
#ndvidata1 = np.vstack((ndvi_13_min,ndvi_14_min,ndvi_15_min,ndvi_16_min,ndvi_17_min))



ndvidata = ndvidata.T 
ndvidata1 = ndvidata1.T 
#print("ndvi shape is")
#print(ndvidata)
ndvidata = ndvidata[yeardata1.min(axis=1)>=0,:]
ndvidata1 = ndvidata1[yeardata1.min(axis=1)>=0,:]
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]

yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

#year = np.array((yeardata))
yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()
ndvidata = ndvidata.tolist()
ndvidata1 = ndvidata1.tolist()

year = np.array((yeardata))
ndvi = np.array((ndvidata))
year1 = np.array((yeardata1))
ndvi1 = np.array((ndvidata1))

inputs = np.dstack((year,ndvi))
inputs1 = np.dstack((year1,ndvi1))

inputs = np.reshape(inputs,(32,4,2))
inputs1 = np.reshape(inputs1,(32,5,2))

model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=2, X=inputs ,algorithm='viterbi', verbose=False, emission_pseudocount=0.5, end_state=False)
count = 0
output_diagnol = []
dictionary = {}
temps = 0
counter_mat = 0
counter_prob = 0
class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0

for j in range(temps,32):
    matrix =[]
    comparison = inputs1[j][4][0]
    
    for i in range(0,5):
        temph = inputs1[j][:][:]
        #print(temph.shape)
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
        elif ind==3:
            class3 += 1
        elif ind==4:
            class4 += 1

print("accuracy_prob of Afghanistan is")
print(counter_prob)
print(counter_prob/32)
print("classwise accuracies are")
print("Class 0")
print(class0/32)
print("Class 1")
print(class1/32)
print("Class 2")
print(class2/32)
print("Class 3")
print(class3/32)
print("Class 4")
print(class4/32)


# In[7]:


frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_southsudan.csv')

print(len(frame.country))

year7 = (frame.jaspreet_2014)
#year8 = (frame.jaspreet_2014_2)
year9 = (frame.jaspreet_2015)
year10 = (frame.jaspreet_2015_2)
year11 = (frame.jaspreet_2016)
year12 = (frame.jaspreet_2016_2)
year13 = (frame.jaspreet_2017)
year14 = (frame.jaspreet_2017_2)
year15 = (frame.jaspreet_2018)




yeardata = np.empty([1,1])


yeardata = np.vstack((year7,year9,year10,year11,year12,year13,year14))

yeardata1 = np.vstack((year7,year9,year10,year11,year12,year13,year14,year15))

ndvi_14 = (frame.ndvi_14)
#ndvi_14_2 = (frame.ndvi_14_2)
ndvi_15 = (frame.ndvi_15)
ndvi_15_2 = (frame.ndvi_15_2)
ndvi_16 = (frame.ndvi_16)
ndvi_16_2 = (frame.ndvi_16_2)
ndvi_17 = (frame.ndvi_17)
ndvi_17_2 = (frame.ndvi_17_2)
ndvi_18 = (frame.ndvi_18)

ndvi14_max = (frame.ndvi14_max)
ndvi15_max = (frame.ndvi15_max)
ndvi15_2_max = (frame.ndvi15_2_max)
ndvi16_max = (frame.ndvi16_max)
ndvi16_2_max = (frame.ndvi16_2_max)
ndvi17_max = (frame.ndvi17_max)
ndvi17_2_max = (frame.ndvi17_2_max)
ndvi18_max = (frame.ndvi18_max)

ndvi14_min = (frame.ndvi14_min)
ndvi15_min = (frame.ndvi15_min)
ndvi15_2_min = (frame.ndvi15_2_min)
ndvi16_min = (frame.ndvi16_min)
ndvi16_2_min = (frame.ndvi16_2_min)
ndvi17_min = (frame.ndvi17_min)
ndvi17_2_min = (frame.ndvi17_2_min)
ndvi18_min = (frame.ndvi18_min)


#ndvidata = np.vstack((ndvi14_max,ndvi15_max,ndvi15_2_max,ndvi16_max,ndvi16_2_max,ndvi17_max,ndvi17_2_max))
#ndvidata1 = np.vstack((ndvi14_max,ndvi15_max,ndvi15_2_max,ndvi16_max,ndvi16_2_max,ndvi17_max,ndvi17_2_max,ndvi18_max))

ndvidata = np.vstack((ndvi14_min,ndvi15_min,ndvi15_2_min,ndvi16_min,ndvi16_2_min,ndvi17_min,ndvi17_2_min))
ndvidata1 = np.vstack((ndvi14_min,ndvi15_min,ndvi15_2_min,ndvi16_min,ndvi16_2_min,ndvi17_min,ndvi17_2_min,ndvi18_min))



ndvidata = ndvidata.T 
ndvidata1 = ndvidata1.T 


yeardata = yeardata.T - 1
yeardata1 = yeardata1.T - 1


ndvidata = ndvidata[yeardata1.min(axis=1)>=0,:]
ndvidata1 = ndvidata1[yeardata1.min(axis=1)>=0,:]
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]

yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

year = np.array((yeardata))
yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()
ndvidata = ndvidata.tolist()
ndvidata1 = ndvidata1.tolist()

year = np.array((yeardata))
ndvi = np.array((ndvidata))
year1 = np.array((yeardata1))
ndvi1 = np.array((ndvidata1))

inputs = np.dstack((year,ndvi))
inputs1 = np.dstack((year1,ndvi1))

inputs = np.reshape(inputs,(74,7,2))
inputs1 = np.reshape(inputs1,(74,8,2))

model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=4, X=inputs ,algorithm='baum-welch', verbose=False, emission_pseudocount=0.5, end_state=False)
count = 0
output_diagnol = []
dictionary = {}
temps = 0
counter_mat = 0
counter_prob = 0
class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0

for j in range(temps,74):
    matrix =[]
    comparison = inputs1[j][7][0]
    
    for i in range(0,5):
        temph = inputs1[j][:][:]
        #print(temph.shape)
        temph[7][0] = i
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
        elif ind==3:
            class3 += 1
        elif ind==4:
            class4 += 1

print("accuracy_prob of south sudan is")
print(counter_prob)
print(counter_prob/74)
print("classwise accuracies are")
print("Class 0")
print(class0/74)
print("Class 1")
print(class1/74)
print("Class 2")
print(class2/74)
print("Class 3")
print(class3/74)
print("Class 4")
print(class4/74)


# In[8]:


frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_somalia.csv')
print(len(frame.Country))

year3 = (frame.jaspreet_2012)
year4 = (frame.jaspreet_2012_2)
year5 = (frame.jaspreet_2013)
year6 = (frame.jaspreet_2013_2)
year7 = (frame.jaspreet_2014)
year8 = (frame.jaspreet_2014_2)
year9 = (frame.jaspreet_2015)
year10 = (frame.jaspreet_2015_2)
year11 = (frame.jaspreet_2016)
year12 = (frame.jaspreet_2016_2)
year13 = (frame.jaspreet_2017)
year14 = (frame.jaspreet_2017_2)
year15 = (frame.jaspreet_2018)


ndvi_12 = (frame.ndvi_12)
ndvi_12_2 = (frame.ndvi_12_2)
ndvi_13 = (frame.ndvi_13)
ndvi_13_2 = (frame.ndvi_13_2)
ndvi_14 = (frame.ndvi_14)
ndvi_14_2 = (frame.ndvi_14_2)
ndvi_15 = (frame.ndvi_15)
ndvi_15_2 = (frame.ndvi_15_2)
ndvi_16 = (frame.ndvi_16)
ndvi_16_2 = (frame.ndvi_16_2)
ndvi_17 = (frame.ndvi_17)
ndvi_17_2 = (frame.ndvi_17_2)
ndvi_18 = (frame.ndvi_18)

ndvi12_max = (frame.ndvi12_max)
ndvi_12_2_max = (frame.ndvi_12_2_max)
ndvi_13_max = (frame.ndvi_13_max)
ndvi_13_2_max = (frame.ndvi_13_2_max)
ndvi_14_max = (frame.ndvi_14_max)
ndvi_14_2_max = (frame.ndvi_14_2_max)
ndvi_15_max = (frame.ndvi_15_max)
ndvi_15_2_max = (frame.ndvi_15_2_max)
ndvi_16_max = (frame.ndvi_16_max)
ndvi_16_2_max = (frame.ndvi_16_2_max)
ndvi_17_max = (frame.ndvi_17_max)
ndvi_17_2_max = (frame.ndvi_17_2_max)
ndvi_18_max = (frame.ndvi_18_max)

ndvi12_min = (frame.ndvi12_min)
ndvi_12_2_min = (frame.ndvi_12_2_min)
ndvi_13_min = (frame.ndvi_13_min)
ndvi_13_2_min = (frame.ndvi_13_2_min)
ndvi_14_min = (frame.ndvi_14_min)
ndvi_14_2_min = (frame.ndvi_14_2_min)
ndvi_15_min = (frame.ndvi_15_min)
ndvi_15_2_min = (frame.ndvi_15_2_min)
ndvi_16_min = (frame.ndvi_16_min)
ndvi_16_2_min = (frame.ndvi_16_2_min)
ndvi_17_min = (frame.ndvi_17_min)
ndvi_17_2_min = (frame.ndvi_17_2_min)
ndvi_18_min = (frame.ndvi_18_min)


#yeardata = np.empty([1,1])
yeardata1 = np.vstack((year3,year4,year5,year6,year7,year8,year9,year10,year11,year12,year13,year14,year15))
yeardata = np.vstack((year3,year4,year5,year6,year7,year8,year9,year10,year11,year12,year13,year14))
yeardata = yeardata.T - 1
yeardata1 = yeardata1.T - 1


#ndvidata = np.empty([1,1])
#ndvidata = np.vstack((ndvi_12,ndvi_12_2,ndvi_13,ndvi_13_2,ndvi_14,ndvi_14_2,ndvi_15,ndvi_15_2,ndvi_16,ndvi_16_2,ndvi_17,ndvi_17_2))
#ndvidata1 = np.vstack((ndvi_12,ndvi_12_2,ndvi_13,ndvi_13_2,ndvi_14,ndvi_14_2,ndvi_15,ndvi_15_2,ndvi_16,ndvi_16_2,ndvi_17,ndvi_17_2,ndvi_18))

#ndvidata = np.vstack((ndvi12_max,ndvi_12_2_max,ndvi_13_max,ndvi_13_2_max,ndvi_14_max,ndvi_14_2_max,ndvi_15_max,ndvi_15_2_max,ndvi_16_max,ndvi_16_2_max,ndvi_17_max,ndvi_17_2_max))
#ndvidata1 = np.vstack((ndvi12_max,ndvi_12_2_max,ndvi_13_max,ndvi_13_2_max,ndvi_14_max,ndvi_14_2_max,ndvi_15_max,ndvi_15_2_max,ndvi_16_max,ndvi_16_2_max,ndvi_17_max,ndvi_17_2_max,ndvi_18_max))

ndvidata = np.vstack((ndvi12_min,ndvi_12_2_min,ndvi_13_min,ndvi_13_2_min,ndvi_14_min,ndvi_14_2_min,ndvi_15_min,ndvi_15_2_min,ndvi_16_min,ndvi_16_2_min,ndvi_17_min,ndvi_17_2_min))
ndvidata1 = np.vstack((ndvi12_min,ndvi_12_2_min,ndvi_13_min,ndvi_13_2_min,ndvi_14_min,ndvi_14_2_min,ndvi_15_min,ndvi_15_2_min,ndvi_16_min,ndvi_16_2_min,ndvi_17_min,ndvi_17_2_min,ndvi_18_min))

ndvidata = ndvidata.T 
ndvidata1 = ndvidata1.T 

ndvidata = ndvidata[yeardata1.min(axis=1)>=0,:]
ndvidata1 = ndvidata1[yeardata1.min(axis=1)>=0,:]
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]
print("yeardata shape is")
print(yeardata.shape[0])
yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

#year = np.array((yeardata))
yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()
ndvidata = ndvidata.tolist()
ndvidata1 = ndvidata1.tolist()

year = np.array((yeardata))
ndvi = np.array((ndvidata))
year1 = np.array((yeardata1))
ndvi1 = np.array((ndvidata1))


inputs = np.dstack((year,ndvi))
inputs1 = np.dstack((year1,ndvi1))

inputs = np.reshape(inputs,(56,12,2))
inputs1 = np.reshape(inputs1,(56,13,2))

model = HiddenMarkovModel.from_samples(MultivariateGaussianDistribution, n_components=3, X=inputs ,algorithm='viterbi', verbose=False, emission_pseudocount=0.5, end_state=False)
count = 0
output_diagnol = []
dictionary = {}
temps = 0
counter_mat = 0
counter_prob = 0
class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0

for j in range(temps,56):
    matrix =[]
    comparison = inputs1[j][12][0]
    
    for i in range(0,5):
        temph = inputs1[j][:][:]
        #print(temph.shape)
        temph[12][0] = i
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
        elif ind==3:
            class3 += 1
        elif ind==4:
            class4 += 1

print("accuracy_prob of somalia is")
print(counter_prob)
print(counter_prob/56)
print("classwise accuracies are")
print("Class 0")
print(class0/56)
print("Class 1")
print(class1/56)
print("Class 2")
print(class2/56)
print("Class 3")
print(class3/56)
print("Class 4")
print(class4/56)


# In[ ]:




