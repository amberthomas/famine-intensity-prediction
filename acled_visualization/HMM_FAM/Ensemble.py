#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
#from sklearn
from pomegranate import *
import pandas as pd
from collections import Counter


# In[4]:


frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_afghanistan.csv')

print("This represents the ensemble model for just the IPC scores")
print("Prints out final accuracy of the model")
#print(len(frame.Country))

afg1 = (frame.jaspreet_2013)
afg2 = (frame.jaspreet_2014)
afg3 = (frame.jaspreet_2015)
afg4 = (frame.jaspreet_2016)
afg5 = (frame.jaspreet_2017)

frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_somalia.csv')

yeardata_afg = np.vstack((afg1,afg2,afg3,afg4))
print(yeardata_afg.shape)

yeardata1_afg = np.vstack((afg1,afg2,afg3,afg4,afg5))

yeardata_afg = yeardata_afg.T - 1
yeardata1_afg = yeardata1_afg.T - 1

som5 = (frame.jaspreet_2013)
#som6 = (frame.jaspreet_2013_2)
som7 = (frame.jaspreet_2014)
#som8 = (frame.jaspreet_2014_2)
som9 = (frame.jaspreet_2015)
#som10 = (frame.jaspreet_2015_2)
som11 = (frame.jaspreet_2016)
#som12 = (frame.jaspreet_2016_2)
som13 = (frame.jaspreet_2017)
#som14 = (frame.jaspreet_2017_2)
som15 = (frame.jaspreet_2018)

yeardata_som = np.vstack((som5,som7,som11,som13))

yeardata1_som = np.vstack((som5,som7,som11,som13,som15))

yeardata_som = yeardata_som.T - 1
yeardata1_som = yeardata1_som.T - 1

frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_southsudan.csv')

print(len(frame.country))
#print(len(frame.country))

year7 = (frame.jaspreet_2014)
#year8 = (frame.jaspreet_2014_2)
year9 = (frame.jaspreet_2015)
#year10 = (frame.jaspreet_2015_2)
year11 = (frame.jaspreet_2016)
#year12 = (frame.jaspreet_2016_2)
year13 = (frame.jaspreet_2017)
#year14 = (frame.jaspreet_2017_2)
year15 = (frame.jaspreet_2018)



yeardata = np.empty([1,1])

#print(yeardata)
yeardata1_ss = np.vstack((year7,year9,year11,year13,year15))
yeardata_ss = np.vstack((year7,year9,year11,year13))

yeardata_ss = yeardata_ss.T - 1
yeardata1_ss = yeardata1_ss.T - 1

frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_mali.csv')

print(len(frame.Country))
year0 = (frame.jaspreet_2013)
year1 = (frame.jaspreet_2014)
year2 = (frame.jaspreet_2015)
#year3 = (frame.jaspreet_2015_2)
year4 = (frame.jaspreet_2016)
#year5 = (frame.jaspreet_2016_2)
year6 = (frame.jaspreet_2017)


yeardata = np.empty([1,1])

#print(yeardata)
yeardata_mali = np.vstack((year0,year1,year2,year4))
yeardata1_mali = np.vstack((year0,year1,year2,year4,year6))

yeardata_mali = yeardata_mali.T - 1
yeardata1_mali = yeardata1_mali.T - 1

frame = pd.read_csv('acled_visualization/HMM_FAM/jaspreet_niger.csv')

year1 = (frame.jaspreet_2014)
year2 = (frame.jaspreet_2015)
#year3 = (frame.jaspreet_2015_2)
year4 = (frame.jaspreet_2016)
#year5 = (frame.jaspreet_2016_2)
year6 = (frame.jaspreet_2017)
#year7 = (frame.jaspreet_2017_2)
year8 = (frame.jaspreet_2018)

yeardata_niger = np.vstack((year1,year2,year4,year6))

yeardata1_niger = np.vstack((year1,year2,year4,year6,year8))
yeardata_niger = yeardata_niger.T - 1
yeardata1_niger = yeardata1_niger.T - 1
#print(yeardata_niger)

#combine all samples 
yeardata = np.vstack((yeardata_afg,yeardata_som,yeardata_ss,yeardata_mali,yeardata_niger))
yeardata1 = np.vstack((yeardata1_afg,yeardata1_som,yeardata1_ss,yeardata1_mali,yeardata1_niger))
yeardata = yeardata[yeardata1.min(axis=1)>=0,:]
yeardata1 = yeardata1[yeardata1.min(axis=1)>=0,:]

yeardata = yeardata.tolist()
yeardata1 = yeardata1.tolist()

#print(yeardata)

model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=7, X=yeardata ,algorithm='viterbi', emission_pseudocount=0.25, end_state=True)

count = 0
output_diagnol = []
output = {}
temp = 0
counter_mat = 0
counter_prob = 0

class0 = 0
class1 = 0
class2 = 0
class3 = 0
class4 = 0
dictionary = []
conf_matrix = np.zeros((5,5))

zerofreq = 0
onefreq = 0
twofreq = 0
threefreq = 0
fourfreq = 0
for element in yeardata1:
    number = element[4]
    if number == 0:
        zerofreq += 1
    elif number == 1:
        onefreq += 1
    elif number == 2:
        twofreq += 1
    elif number == 3:
        threefreq += 1
    elif number == 1:
        fourfreq += 4




for j in range(temp,266):
    
    matrix =[]
    for i in range(0,4):
        
        sequence = []
        temp = []
        
        
        temp = yeardata[j]
        
        if len(temp) == 4:
            temp.append(i)
            sequence = temp
            
        else:
            
            temp[4] = i
            sequence = temp
        
        probability = model.log_probability(sequence)
        
        matrix.append(probability)
        trans, ems = model.forward_backward(sequence)
        
        ind1 = np.argmax(ems[4,:])
        temp = temp.pop()
    count = 0
    for k in range(0, len(matrix)):
        
        if probability == matrix[k]:
            count +=1
            
    if count == 4:
        pass
        #print("All same")
    else :
        pass
        
    ind = np.argmax(matrix)
    
    if yeardata1[j][4] == ind1:
        
        counter_mat +=1
        
    if yeardata1[j][4] == ind:
        counter_prob +=1
        if ind==0:
            conf_matrix[0][0] += 1
        elif ind==1:
            conf_matrix[1][1] += 1
        elif ind==2:
            #print("come here")
            conf_matrix[2][2] += 1
        elif ind==3:
            conf_matrix[3][3] += 1
        elif ind==4:
            conf_matrix[4][4] += 1
            
    else:
        dictionary.append((yeardata1[j][4],ind))

        
        
#print(counter)
print("accuracy_mat is")
print(counter_mat/266)
print("accuracy_prob is")
print(counter_prob/266)
print("classwise accuracies are")
print("Class 0")
print(conf_matrix[0][0]/zerofreq)
print("Class 1")
print(conf_matrix[1][1]/onefreq)
print("Class 2")
print(conf_matrix[2][2]/twofreq)
print("Class 3")
print(conf_matrix[3][3]/threefreq)
print("Class 4")
print(conf_matrix[4][4]/fourfreq)

print("classwise numbers are")
#print("Class 0")

print(conf_matrix)

print("missclassifications are")
print("format is actual, misclassified")
print(Counter(dictionary))
print("misclassified")
#print(dictionary(1,2))

#print(yeardata.shape)
#print(yeardata1.shape)


# In[ ]:




