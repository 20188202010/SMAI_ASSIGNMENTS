#!/usr/bin/env python
# coding: utf-8

# ## Q2. Use  the  Admission dataset to perform the following task. 

# ### PART-2

# #### Compare  the  performances  of  logistic  regression  model  withKNN model on the Admission dataset

# In[1]:


import numpy as np
import pandas as pd
import math


# In[2]:


dataset = pd.read_csv("../input_data/AdmissionDataset/data.csv")
df_r, df_c = dataset.shape #without drop
print "Dataset: ", df_r, df_c
target = 'Chance of Admit '


# In[3]:


dataset = dataset.drop(['Serial No.'], axis=1)


# In[4]:


#normalise dataframe
def normalise(df):
    cols = df.columns
    for c in cols:
        if c != target:
            sd = df[c].std()
            mean = df[c].mean()
            df[c] = (df[c] - mean)/sd
    return df
dataset = normalise(dataset)
print dataset.head()


# Finds classes from chance of admit for ENTIRE dataset

# In[5]:


def findClasses(row, threshold, default=0):
    if row[target] <= threshold:
        label = 0
    else:
        label = 1
    return label


# In[6]:


def helper_classes(df, predicted_class):
    threshold = 0.5
    df[predicted_class] = df.apply(findClasses, axis=1, args=(threshold,0,))
    return df[predicted_class]
dataset['class'] = helper_classes(dataset, 'class')
print dataset.head()


# In[7]:


# dataset = dataset.sample(frac=1)
train, validate = np.split(dataset, [int(.8*len(dataset))])
train_rows, train_cols = train.shape
print train_rows, train_cols
print validate.head()


# In[8]:


def findEuclidean(unknown_class, known_class):

    cols = ['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']
    sum_of_squares = 0
    class_dist = []
    for i in cols:
        if i != target:
            xi = (unknown_class[i] - known_class[i])**2
            sum_of_squares += xi
    
    dist = math.sqrt(sum_of_squares)
    class_dist = [dist, known_class['class']]

    return class_dist


# In[9]:


def findKNeighbours(class_dist, k):
    
    topK = class_dist[0:k]

    count0 = 0
    count1 = 0   
    #traverse the second list and count no of 1,0
    for i in topK:
        if i[1] == 0:
            count0 += 1
        elif i[1] == 1:
            count1 += 1
    if count0 > count1:
        label = 0
    else:
        label = 1
#     print "label: ",label
    return label


# In[10]:


def findKnn(row, df_t, findNeigh, distanceMeasure, k, default=0):
    listOfList=[]

    #iterate on each row of training set.
    for i, r in df_t.iterrows():
        
        temp = distanceMeasure(row,r) #temp: [ <distance>, <class>]
        listOfList.append(temp)    
    listOfList.sort()

    pred = findNeigh(listOfList, k)

    return pred


def helper_knn(df, df_t, predict_col, findNeigh, distanceMeasure, k):
    df[predict_col] = df.apply(findKnn, axis=1, args=(df_t, findNeigh, distanceMeasure, k, 0))
    return df[predict_col]
validate['predict_col'] = helper_knn(validate, train, 'predict_col', findKNeighbours, findEuclidean, 3)
print validate.head()


# In[11]:


#returns Confusion Matrix
def createCM(predicted, actual):

    pred = pd.Series(predicted, name='Predicted')

    actu = pd.Series(actual,    name='Actual')

    conf = pd.crosstab(actu, pred)

    return conf


# In[12]:


#function to find accuracy, precision, recall
#parameter: confusion matrix
def findMeasures(mat):    
    diag = 0
    tot = 0
    for i in mat:
        diag += mat[i][i]
        tot += mat[i].sum()
    accuracy = float(diag)/tot

    precision = np.diag(mat) / np.sum(mat, axis = 0)
    recall = np.diag(mat) / np.sum(mat, axis = 1)
    f1_score_den = 1/precision + 1/recall
    f1_score = float(2)/f1_score_den
    return accuracy, precision, recall,f1_score


# In[13]:


def completeAnalysis(df_v, df_t, predict_col, funNeigh, funDist, bestK):
    df_v[predict_col] = helper_knn(df_v, df_t, predict_col, funNeigh, funDist, bestK)
    
    predicted_val = df_v[predict_col]
    print "predicted: ",len(predicted_val)
    
    actual_val = df_v[target]
    print "actual val: ", len(actual_val)
    mean_actual = actual_val.mean()
    

    
    class_actual = df_v['class']
    confusion_mat = createCM(predicted_val, class_actual)
    print "Confusion Matrix:"
    print confusion_mat
    a,p,r,f = findMeasures(confusion_mat)
    print "Accuracy: ",a*100, "\nPrecision: ", p*100,"\nRecall:", r*100,"\nF1Score: ",f
    


# In[14]:


completeAnalysis(validate, train, 'validate', findKNeighbours, findEuclidean, 3)

