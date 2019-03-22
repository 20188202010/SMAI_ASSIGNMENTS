#!/usr/bin/env python
# coding: utf-8

# ### Implement a model using linear regression to predict the probablity of getting the admit

# In[61]:


import numpy as np
import pandas as pd


# In[62]:


from pylab import *
import matplotlib
import matplotlib.pyplot as plt


# In[63]:


dataset = pd.read_csv("AdmissionDataset/data.csv")
rows , cols = dataset.shape
print "Dataset: ", rows, cols
target = 'Chance of Admit '


# In[64]:


dataset = dataset.drop(['Serial No.'], axis=1)


# In[65]:


dataset = dataset.sample(frac=1)
train, validate = np.split(dataset, [int(.8*len(dataset))])


# Compute feature matrix X

# In[66]:


def makeX(df):
    n = train.shape[0]
    dimensions = train.shape[1]
    if target in df.columns:
        df = df.drop([target], axis=1)
    X = df.values
    X = np.insert(X, 0, values=1, axis=1)
#     print "row: ",X.shape[0], "cols: ",X.shape[1]
    return X
    
feature = makeX(train)
feature


# In[67]:


def computeBeta(mat, df):    
    transpose = mat.transpose()
    product = np.dot(transpose, mat)
    inverse = np.linalg.inv(product)
    inverse_x = np.dot(inverse, transpose)
    beta = np.dot(inverse_x, df[target])
    return beta
    
beta = computeBeta(feature, train)
# print beta


# In[68]:


def predict(df,beta):
    X = makeX(df)
#     print X
    y = np.dot(X, beta)
    return y

prediction = predict(validate, beta)
print prediction


# #### Compare  the  performance  of  Mean  square  error  loss  function  vs  Mean  Absolute error function vs Mean absolute percentage error function and explain the reasons for the observed behaviour

# In[69]:


def MSE(predicted_y, original_y):
    original = original_y.values
    sum = 0
    n = len(original)
    for i in range(0, n):
        diff = predicted_y[i] - original[i]
        squared_diff = diff**2
        sum += squared_diff
    error = sum/n
    return error

MSE(prediction, validate[target])


# In[70]:


def MAE(predicted_y, original_y):
    original = original_y.values
    sum = 0
    n = len(original)
    for i in range(0, n):
        diff = abs(predicted_y[i] - original[i])
        sum += diff
    error = sum/n
    return error

MAE(prediction, validate[target])


# In[71]:


def MAPE(predicted_y, original_y):
    original = original_y.values
    sum = 0
    n = len(original)
    for i in range(0, n):
        diff = abs(predicted_y[i] - original[i])
        ratio = float(diff)/original[i]
        sum += ratio
    error = sum/n
    return error*100
MAPE(prediction, validate[target])


# In[72]:


filename = raw_input("Enter file for testing: ")
test = pd.read_csv(filename)

test = test.drop(['Serial No.'], axis=1)
# print test.columns
predict(test, beta)

