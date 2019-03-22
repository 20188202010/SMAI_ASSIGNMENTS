#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("../input_data/AdmissionDataset/data.csv")
df_r, df_c = dataset.shape
target = 'Chance of Admit '


# Shuffle the rows

# In[3]:


# dataset = dataset.sample(frac=1)
dataset = dataset.drop(['Serial No.'], axis=1)


# In[4]:


def normalise(df):
    global target
    cols = df.columns
    for c in cols:
        if c != target:
            sd = df[c].std()
            mean = df[c].mean()
            
            df[c] = (df[c] - mean)/sd
    return df
dataset = normalise(dataset)


# In[5]:


target_values = dataset[target]


# Split in k folds

# In[6]:


from sklearn.model_selection import KFold


# In[7]:


def makeX(df):
    if target in df.columns:
        df = df.drop([target], axis=1)
    X = df.values
    X = np.insert(X, 0, values=1, axis=1)
    return X


# In[8]:


X = makeX(dataset)


# In[9]:


def initTheta(n):
    theta = np.zeros(n - 2 + 1) #remove chance of admit, serial no. add 1 b0 col
    return theta


# In[10]:


def gradientDescent(mat, actual, alpha, lmbda, iterations):
  
    theta = initTheta(df_c) #df_c: no of columns in original dataframe
    m = len(mat)
    for i in range(0, iterations):
        pred =  np.dot(mat, theta)
        error = np.array(pred - actual)
        gradient =  np.dot(error, mat[:,0])/m
        theta[0] = theta[0] - alpha*gradient

        for j in range(1, len(theta)):
            gradient =  np.dot(error, mat[:,j])/m

            coeff = (lmbda * theta[j])/m

            gradient += coeff
            theta[j] = theta[j] - alpha*gradient
            
    return theta


# In[11]:


def costFunc(true, pred, theta, lmbda):   
    loss = pred - true
    squared_error = np.sum((loss)**2)
    
    cost = squared_error + lmbda*np.sum(np.dot(theta.T, theta))
    
    return cost


# In[12]:


from sklearn.metrics import mean_squared_error


# In[13]:


alpha = 0.001
lmbda = 0.01
iterations = 1000


# In[14]:


y = target_values.values
k_error_list = []
k_val =  np.arange(2,50,1)
for k in k_val:
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    error_list = []
#     print k,
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        theta = gradientDescent(X_train, y_train, alpha, lmbda, iterations)
     
        y_pred = np.dot(X_test, theta)
        err = mean_squared_error(y_pred,y_test)
        error_list.append(err)
#     print "-----------------"
    k_error_list.append(np.mean(error_list))
print "Error list:", k_error_list


# In[15]:


fig, ax = plt.subplots(figsize=(12,6))
ax.plot(k_val, k_error_list)
ax.set_xlabel("K")
ax.set_ylabel("Cross Validation Error")
ax.set_title("Cross Validation Error vs K")
plt.show()


# In[16]:


k_error_list = []
kf = KFold(n_splits=len(X))
kf.get_n_splits(X)
error_list = []
for train_index, test_index in kf.split(X):
#     print "train index: ",len(train_index), "test index: ", len(test_index)
    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    theta = gradientDescent(X_train, y_train, alpha, lmbda, 1000)

    y_pred = np.dot(X_test, theta)
    err = costFunc(y_pred,y_test, theta, lmbda)
    error_list.append(err)
print len(error_list)


# In[17]:


print np.mean(error_list)

