#!/usr/bin/env python
# coding: utf-8

# ### Q3 Implement logistic regression using One vs All and One vs One approaches

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("../input_data/WineQuality/data.csv", sep = ";")
target = 'quality'
new_target = dataset[target]
# print dataset.head()
col_list = dataset.columns[:-1]
df_c = dataset.shape[1]


# In[3]:


#standardise dataframe
def standardise(df):
    global target
    cols = df.columns
    for c in cols:
        if c != target:
            sd = df[c].std()
            mean = df[c].mean()
            
            df[c] = (df[c] - mean)/sd
    return df

dataset = standardise(dataset)
# print dataset.head()


# In[4]:


train, validate = np.split(dataset, [int(.8*len(dataset))])
train_row, train_col = train.shape
print "TRAIN: ", train.shape
print "Validate: ", validate.shape


# In[5]:


#init 12 theta b0,b1...b11
def initTheta(n):
    theta = np.zeros(n - 1 + 1) #remove chance of admit, serial no. add 1 b0 col
    return theta


# In[6]:


def makeX(df):
    if target in df.columns:
        df = df.drop([target], axis=1)
    X = df.values
    X = np.insert(X, 0, values=1, axis=1)
    return X

matrix = makeX(train)
matrix


# ### ONE VS ALL

# In[7]:


def gradientDescent(mat, actual, eta, length = None):
    mat_tr = mat.transpose()
    ilist = []
    clist = []
    theta = initTheta(df_c) #df_c: no of columns in original dataframe
    for i in range(0, 1000):
        pred = np.dot(mat, theta)
        loss = pred - actual
        if length is None:
            cost = np.sum((loss)**2) / (2 * train_row)
        else:
            cost = np.sum((loss)**2) / (2 * length)
            
        ilist.append(i)
        clist.append(cost)
        if length is None:
            gradient = np.dot(mat_tr, loss) / train_row
        else:
            gradient = np.dot(mat_tr, loss) / length
            
        theta = theta - eta * gradient
    return theta, ilist, clist
theta, iterations, cost = gradientDescent(matrix, train[target], 0.001)
print theta


# In[8]:


def findNewActual(df, val):
    new_y_actual = []
    for index, row in df.iterrows():
        if row['quality'] == val:
            new_y_actual.append(1)
        else:
            new_y_actual.append(0)
    return new_y_actual


# In[9]:


validate_m = makeX(validate)


# In[10]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[11]:


predictedDF = pd.DataFrame()
print predictedDF.shape


# In[12]:


def predict(train_mat, target_col, alpha, toBePredicted): 
    unique = target_col.unique()
   
    for u in unique:
        new_actual = findNewActual(train, u)
        reqd_theta, reqd_ilist, reqd_clist = gradientDescent(train_mat, new_actual, alpha)
        y_matrix = np.dot(toBePredicted, reqd_theta)
        predict = 1.0 / (1 + np.exp(-y_matrix)) #sigmoid function

        title = str(u)
        predictedDF[title] = predict
    class_predict = predictedDF.idxmax(axis = 1)

    return class_predict

y_predict_series = predict(matrix, train[target], 0.1, validate_m)
y_predict = y_predict_series.astype(int)

def actualClasses(validate):
    target_col = validate[target]
    target_m = target_col.values
    return target_m
y_actual = actualClasses(validate)


# In[13]:


results = confusion_matrix(y_actual, y_predict) 
print 'Confusion Matrix :'
print(results) 
print 'Accuracy Score :'
print accuracy_score(y_actual, y_predict)
print 'Precision Score :'
print precision_score(y_actual, y_predict, average=None).tolist()
print 'Recall Score :'
print recall_score(y_actual, y_predict, average=None).tolist()


# In[14]:


# print predictedDF


# ### ONE VS ONE

# In[15]:


def predictOne(train_mat, target_col, alpha, toBePredicted): 
    rows = toBePredicted.shape[0]
    prob_row = np.zeros(shape=(rows,11))
    #   class    0 1 2 3 4 5 6 7 8 9 10 
    #  rowno. 0[ [ c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 ] 
    #         1  [ c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 ] 
    #         :
    #      rows  [ c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 ] ]
    cnt = 0
    
    #runs for nC2 combinations
    for i in range(0,10):
        for j in range(i+1, 11):

            any_two = train[(train.quality == i) | (train.quality == j)]

            if len(any_two['quality'].unique()) != 2:
                continue
            else:
                any_matrix = makeX(any_two)
                y_actual_temp = findNewActual(any_two, i)
                len_matrix = any_matrix.shape[0]
                reqd_theta, reqd_ilist, reqd_clist = gradientDescent(any_matrix, y_actual_temp, alpha, len_matrix)
                y_matrix = np.dot(toBePredicted, reqd_theta)
                predict = 1.0 / (1 + np.exp(-y_matrix)) #sigmoid function

                class_predict = []
                for p in predict:
                    if p >= 0.6:
                        class_predict.append(i)
                    else: 
                        class_predict.append(j)
                cnt+=1

                for r in range(rows):
                    this_class = class_predict[r]
                    prob_row[r][this_class] += 1


    lis = []

    
    for r in range(rows):
        max_vote = prob_row[r].argmax()
        lis.append(max_vote)

    return lis


# In[16]:


y_predict = predictOne(matrix, train[target], 0.1, validate_m)
print "Confusion Matrix: "
results = confusion_matrix(y_actual, y_predict) 
print 'Accuracy Score :'
print accuracy_score(y_actual, y_predict)
print 'Precision Score :'
print precision_score(y_actual, y_predict, average=None).tolist()
print 'Recall Score :'
print recall_score(y_actual, y_predict, average=None).tolist()

