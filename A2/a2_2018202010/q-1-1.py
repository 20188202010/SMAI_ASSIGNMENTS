#!/usr/bin/env python
# coding: utf-8

# In[424]:


import numpy as np
import pandas as pd
import math #sqrt
import sys #max


# In[425]:


import matplotlib.pyplot as plt


# In[426]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score


# In[427]:


#normalise dataframe
def normalise(df, cols):
    for c in cols:
        maxi = df[c].max()
        mini = df[c].min()
        diff = maxi - mini
        df[c] = (df[c] - mini)/diff
    return df


# ### KNN classifier

# findKnn parameters:
# - row: one row of validation
# - df_t: training dataset
# - cols: numerical attribute list
# - target: which class the attribute belongs to, eg class for robot, species for iris
# - label: unique identifier for attribute
# - findNeigh: k neighbour finder fn
# - distanceMeasure: which distance is to be used

# In[428]:


def findKnn(row, df_t, cols, target, findNeigh, distanceMeasure, k, default=0):
    listOfList=[]
    
    #iterate on each row of training set.
    for i, r in df_t.iterrows():
        temp = distanceMeasure(row,r,cols,target) #temp: [ <distance>, <class>]
        listOfList.append(temp)
    
    listOfList.sort()
#     print "lol: ",listOfList
    pred = findNeigh(listOfList, k)

#     print "pred: ", pred, "actual: ", row[target]
    return pred


def helper_knn(df, df_t, predict_col, cols, target, findNeigh, distanceMeasure, k):
    df[predict_col] = df.apply(findKnn, axis=1, args=(df_t, cols, target, findNeigh, distanceMeasure, k, 0))
    return df[predict_col]


# ###  Report precision, recall, f1 score and accuracy. 

# In[429]:


#returns Confusion Matrix
def createCM(predicted, actual):
    pred = pd.Series(predicted, name='Predicted')
    actu = pd.Series(actual,    name='Actual')
    conf = pd.crosstab(actu, pred)
    return conf


# In[430]:


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


# In[431]:


def findAllK(df, df_t, cols, target, predict_col_name, funNeigh, funDistance):
    n = math.sqrt( len(df_t) ) + 1 #k is less than equal to root n

    length = int(n)

    all_predictions = []
    max_acc = -sys.maxint - 1
    actual = df[target]
    x_axis = range(1, length,2)
    print "x: ", x_axis,
    for i in x_axis:

        predict_col = predict_col_name + str(i) #eg for r1, v_robo_<1>
#         print "K: ",i
        helper_knn(df, df_t, predict_col, cols, target, funNeigh, funDistance, i)
        all_predictions.append(df[predict_col])
    
    y_axis = []
    count = 1
    max_k = 0
    for one_prediction in all_predictions:
        confusion_matrix = createCM(one_prediction, actual)
        accuracy = findMeasures(confusion_matrix)[0]
        if accuracy > max_acc:
            max_acc = accuracy
            max_k = count
        y_axis.append(accuracy)
        if 'iris' in predict_col_name:
            count += 2
        elif 'robo' in predict_col_name:
            count += 1
    print "y: ",y_axis, "maxK: ",max_k
    return x_axis, y_axis, max_k


# In[432]:


def plotGraphs(x1, y1, x2, y2, x3, y3,part):
    fig, axes = plt.subplots(figsize=(7, 7))
    axes.plot(x1, y1, label="euclidean distance")
    axes.plot(x2, y2, label="manhattan distance")
    axes.plot(x3, y3, label="chebyshev distance")
    axes.grid(True)
    
    axes.set_xlabel('k')
    axes.set_ylabel('Accuracy %')
    axes.legend(loc='best')
    name = part+str(runs)+'.png'
    axes.set_title('k vs Accuracy')
    fig.savefig(name)


# In[433]:


def completeAnalysis(df_v, df_t, predict_col, num_col, target, funNeigh, funDist, bestK):
    df_v[predict_col] = helper_knn(df_v, df_t, predict_col, num_col, target, funNeigh, funDist, bestK)
    predicted_val = df_v[predict_col]
    actual_val = df_v[target]
    confusion_mat = createCM(predicted_val, actual_val)
    print "Confusion Matrix:"
    print confusion_mat
    a,p,r,f = findMeasures(confusion_mat)
    print "Accuracy: ",a*100, "\nPrecision: ", p*100,"\nRecall:", r*100,"\nF1Score: ",f
    


# ### Different distance measures

# In[434]:


def findEuclidean(unknown_class, known_class ,cols, target):
    sum_of_squares = 0
    class_dist = []
    for i in cols:
        xi = (unknown_class[i] - known_class[i])**2
        sum_of_squares += xi
    
    dist = math.sqrt(sum_of_squares)
    class_dist = [dist, known_class[target]]
    return class_dist


# In[435]:


def findChebyshev(unknown_class, known_class ,cols, target):
    class_dist = []
    maxi = -sys.maxint - 1
    for i in cols:
        xi = abs(unknown_class[i] - known_class[i])
        if maxi < xi:
            maxi = xi
    class_dist = [maxi, known_class[target]]
    return class_dist


# In[436]:


def findManhattan(unknown_class, known_class ,cols, target):
    dist = 0
    class_dist = []
    for i in cols:
        xi = abs(unknown_class[i] - known_class[i])
        dist += xi
    
    class_dist = [dist, known_class[target]]
    return class_dist


# ### Compare SciKit

# In[437]:


def compareScikit(df,target,identifier=None):
    le = preprocessing.LabelEncoder()
    if identifier is not None:
        df[identifier] = le.fit_transform(df[identifier])
    # dataset_R1=dataset_R1.drop('index',1)
    if identifier is not None:
        cols = [col for col in df.columns if col not in [target,identifier]]
    else:
        cols = [col for col in df.columns if col not in [target]]
    data = df[cols]
    target = df[target]
    data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.20, random_state = 10)

    scaler = StandardScaler()  
    scaler.fit(data_train)

    data_train = scaler.transform(data_train)  
    data_test = scaler.transform(data_test)  
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(data_train, target_train) 
    y_pred = classifier.predict(data_test) 
    error = []

    for i in range(1, int(math.sqrt(len(data_train)))):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(data_train, target_train)
        pred_i = knn.predict(data_test)
        error.append(np.mean(pred_i != target_test))
#     plt.figure(figsize=(7, 7))  
#     plt.plot(range(1, int(math.sqrt(len(data_train)))), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
#     plt.title('Error Rate K Value')  
#     plt.xlabel('K Value')  
#     plt.ylabel('Mean Error')
    score = accuracy_score(target_test, y_pred)
    print "score",score


# #### Part-1: Robot1 & Robot2

# In[438]:


robot_num = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'] #numerical attr for robot


# findKNeighbours(class_dist, k) function returns label of unknown class based on k nearest neighbours
# - class_dist: list of list. Each sublist has distance and corresponding class labe
# - k: value of k

# In[439]:


def findKNeighbours(class_dist, k):
    #slice first K elements
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
    return label


# ### ROBOT 1

# In[440]:


dataset_R1 = pd.read_csv("RobotDataset/Robot1", header = None, delim_whitespace=True)
dataset_R1.columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
dataset_R1 = dataset_R1.sample(frac=1)
dataset_R1 = normalise(dataset_R1, robot_num)
train_R1, validate_R1 = np.split(dataset_R1, [int(.8*len(dataset_R1))])


# Test file Input

# In[441]:


test_dR1 =raw_input("Enter test file for Robot 1: ")
test_dR1 = pd.read_csv(test_dR1, header = None, delim_whitespace=True)
test_dR1.columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
test_dR1 = normalise(test_dR1, robot_num)


# Finds accuracy for all values of k

# In[442]:


print "EUCLIDEAN:"
xe,ye,ke = findAllK(validate_R1, train_R1, robot_num, 'class', 'v_robo_', findKNeighbours, findEuclidean)


# In[443]:


print "MANHATTAN:"
xm,ym,km = findAllK(validate_R1, train_R1, robot_num, 'class', 'v_robo_', findKNeighbours, findManhattan)


# In[444]:


print "CHEBYSHEV:"
xc,yc,kc = findAllK(validate_R1, train_R1, robot_num, 'class', 'v_robo_', findKNeighbours, findChebyshev)


# In[445]:


plotGraphs(xe, ye, xm, ym, xc, yc, 'robot1')


# Accuracy, Precision, Recall, F1 Score for best value of k

# In[446]:


print "Euclidean", ke
completeAnalysis(validate_R1, train_R1, 'validate_R1',robot_num, 'class', findKNeighbours, findEuclidean, ke)


# In[447]:


print "Manhattan", km
completeAnalysis(validate_R1, train_R1, 'validate_R1',robot_num, 'class', findKNeighbours, findManhattan, km)


# In[448]:


print "Chebyshev", kc
completeAnalysis(validate_R1, train_R1, 'validate_R1',robot_num, 'class', findKNeighbours, findChebyshev, kc)


# TESTING

# In[449]:


helper_knn(test_dR1, train_R1, 'robot1_test', robot_num, 'class', findKNeighbours, findEuclidean, ke)


# In[450]:


compareScikit(dataset_R1, 'class', 'Id')


# ### ROBOT 2

# In[451]:


dataset_R2 = pd.read_csv("RobotDataset/Robot2", header = None, delim_whitespace=True)
dataset_R2.columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
dataset_R2 = dataset_R2.sample(frac=1)
dataset_R2 = normalise(dataset_R2, robot_num)
train_R2, validate_R2 = np.split(dataset_R2, [int(.8*len(dataset_R2))])


# Test file input

# In[452]:


test_dR2 = raw_input("Enter test file for Robot 2: ")
test_dR2 = pd.read_csv(test_dR2, header = None, delim_whitespace=True)
test_dR2.columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
test_dR2 = normalise(test_dR2, robot_num)


# Finds accuracy for all values of k

# In[ ]:


print "EUCLIDEAN"
xe2,ye2,ke2 = findAllK(validate_R2, train_R2, robot_num, 'class', 'v_robo2_', findKNeighbours, findEuclidean)


# In[ ]:


print "MANHATTAN"
xm2,ym2,km2 = findAllK(validate_R2, train_R2, robot_num, 'class', 'v_robo2_', findKNeighbours, findManhattan)


# In[ ]:


print "CHEBYSHEV"
xc2,yc2,kc2 = findAllK(validate_R2, train_R2, robot_num, 'class', 'v_robo2_', findKNeighbours, findChebyshev)


# In[ ]:


plotGraphs(xe2, ye2, xm2, ym2, xc2, yc2,'robot2')


# Accuracy, Precision, Recall, F1 Score for best value of k

# In[ ]:


print "Euclidean"
completeAnalysis(validate_R2, train_R2, 'validate_R2',robot_num, 'class', findKNeighbours, findEuclidean, ke2)


# In[ ]:


print "Manhattan"
completeAnalysis(validate_R2, train_R2, 'validate_R2',robot_num, 'class',  findKNeighbours, findManhattan, km2)


# In[ ]:


print "Chebyshev"
completeAnalysis(validate_R2, train_R2, 'validate_R2',robot_num, 'class',  findKNeighbours, findChebyshev, kc2)


# TESTING

# In[ ]:


helper_knn(test_dR2, train_R2, 'robot2_test', robot_num, 'class',  findKNeighbours, findEuclidean, ke2)


# In[ ]:


compareScikit(dataset_R2, 'class', 'Id')


# #### Part 2: Iris.csv The data set consists of samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

# Read the dataset

# In[ ]:


dataset_iris = pd.read_csv("Iris/Iris.csv", header=None)
dataset_iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


# Add column names

# In[ ]:


iris_num = ['sepal_length', 'sepal_width','petal_length', 'petal_width']


# Randomise dataset

# In[ ]:


dataset_iris = dataset_iris.sample(frac=1)


# Normalise the dataframe

# In[ ]:


dataset_iris = normalise(dataset_iris, iris_num)


# Split Dataframe

# In[ ]:


train_iris, validate_iris = np.split(dataset_iris, [int(.8*len(dataset_iris))])


# Test file input

# In[ ]:


test_iris = raw_input("Enter test file for Iris: ")
test_iris = pd.read_csv(test_iris, header = None)
test_iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
test_iris = normalise(test_iris, iris_num)


# findKNeighboursIris(class_dist, k) function returns label of unknown class based on k nearest neighbours
# - class_dist: list of list. Each sublist has distance and corresponding class labe
# - k: value of k

# In[ ]:


def findKNeighboursIris(class_dist, k):
    #slice first K elements
    topK = class_dist[0:k]
    count0 = 0
    count1 = 0
    count2 = 0
    #traverse the second list and count no of 1,0
    for i in topK:
        if i[1] == 'Iris-setosa':
            count0 += 1
        elif i[1] == 'Iris-virginica':            
            count1 += 1
        elif i[1] == 'Iris-versicolor':          
            count2 += 1
    
    if (count0 >= count1) and (count0 >= count2): 
        label = 'Iris-setosa'
  
    elif (count1 >= count0) and (count1 >= count2): 
        label = 'Iris-virginica'
    else: 
        label = 'Iris-versicolor' 
    return label


# Find accuracy for all values of k

# In[ ]:


print "Euclidean:"
xei,yei,kei = findAllK(validate_iris, train_iris, iris_num, 'species',  'v_iris_',findKNeighboursIris, findEuclidean)


# In[ ]:


print "Manhattan"
xmi,ymi,kmi = findAllK(validate_iris, train_iris, iris_num, 'species',  'v_iris_',findKNeighboursIris, findManhattan)


# In[ ]:


print "Chebyshev"
xci,yci,kci = findAllK(validate_iris, train_iris, iris_num, 'species', 'v_iris_',findKNeighboursIris, findChebyshev)


# Plot the graph

# In[ ]:


plotGraphs(xei,yei, xmi,ymi, xci, yci,'iris')


# Accuracy, Precision, Recall, F1 Score for best value of k

# In[ ]:


print "Euclidean"
completeAnalysis(validate_iris, train_iris, 'validate_iris',iris_num, 'species', findKNeighboursIris, findEuclidean, kei)


# In[ ]:


print "Manhattan"
completeAnalysis(validate_iris, train_iris, 'validate_iris',iris_num, 'species', findKNeighboursIris, findManhattan, kmi)


# In[ ]:


print "Chebyshev"
completeAnalysis(validate_iris, train_iris, 'validate_iris',iris_num, 'species', findKNeighboursIris, findChebyshev, kci)


# TESTING

# In[ ]:


helper_knn(test_iris, train_iris, 'iris_test', iris_num, 'species',findKNeighboursIris, findEuclidean, kei)


# In[ ]:


compareScikit(dataset_iris, 'species',None)


# #### Possible Reasons for Better Performance

# There is no training phase included in KNN classifier. So it performs faster than other classifiers. This is also effective when training data is large
# No assumptions are made about the data
