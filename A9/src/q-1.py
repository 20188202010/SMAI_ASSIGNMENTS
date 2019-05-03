#!/usr/bin/env python
# coding: utf-8

# ## Q1: Anomaly Detection

# ### **PART 1**: Apply dimensionality reduction on the dataset using:

# Common preprocessing

# In[47]:


import numpy as np
import pandas as pd
from math import sqrt, isnan, ceil
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import mean_squared_error


# In[2]:


dataset = pd.read_csv('../input_data/data.csv')


# In[3]:


rows, cols = dataset.shape
target = 'xAttack'
classes = dataset[target].nunique() #no of unique classes 0-9
actual_classes = dataset[target].values #all actual classes of both train and validate

print "classes: ",classes, "rows x cols: ",rows,"x",cols


# Standardising the dataset

# In[4]:


def normalise(df):
    col_list = df.columns
    for c in col_list:
        if c != target:
            mini = df[c].min()
            maxi = df[c].max()
            try:
            
                df[c] = (df[c] - mini)/(maxi-mini)
            except:
                df[c] = 0
            
    return df

dataset = normalise(dataset)


# Convert dataframe to matrix without target column

# In[5]:


def makeX(df):
    if target in df.columns:
        df = df.drop([target], axis=1)
    X = df.values
    return X


# In[6]:


matrix = makeX(dataset) 


# Function to make mini batches of input matrix and output matrix and specified batch size.

# In[7]:


def makeMiniBatches(inputs, batch_size):
    mini_batches = []
    for i in range (0, inputs.shape[0], batch_size):
        x = inputs[i:i + batch_size, :]
        mini_batches.append([x])
    return mini_batches


# In[8]:


def plot(x, y, x_label, y_label, title):
    fig, axes = plt.subplots(figsize=(12,3))
    axes.plot(x, y, 'r')
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title);
    axes.grid(True)


# Function to train autoencoder

# In[9]:


class NeuralNetworkMultiple:
    def __init__(self, x, n, node_list):
        self.inpNodes  = x
        self.outNodes = x
        self.hNodesList = node_list #all hidden layers  DO NOT have same size
        self.numHiddenLayers = n 
        
        if len(self.hNodesList) != self.numHiddenLayers:
            raise Exception('Number of hidden nodes and layers mismatch.')

        self.weights = [] #list of numpy arrays
        self.bias    = [] # "same^"

        '''
        i/p -> h0 -> h1 -> h2 -> o/p    n=3
        w0    w1    w2    w3
        '''

        weights_in_h    = np.random.randn(self.inpNodes, self.hNodesList[0])  * sqrt(2.0/self.inpNodes) #w0
        bias_1          = np.random.randn(self.hNodesList[0])

        self.weights.append(weights_in_h)
        self.bias.append(bias_1)

        for layer in range(n-1):

            weights_intermediate = np.random.randn(self.hNodesList[layer], self.hNodesList[layer+1])  * sqrt(2.0/self.hNodesList[layer]) #w1 to wn-1
            bias_intermediate    = np.random.randn(self.hNodesList[layer+1])
            
            self.bias.append(bias_intermediate)
            self.weights.append(weights_intermediate)

        weights_h_op    = np.random.randn(self.hNodesList[n-1], self.outNodes)  * sqrt(2.0/self.hNodesList[n-1])  #b/w last hidden and o/p
        bias_2          = np.random.randn(self.outNodes)
    
        self.bias.append(bias_2)
        self.weights.append(weights_h_op) #wn

    def activations(self, func_name, var):
        if func_name == "sigmoid":
            val = 1/(1 + np.exp(-var))
        elif func_name == "tanh":
            val = np.tanh(var)
        elif func_name == "relu":
            val = var * (var > 0)
        elif func_name == "linear":
            return var
        return val
    
    
    def derivatives(self, func_name, var):
        if func_name == "sigmoid":
            der = self.activations(func_name, var) * (1 - self.activations(func_name, var))
        elif func_name == "tanh":
            der = 1 - np.tanh(var)**2
        elif func_name == "relu":
            der = 1. * (var > 0)
        elif func_name == "linear":
            der = np.ones(var.shape)
        return der
    
    def feedForward(self, input_mat, activation_func):
        self.input_features = input_mat
        self.z2 = []
        self.a2 = []

        self.activation_func = activation_func
        '''
        w0         w1         w2         w3
        i/p ------> h0 ------> h1 ------> h2 ------> o/p -->   n=3
        z0  a0   z1    a1   z2   a2    z3   activation
        '''
        z_0 = np.dot(input_mat, self.weights[0]) + self.bias[0] 
        self.z2.append(z_0)
        a_0  = self.activations(activation_func, z_0)
        
        self.a2.append(a_0)
        bottle_neck_idx = self.numHiddenLayers//2
        num = self.numHiddenLayers-1

        for layer in range(num): #0,1
            z_intermediate = np.dot(self.a2[layer], self.weights[layer + 1]) + self.bias[layer + 1] 
            
            a_intermediate = self.activations(activation_func, z_intermediate)
            
            self.z2.append(z_intermediate)
            self.a2.append(a_intermediate)
        z_final = np.dot(self.a2[num], self.weights[num + 1]) + self.bias[num + 1]
        self.z2.append(z_final)
        y_pred = self.activations(activation_func,z_final) #final output

        return y_pred, self.a2[bottle_neck_idx]
    

    def backprop(self, y_pred, y_act, activation_func):
        num = self.numHiddenLayers
        
        cost = np.multiply(y_pred - y_act, self.derivatives(activation_func, y_pred))
        '''
                 w0         w1         w2         w3
        i/p <----- h0 <----- h1 <----- h2 <----- o/p <--   n=3
                 z0    a0   z1    a1   z2   a2    z3   activation
        '''
        weight_der = []
        bias_der = []
        for layer in range(num, 0, -1):

            z3_wrt_who = self.a2[layer - 1]
            weight_der.append( np.dot(z3_wrt_who.transpose(), cost) )
            bias_der.append( np.average(cost, axis = 0))
            cost = np.dot(cost, self.weights[layer].transpose())

        
        cost *= self.derivatives(activation_func,z3_wrt_who)
              
        weight_der.append( np.dot(self.input_features.transpose(), cost))
        bias_der.append( np.average(cost, axis = 0))
     
        weight_der = weight_der[::-1] 
        bias_der = bias_der[::-1]

        return weight_der, bias_der
    
    def updateWeights(self, weight, bias, alpha):
        num = self.numHiddenLayers

        for layer in range(num+1):
            self.weights[layer] -= alpha * weight[layer]
            self.bias[layer] -= alpha * bias[layer].sum(axis=0)
  
    def loss(self, y_pred, y_act):
        loss = mean_squared_error(y_act ,y_pred)
        print ":", loss,") ",
        return loss


# In[10]:


def trainNetwork(hiddenLayers, node_list, iterations, alpha, batchsize, function_name):
    num_points_to_plot = 50
    mod = ceil(iterations // num_points_to_plot)
    if mod == 0:
        mod = 1
    ep_list = [] 
    err = []
    #initialize a NN with random weights
    DeepAutoEncoder = NeuralNetworkMultiple(cols - 1, hiddenLayers, node_list) #<x, n, node_list>

    for i in range(iterations):
        if i % mod == 0:
            print "iter: ",i,
        else:
            print "iter: ",i,
        #make mini batches of the training set
        mini_batches = makeMiniBatches(matrix, batchsize) #returns a list of list: [ [ <i/p matrix of 1st mini batch>], ... ]

        #create empty matrices for all predictions and actual values
        all_pred = np.array([])
        all_act = np.array([])
        all_reduced = np.array([])
        
        for one_mini_batch in mini_batches:
            inp = one_mini_batch[0]
       
            #prediction for this mini batch
            y_hat, reduced_dim = DeepAutoEncoder.feedForward(inp, function_name)
            #do back propagation
            c1, c2 = DeepAutoEncoder.backprop(y_hat, inp, function_name)
             #update the weights
            DeepAutoEncoder.updateWeights(c1, c2, alpha)

            #concatenate prediction and actual to bigger matrix
            all_pred = np.vstack([all_pred, y_hat]) if all_pred.size else y_hat
            all_act = np.vstack([all_act, inp]) if all_act.size else inp
            all_reduced = np.vstack([all_reduced, reduced_dim]) if all_reduced.size else reduced_dim
            
        if i % mod == 0:
            print "(",i,
            ep_list.append(i)
            err.append(DeepAutoEncoder.loss(all_pred, all_act))
            print
    plot(ep_list, err, 'iteration', 'error', 'iteration vs error '+function_name)
    return DeepAutoEncoder, all_reduced


# ####  3-Layer autoencoder consisting of input , output and bottleneck layers

# #### a. input and output layers have linear activation functions

# In[11]:


#<(hiddenLayers, node_list, iterations, alpha, batchsize, function_name)>
linear_1, reduced_df = trainNetwork(1, [14], 5, 0.00001, 200, 'linear') #<hNeurons, iterations, alpha, batchsize, function_name>


# In[12]:


print reduced_df.shape


# #### b. input and output layers layers have non-linear activation functions

# *SIGMOID 1 layer*

# In[16]:


sigmoid_1, reduced_df_sigmoid1 = trainNetwork(1, [14], 10, 0.001, 200, 'sigmoid') 


# *TANH 1 layer*

# In[17]:


tanh1, reduced_df_tanh1 = trainNetwork(1, [14], 5, 0.001, 200, 'tanh') 


# *ReLU 1 layer*

# In[18]:


relu1, reduced_df_relu1 = trainNetwork(1, [14], 5, 0.00001, 200, 'relu') #<hNeurons, iterations, alpha, function_name>


# ####  Deep autoencoder

# In[24]:


hidden_layer_node_list = [26,20,14,20,26]
hidden_layer_size = len(hidden_layer_node_list)


# *Sigmoid*

# In[20]:


#<hiddenLayers, nodes, iterations, alpha, batchsize, function_name>


# In[25]:


sigmoid_mul, reduced_df_deepSig = trainNetwork(hidden_layer_size, hidden_layer_node_list, 5, 0.0001, 32, 'sigmoid') 


# *Tanh*

# In[22]:


#<hiddenLayers, nodes, iterations, alpha, batchsize, function_name>


# In[26]:


tanh_mul, reduced_df_deepTanh = trainNetwork(hidden_layer_size, hidden_layer_node_list, 5, 0.001, 50, 'tanh') 


# *ReLU*

# In[27]:


#<hiddenLayers, nodes, iterations, alpha, batchsize, function_name>


# In[28]:


relu_mul, reduced_df_deepRelu = trainNetwork(hidden_layer_size, hidden_layer_node_list, 5, 0.00001, 32, 'relu') 


# In[29]:


print reduced_df_deepRelu.shape


# - Common utility functions: make df, plot pie chart and find % purity for parts 2 to 5

# In[30]:


def makeNewDf(pred, actual=actual_classes):
    matrix = np.vstack((pred,actual))
    df = pd.DataFrame(matrix.transpose(), columns = ['PredictedClass', 'ActualLabel'])
    return df


# In[31]:


# Data to plot
def plotPieChart(values, cluster_num, method, unique_attacks):
            
    y=np.array(values)
    percent = 100.0*y/y.sum()
    legends = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(unique_attacks, percent)]
    
    with plt.style.context({"axes.prop_cycle" : plt.cycler("color", plt.cm.tab20.colors)}):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.axis('equal')
        title = method + ": cluster no."+ str(cluster_num)
        ax.set_title(title)
        wedges, labels = ax.pie(y, shadow=True, startangle=90)
        wedges, legends, dummy =  zip(*sorted(zip(wedges, legends, y),key=lambda x: x[2],reverse=True))
        ax.legend(wedges, legends, loc='best', bbox_to_anchor=(-0.1, 1.),fontsize=12)


# In[64]:


def makeTable(row, col):
    # row: [ [<row1>] [<row2>] [<row 3>]..]
    print "total samples: ",row[0][5]+row[1][5]+row[2][5]+row[3][5]+row[4][5]
    print tabulate( [
                     ['Cluster 0', row[0][0], row[0][1], row[0][2], row[0][3], row[0][4], row[0][5]], 
                     ['Cluster 1', row[1][0], row[1][1], row[1][2], row[1][3], row[1][4], row[1][5]],
                     ['Cluster 2', row[2][0], row[2][1], row[2][2], row[2][3], row[2][4], row[2][5]],
                     ['Cluster 3', row[3][0], row[3][1], row[3][2], row[3][3], row[3][4], row[3][5]],
                     ['Cluster 4', row[4][0], row[4][1], row[4][2], row[4][3], row[4][4], row[4][5]]
                    ],
                    
                   headers=['Attack', col[0], col[1], col[2], col[3], col[4], 'Num Samples']
                  )


# In[67]:


def findFraction(method, df): #method: KMeans, GMM, Hierarchial etc
    all_count = []
    unique_attacks = df['ActualLabel'].unique()
    for i in range(classes):
        this_cluster = df[df['PredictedClass']== i]
        num_samples = this_cluster.shape[0]
        fraction_list = []
        
        count_i = []
        for u in unique_attacks:
            count = this_cluster[this_cluster['ActualLabel']==u].shape[0]
            try:
                fraction = float(count)/num_samples
            except:
                fraction = 0
            count_i.append(count)
            fraction_list.append(fraction*100)
        count_i.append(num_samples)
        all_count.append(count_i)
        
        if all(i == 0 for i in fraction_list):
            print "NO CLUSTERS FOR CENTER ",i
        else:
            
            plotPieChart(fraction_list, i, method, unique_attacks)
    makeTable(all_count, unique_attacks)


# ## PART 2: Use the reduced dimensions from all the techniques in the firstpart and perform K-means clustering with k equal to five(number of classes in thedata).  Also calculate the purity of clusters with given class label

# In[68]:


from sklearn.cluster import KMeans

def doKmeans(df):
    kmeans = KMeans(n_clusters=classes, random_state=0).fit(df)
    kmeans_labels = kmeans.labels_
    
    pred_act_df = makeNewDf(kmeans_labels)
   
    findFraction("KMeans", pred_act_df)


# In[69]:


doKmeans(reduced_df_deepRelu)


# ### PART 3: Perform GMM (with five Gaussian) on the reduced dimensions from first part and calculate the purity of clusters.  You can use python library for GMM

# In[71]:


from sklearn.mixture import GaussianMixture
def doGMM(df):
    gmm = GaussianMixture(n_components=classes).fit(df)
    gmm_labels = gmm.predict(reduced_df_deepRelu)
    
    pred_act_df = makeNewDf(gmm_labels)
    findFraction("GMM", pred_act_df)
doGMM(reduced_df_deepRelu)


# ### PART 4: Perform  Hierarchical  clustering  with  single-linkage  and  five clusters on the reduced dimensions from all the techniques in the first part and cal-culate the purity of clusters.You can use python library for hierarchical clustering

# In[72]:


from sklearn.cluster import AgglomerativeClustering
def doAgglomerative(df):
    rows_to_sample = 20000
    more_reduced = df[:rows_to_sample,0:14]
    cluster = AgglomerativeClustering(n_clusters = classes, affinity = 'euclidean', linkage = 'single')
    agglo_labels = cluster.fit_predict(more_reduced)
    
    
    pred_act_df = makeNewDf(agglo_labels, actual = actual_classes[:rows_to_sample])
    findFraction("Agglomerative", pred_act_df)
doAgglomerative(reduced_df_deepRelu)


# ### PART 5:Create a pie chart comparing purity of different clustering methods you have tried for all classes for the different autoencoders

# In[73]:


doKmeans(reduced_df_relu1)


# In[74]:


doGMM(reduced_df_relu1)


# In[75]:


doAgglomerative(reduced_df_relu1)

