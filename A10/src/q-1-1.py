#!/usr/bin/env python
# coding: utf-8

# # Q1 Problem of Stock Prediction

# ## Part-1:  RNN

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


url = 'https://raw.githubusercontent.com/20188202010/SMAI_ASSIGNMENTS/master/GoogleStocks.csv?token=AonRVlOgsC5Fq9KbbuBNhikeTv0sqEz7ks5csX4FwA%3D%3D'


# In[5]:


dataset = pd.read_csv(url, thousands=',')
# dataset = pd.read_csv('../input_data/GoogleStocks.csv', thousands=',')
print "dataset.shape: ",dataset.shape, "cols: ", dataset.columns
dataset['date'] =pd.to_datetime(dataset.date)
dataset = dataset.sort_values(by='date')
dataset.head()


#  The features you will be using are:
#  - Average of the low and high of the Google Inc.  stock for the day.
#  - Volume of the stocks traded for the day.
#  <br>These will be used by you for predicting stock prices.  

# In[ ]:


features = ['Average','Volume']
target = 'open'


# In[ ]:


def findMean(df, col1, col2, new_col):
  
    df[new_col] = df[[col1, col2]].mean(axis=1)
    
    return df[new_col]


# In[ ]:


average_col = findMean(dataset, 'low', 'high', 'avg')


# In[ ]:


feature_set = dataset[['volume','avg']].values
open_set = dataset[target]


# In[ ]:


def divideTimestamp(df_x, ts, df_y=None, pred=False):
    x_train, y_train = [], []
    x_rows = df_x.shape[0]
    for i in range(ts, x_rows):
        x_train.append(df_x[i-ts:i, :])
        if pred==False:
          y_train.append(df_y[i])
        
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    return x_train, y_train


# In[ ]:


def makeInput(df_x, ts, df_y=None, flag=False):
  global features
  n = len(features)
  X,Y = divideTimestamp(df_x, ts, df_y, flag)
  
  X = np.reshape(X, (X.shape[0], X.shape[1], n)) #batchsize, timesteps, input features

  
  return X, Y


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten


# In[ ]:


def RNNModel(X, Y, num_units, num_layers, optimizer, epochs, batchsize):

  global features
  n = len(features)
  regressor = Sequential()

  regressor.add(LSTM(units = num_units, return_sequences = True, input_shape = (X.shape[1], n)))
  regressor.add(Dropout(rate = 1-0.2))
  
  regressor.add(LSTM(units = num_units, return_sequences = True))
  regressor.add(Dropout(rate = 1-0.2))
  
  if num_layers == 3:
    regressor.add(LSTM(units = num_units, return_sequences = True))
    regressor.add(Dropout(rate = 1-0.2))
  
  regressor.add(Flatten())
  
  regressor.add(Dense(units = 1))
  
  regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
  
  regressor.fit(X, Y, epochs = epochs, batch_size = batchsize)
  
  return regressor


# In[ ]:


def predict(x_attr, y_attr, regressor, ts, sc):
  
  inputs = x_attr[len(x_attr) - len(y_attr) - ts:,:]
  
  print "predict: ",
  X_test, none_ = makeInput(inputs, ts, flag=True)
  predicted_stock_price = regressor.predict(X_test)
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  return predicted_stock_price


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def plotGraph(actual_arr, pred_arr, title, sc):
  actual_arr = sc.inverse_transform(actual_arr)
  fig, axes = plt.subplots(figsize=(10, 4))
  axes.plot(pred_arr, color = 'm', label = 'Pred Open')
  axes.plot(actual_arr, color = 'g', label = 'Actual Open')
  axes.grid(True)
  axes.set_xlabel('Time')
  axes.set_ylabel('Open')
  axes.legend(loc='best')
  
  axes.set_title(title+' Open vs Time') 
  fig.savefig(title)


# In[ ]:


def testModel(attributes, target_col, num_layers,num_units,timestep,optimizer, epochs, batchsize ):

  sc = MinMaxScaler(feature_range = (0, 1))
  attributes = sc.fit_transform(attributes)
  target_col = target_col.reshape(-1,1)
  target_col = sc.fit_transform(target_col)
  
  train, validate = np.split(attributes, [int(.8*len(attributes))])
  train_y, validate_y = np.split(target_col, [int(.8*len(target_col))])
  
  
  X, Y = makeInput(train, timestep, df_y=train_y)
  model = RNNModel(X,Y,num_units, num_layers, optimizer, epochs, batchsize)
  prediction = predict(attributes, validate, model, timestep, sc)

  title = "("+str(num_layers)+" , "+str(num_units)+" , "+str(timestep)+")"
  plotGraph(validate_y, prediction, title, sc)
  return prediction


# ### 20 TIMESTEPS

# #### Two Layer

# **1. 2 layers 30 hidden nodes 20 timesteps**

# In[21]:


prediction = testModel(feature_set, open_set, 2, 30, 20, 'adam', 125, 16) 


# **2. 2 layers 50 hidden nodes 20 timesteps**

# In[22]:


prediction = testModel(feature_set, open_set, 2, 50, 20, 'adam', 150, 64) 


# **3. 2 layers 80 hidden nodes 20 timesteps**

# In[23]:


prediction = testModel(feature_set, open_set, 2, 80, 20, 'adam', 20, 128) 


# #### Three Layer

# **4. 3 layers 30 hidden nodes 20 timesteps**

# In[26]:


prediction = testModel(feature_set, open_set, 3, 30, 20, 'adam', 100, 64) 


# **5. 3 layers 50 hidden nodes 20 timesteps**

# In[27]:


prediction = testModel(feature_set, open_set, 3, 50, 20, 'adam', 75, 32) 


# **6. 3 layers 80 hidden nodes 20 timesteps**

# In[28]:


prediction = testModel(feature_set, open_set, 3, 80, 20, 'adam', 50, 128) 


# ### 50 TIMESTEPS

# #### Two Layer

# **7. 2 layers 30 hidden nodes 50 timesteps**

# In[29]:


prediction = testModel(feature_set, open_set, 2, 30, 50, 'adam', 200, 32) 


# **8. 2 layers 50 hidden nodes 50 timesteps**

# In[30]:


prediction = testModel(feature_set, open_set, 2, 50, 50, 'adam', 80, 8)


# **9. 2 layers 80 hidden nodes 50 timesteps**

# In[31]:


prediction = testModel(feature_set, open_set, 2, 80, 50, 'adam', 50, 16) 


# #### Three Layer

# **10. 3 layers 30 hidden nodes 50 timesteps**

# In[32]:


prediction = testModel(feature_set, open_set, 3, 30, 50, 'adam', 60, 32)


# **11. 3 layers 50 hidden nodes 50 timesteps**

# In[33]:


prediction = testModel(feature_set, open_set, 3, 50, 50, 'adam', 30, 64) 


# **12. 3 layers 80 hidden nodes 50 timesteps**

# In[34]:


prediction = testModel(feature_set, open_set, 3, 80, 50, 'adam', 25, 32) 


# ### 75 TIMESTEPS

# #### Two Layer

# **13. 2 layers 30 hidden nodes 75 timesteps**

# In[35]:


prediction = testModel(feature_set, open_set, 2, 30, 75, 'adam', 100, 32)


# **14. 2 layers 50 hidden nodes 75 timesteps**

# In[36]:


prediction = testModel(feature_set, open_set, 2, 50, 75, 'adam', 50, 64) 


# **15. 2 layers 80 hidden nodes 75 timesteps**

# In[37]:


prediction = testModel(feature_set, open_set, 2, 80, 75, 'adam', 30, 16) 


# #### Three Layer

# **16. 3 layers 30 hidden nodes 75 timesteps**

# In[38]:


prediction = testModel(feature_set, open_set, 3, 30, 75, 'adam', 30, 32)


# **17. 3 layers 50 hidden nodes 75 timesteps**

# In[39]:


prediction = testModel(feature_set, open_set, 3, 50, 75, 'adam', 50, 32) 


# **18. 3 layers 80 hidden nodes 75 timesteps**

# In[40]:


prediction = testModel(feature_set, open_set, 3, 80, 75, 'adam', 50, 16) 

