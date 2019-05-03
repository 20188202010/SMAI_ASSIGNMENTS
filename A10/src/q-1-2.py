#!/usr/bin/env python
# coding: utf-8

# # Q1 Problem of Stock Prediction
# ## Part-2: HMM
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


url = 'https://raw.githubusercontent.com/20188202010/SMAI_ASSIGNMENTS/master/GoogleStocks.csv?token=AonRVlOgsC5Fq9KbbuBNhikeTv0sqEz7ks5csX4FwA%3D%3D'


# In[3]:


dataset = pd.read_csv(url, thousands=',')
dataset['date'] =pd.to_datetime(dataset.date)
dataset = dataset.sort_values(by='date')
print dataset.shape


# In[4]:


def findMean(df, col1, col2, new_col):
  
    df[new_col] = df[[col1, col2]].mean(axis=1)
    
    return df[new_col]


# In[5]:


average_col = findMean(dataset, 'low', 'high', 'avg')


# In[6]:


feature_set = dataset[['volume','avg']].values
open_set = dataset['open'].values
open_set = open_set.reshape(-1,1)
print open_set.shape, feature_set.shape


# In[7]:


train, validate = np.split(feature_set, [int(.8*len(feature_set))])
print train.shape, validate.shape
train_y, validate_y = np.split(open_set, [int(.8*len(open_set))])
print train_y.shape, validate_y.shape


# In[8]:


def extractFeatures(df = train, df_y = train_y):
    avg_arr = df[:,1].reshape(-1,1)
    
    avg_by_open = (df_y-avg_arr)/df_y

    return avg_by_open, avg_arr


# In[9]:


feature_vector, avg_col = extractFeatures()
print "feature_vector.shape: ",feature_vector.shape


# In[10]:


from hmmlearn.hmm import GaussianHMM


# In[11]:


def makeHMM_model(components):
    global feature_vector
    hmm = GaussianHMM(n_components=components)
    hmm = hmm.fit(feature_vector)
    return hmm


# In[12]:


from itertools import product


# In[23]:


def findAllCombinations(maxi,steps):
        mini = -1 * maxi
        avg_rn = np.linspace( mini,maxi, steps)
        all_outcomes = avg_rn.reshape(-1,1)

        return all_outcomes
    


# In[21]:


def findMaxLikelihood(outcomes_list, day_index, ts, hmm):
        score_list = []
        start = max(0, day_index - ts)
        end = max(0, day_index - 1)
        previous_data_y = validate_y[start:end]
        previous_data_x = validate[start:end]

        previous_data_features,_ = extractFeatures(df = previous_data_x ,df_y = previous_data_y)
        
        for one_outcome in outcomes_list:
            total_data = np.vstack((previous_data_features, one_outcome))
            score = hmm.score(total_data)
            score_list.append(score)
        

        max_likelihood_outcome = outcomes_list[np.argmax(score_list)]
 
        return max_likelihood_outcome


# In[15]:


def predictOneOpen(outcomes_list, day_index, ts, hmm):
       
       global validate
       avg_y = validate[day_index][1]
       average_change = findMaxLikelihood(outcomes_list, day_index, ts, hmm)
       
       return  avg_y *(1 + average_change)


# In[16]:


def predictAllOpen(outcomes_list, ts, hmm):
        pred_open = []
        for day_index in range(len(validate_y)):
            pred_open.append(predictOneOpen(outcomes_list, day_index, ts, hmm))
        return pred_open


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


def plotGraph(actual_arr, pred_arr, title):
    fig, axes = plt.subplots(figsize=(10, 4))
    axes.plot(pred_arr, color = 'm', label = 'Pred Open')
    axes.plot(actual_arr, color = 'g', label = 'Actual Open')
    axes.grid(True)
    axes.set_xlabel('Time')
    axes.set_ylabel('Open')
    axes.legend(loc='best')

    axes.set_title(title+' Open vs Time') 
    fig.savefig("../output_data/"+title)



# In[24]:


def testModel(hidden_states, steps, timestep):
    hmm = makeHMM_model(hidden_states)
    all_combinations = findAllCombinations(1,steps)
    print hmm
    pred = predictAllOpen(all_combinations, timestep, hmm)
#     print pred
    title = "("+str(hidden_states) + "," + str(timestep) + ")"
    plotGraph(validate_y, pred, title)


# In[25]:


testModel(4, 100, 20 )


# In[26]:


testModel(4, 100, 50 )


# In[ ]:


testModel(4, 100, 75 )


# In[ ]:


testModel(8, 100, 20 )


# In[ ]:


testModel(8, 100, 50 )


# In[ ]:


testModel(8, 100, 75 )


# In[ ]:


testModel(12, 100, 20 )


# In[ ]:


testModel(12, 100, 50 )


# In[ ]:


testModel(12, 100, 75 )

