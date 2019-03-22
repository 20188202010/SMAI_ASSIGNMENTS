#!/usr/bin/env python
# coding: utf-8

# # Part 4:  Visualise training data on a 2-dimensional plot taking one fea-ture (attribute) on one axis and other feature on another axis.  Take two suitable features to visualise decision tree boundary (Hint:  use scatter plot with different colors for each label).

# In[1]:


import pandas as pd 
import numpy as np

#for plots
import matplotlib
import matplotlib.pyplot as plt
from pylab import *


# In[2]:


dataset = pd.read_csv("../input_data/train.csv")
dataset = dataset.sample(frac=1)
train, validate = np.split(dataset, [int(.8*len(dataset))])

numerical = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']
categorical = ['Work_accident','promotion_last_5years','sales','salary','left']


# In[3]:


attr_1 = 'last_evaluation'
attr_2 = 'satisfaction_level'

left_df = train['left']
df1 = train[attr_1]
df2 = train[attr_2]

left_0 = train[train['left']==0]
left_1 = train[train['left']==1]

sl0 = left_0[attr_1]
sl1 = left_1[attr_1]
print attr_1, " left = 0", len(sl0), " left = 1", len(sl1)

amh0 = left_0[attr_2]
amh1 = left_1[attr_2]
print attr_2, " left = 0", len(amh0), " left = 1", len(amh1)


fig, axes = plt.subplots(figsize=(7, 7))

axes.scatter(sl0,amh0, label=r"$left=0$")
axes.scatter(sl1,amh1, label=r"$left=1$")


legend = axes.legend(loc='best')
axes.set_title('Comparison')


plt.xlabel(attr_1)
plt.ylabel(attr_2)

fig.savefig("q4.png")
fig2, axes2 = plt.subplots(1, 2, figsize=(10,10))

i=0
yticks =[]
for y in range (0,10):
    i = i+0.1
    yticks.append(i)

axes2[0].scatter(sl0,amh0, label=r"$left=0$")
axes2[1].scatter(sl1,amh1,label=r"$left=1$",color="orange")

legend = axes2[0].legend(loc='best')
axes2[0].set_xlabel(attr_1)
axes2[0].set_ylabel(attr_2)
axes2[0].set_title("left=0")

legend2 = axes2[1].legend(loc='best')
axes2[1].set_xlabel(attr_1)
axes2[1].set_ylabel(attr_2)
axes2[1].set_title("left=1")

axes2[1].set_yticks(yticks)
axes2[1].set_xticks(yticks)

axes2[0].set_xticks(yticks)
axes2[0].set_yticks(yticks)
fig2.savefig("q4_2.png")

