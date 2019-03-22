#!/usr/bin/env python
# coding: utf-8

# ## Implement Naive Bayes classifier on Loan dataset to help bank achieve its goal.

# In[36]:


import numpy as np
import pandas as pd
import math


# In[37]:


dataset = pd.read_csv("LoanDataset/data.csv", header = None)


# Attributes:
# 
# ID - A unique identifier <br/>
# Age <br/>
# Number of years of experience <br/>
# Annual Income <br/>
# ZIPCode <br/>
# Family size <br/>
# Avgerage spending per month <br/>
# Education Level. 1: 12th; 2: Graduate; 3: Post Graduate <br/>
# Mortgage Value of house if any <br/>
# Did this customer accept the personal loan offered in the last campaign? -- Output label <br/>
# Does the customer have a securities account with the bank? <br/>
# Does the customer have a certificate of deposit (CD) account with the bank? <br/>
# Does the customer use internet banking facilities? <br/>
# Does the customer uses a credit card issued by UniversalBank? <br/>

# In[38]:


dataset.columns = ['id', 'age', 'exp', 'income', 'zip', 'fam_size', 'spending', 'education', 'mortgage', 'loan_accept', 'securities_account', 'certi_dep', 'net_banking', 'UniversalBank_cc']
numerical = ['age', 'exp', 'income', 'fam_size', 'spending', 'mortgage']
categorical = ['education', 'securities_account', 'certi_dep', 'net_banking', 'UniversalBank_cc']
dataset = dataset.drop(dataset.index[0])
# rows , cols = dataset.shape
target = 'loan_accept'


# In[39]:


dataset = dataset.sample(frac=1)


# In[40]:


train, validate = np.split(dataset, [int(.8*len(dataset))])


# In[41]:


target0 = train[train[target]==0]
target1 = train[train[target]==1]

total_len = train.shape[0]

total_zero = target0.shape[0]
total_one = target1.shape[0]

p0 = float(total_zero)/total_len
p1 = float(total_one)/total_len

prob_ca = {}
prob_nu = {}


# In[42]:


def numCalc():
    #{ age: [ [mean0, std0], [mean1,std1] ]}
    for colName in numerical:
        zeros = [] #list to store mean, std for 0 class
        ones = [] #list to store mean, std for 1 class
        summary = [] #combined list of 0 and 1 class mean, std
        zeroClass = target0[colName] #eg temp = 66, play = no
        oneClass = target1[colName] #eg temp = 66, play = yes

        mean0 = zeroClass.mean()
        mean1 = oneClass.mean()
        ones.append(mean1) #mean of 1 class
        zeros.append(mean0)

        std0 = zeroClass.std()
        std1 = oneClass.std()
        ones.append(std1)
        zeros.append(std0)

        # summary: [ <zeros>, <ones>]

        summary.append(zeros)
        summary.append(ones)
        
        prob_nu[colName] = summary


# In[43]:


numCalc()
# print prob_nu


# In[44]:


def catCalc():
    #{ <humidity>: [ {<low>: [p0, p1], <medium>: [p0, p1], <high>: [p0, p1]} ] }
    for colName in categorical:
        summary = [] #combined list of 0 and 1 class prob for all unique values of that column
        unique_val = train[colName].unique()
        sub_dict = {}
        for val in unique_val:
            
            sub_summary = [] #list of 0 and 1 class prob for that particular unique val
            val_df = train[train[colName] == val] #eg outlook = sunny

            zeroClass = val_df[val_df[target] == 0] #sunny , play = no
            oneClass = val_df[val_df[target] == 1] #sunny , play = yes

            num_zero = zeroClass.shape[0]
            num_one = oneClass.shape[0]

            prob_zero = float(num_zero)/total_zero
            prob_one = float(num_one)/total_one
            sub_summary.append(prob_zero)
            sub_summary.append(prob_one)
#             print "colName: ",colName,"val: ",val,"subSum: ",sub_summary
            sub_dict[val] = sub_summary
#             print "sub_dict: ",sub_dict
        summary.append(sub_dict)
#         print "summary: ",summary
        prob_ca[colName] = summary
            
      


# In[45]:


catCalc()
# print prob_ca


# In[46]:


def numProb(sd,x,mean):
    constant = math.sqrt(2*3.14)
    den = constant*sd
    
    num = (x - mean)**2
    power =  -1 * float (num) / ( 2 * (sd**2) )
    
    ans = float( math.exp(power))/ den
    return ans
    


# In[47]:


def predict(row, default=0):
    likelihood_1 = 0
    likelihood_0 = 0
    
    for colName, colVal in row.items():

        lognP0 =0
        lognP1 =0
        logcP0 =0
        logcP1 =0
        
        if colName in numerical:  #{ age: [ [mean0, std0], [mean1,std1] ]}
            mean0 = prob_nu[colName][0][0]
            mean1 = prob_nu[colName][1][0]
            std0 = prob_nu[colName][0][1]
            std1 = prob_nu[colName][1][1]

            pr0 = numProb(std0, colVal, mean0)
            lognP0 = math.log(pr0, 10)
            
            pr1 = numProb(std1, colVal, mean1)
            lognP1 = math.log(pr1, 10)

        elif colName in categorical: #{ <humidity>: [ {<low>: [p0, p1], <medium>: [p0, p1], <high>: [p0, p1]} ] }
            pr_c0 = prob_ca[colName][0][colVal][0]
            pr_c1 = prob_ca[colName][0][colVal][1]
        
            logcP0 = math.log(pr_c0,10)
            
            logcP1 = math.log(pr_c1,10)

        likelihood_1 = likelihood_1 + lognP1 + logcP1
        likelihood_0 = likelihood_0 + lognP0 + logcP0
    

    if likelihood_1 + math.log(p1,10) >= likelihood_0 + math.log(p0,10):
        label = 1
    else:
        label = 0
    
    return label


# In[48]:


def helper(df, predict_col):
    df[predict_col] = df.apply(predict, axis=1, args=(0))    
    
    return df[predict_col]
helper(validate,'prediction')
# print validate['validate']


# In[49]:


#finds measures for tp, fp, tn, fn , accuracy,precision, recall
def findMeasures(df, predict_col):
    truePos=0
    trueNeg=0
    falsePos=0
    falseNeg=0
    
#     for index, row in validate.iterrows():
    for index, row in df.iterrows():
    
        
        if row[predict_col]==0 and row[target]==0:
            trueNeg += 1
            
            
        elif row[predict_col]==0 and row[target]==1:
    
            falseNeg += 1
    
        elif row[predict_col]==1 and row[target]==1:
           
            truePos += 1
           
        
        elif row[predict_col]==1 and row[target]==0:
            falsePos += 1

    sumtotal = truePos + trueNeg + falsePos + falseNeg
    accuracy = ((float)(truePos + trueNeg))/sumtotal
    precision = ((float)(truePos))/(truePos + falsePos)
    recall = ((float)(truePos))/(truePos + falseNeg)
    try:
        f1_score_den = 1.0/recall + 1.0/precision
        f1_score = 2.0/f1_score_den
    except:
        f1_score=0
    print "TP, TN, FP, FN: ", truePos, trueNeg, falsePos, falseNeg
    print "A, P, R, F: ",accuracy*100, precision*100, recall*100, f1_score
    return accuracy*100, precision*100, recall*100, f1_score


# ### Observations

# In[50]:


findMeasures(validate, 'prediction')


# ### Testing

# In[51]:


filename = raw_input("Enter file for testing: ")
test = pd.read_csv(filename, header = None)
test.columns = ['id', 'age', 'exp', 'income', 'zip', 'fam_size', 'spending', 'education', 'mortgage', 'securities_account', 'certi_dep', 'net_banking', 'UniversalBank_cc']
numerical = ['age', 'exp', 'income', 'fam_size', 'spending', 'mortgage']
categorical = ['education', 'securities_account', 'certi_dep', 'net_banking', 'UniversalBank_cc']
test = test.drop(test.index[0])
helper(test,'label')

