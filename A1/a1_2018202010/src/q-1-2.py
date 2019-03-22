#!/usr/bin/env python
# coding: utf-8

# # Part 2: Train the decision tree with categorical and numerical features. Report precision, recall, f1 score and accuracy.

# In[1]:


import pandas as pd 
import numpy as np

import sys #for maxint, minint

class Node(object):
    def __init__(self, root, pos, neg, tf, split):
        self.root = root
        self.pos = pos #no of yes
        self.neg = neg #no of no
        self.children = {} #subtrees
        self.isLeaf = tf #whether a leaf not or not
        self.split = split #where to split for numerical
        if(pos>neg):
            self.result=1 #decision here
        else:
            self.result=0
    def add_child(self, key, val):
        self.children[key]=val

def countYesNo(dataframe): #counts no of positives and negatives at a node
    left_col = dataframe['left']
    pos = dataframe[left_col == 1] 
    
    #no of rows of yes
    yes_r = pos.shape[0]

    neg = dataframe[left_col == 0]
    
    #no of rows of no
    no_r = neg.shape[0]
    
    return yes_r, no_r

#build Tree using entropy
#param: dataframe, numerical, categorical lists
#return: root of the tree- tree_root
def buildTree(dataframe, num, categ): 
    yes, no = countYesNo(dataframe)
    
    if no==0: #node is yes->1
        return Node(1,yes,0,True,-1)
        
    elif yes==0: #node is no->0
        return Node(0,0,no,True,-1)
    
    root_node,split = findMaxInfoGain(dataframe, num, categ) #work_acc
    
    tree_root = Node(root_node, yes, no, False,split)
    
    
    root_col = dataframe[root_node] #work_acc col
    
    if root_node in num:
        #numerical attribute
        grea_than = dataframe[dataframe[root_node] > split]
        less_than = dataframe[dataframe[root_node] <= split]
        
        less_than_tree = buildTree(less_than, num, categ)
        grea_than_tree = buildTree(grea_than, num, categ)
        
        lname = "less_than_"+(str(split))
        gname = "greater_than"+str(split)
        tree_root.add_child(lname,less_than_tree)
        tree_root.add_child(gname,grea_than_tree)
        
    
    elif root_node in categ:
        s = pd.Series(root_col) 
        unique_val = s.unique() 
        for i in unique_val:
            array = dataframe[root_col == i]

            #dataframe for current unique value
            curr = pd.DataFrame(array)

            #now drop this col
            curr = curr.drop(root_node , 1)

            recursive_root = buildTree(curr, num, categ)

            tree_root.add_child(i,recursive_root)
    
    return tree_root


# In[2]:


#Step-1: Compute impurity score of training label distribution
#entropy of entire dataset, params: dataframe, target- target col
#returns entropy 
def entropyCalculate(dataframe, target):
    total_len=len(dataframe)
    col=dataframe[target]
    s = pd.Series(col) 
    count_arr=s.value_counts() #has no of 0s and 1s
    entropy=0
    for counts in count_arr:
        prob = float(counts)/total_len
        entropy += -1*prob* np.log2(prob)
    
    return entropy

#Step-2: Compute impurity score for each unique value of candidate attributes
#Step-3: Compute impurity score for candidate attribute 
#entropy for each attribute, params: dataframe, coloumnName, target
#returns entropy for sub attributes
def entropyAttribute(dataframe, col_name, label):
    col=dataframe[col_name] 
    
    #total rows in this col, eg sales -8990
    col_len=len(col) 
    
    #convert the column to series
    s = pd.Series(col)
    
    #find all unique attr, eg low,med,high
    unique_val = s.unique() 

    total_entropy=0
    
    for i in unique_val:
        #all rows in the col where sub attribute is i, eg for sales , i=accounting..
        ar = dataframe[col==i]
        
        #find no. of rows for that subattribute
        total_r = ar.shape[0]
        
        #made a dataframe for this attribute
        cur_df = pd.DataFrame(ar) 
        
        #now suppose attribute=sales, value=accounting, find entropy now for accounting
        curr_entropy = entropyCalculate(cur_df, label)
        
        fraction = float(total_r)/col_len
        total_entropy += fraction * curr_entropy
        
    return total_entropy    


# In[3]:


#finds split for numerical column, calculates entire entropy and finds the minimum one
#param: dataframe, numerical col
#return: minimum entropy and corresponding split point
def findSplit(dataframe, col):
    label = 'left'
    min = sys.maxint
    idx=0
    for j in pd.Series(dataframe[col]).unique():
        less_than = dataframe[dataframe[col] > j]
        grea_than = dataframe[dataframe[col] <= j]

        less_rows = less_than.shape[0]
        grea_rows = grea_than.shape[0]
        tot = less_rows + grea_rows
        e1 = entropyCalculate(less_than, label)
        e2 = entropyCalculate(grea_than, label)

        entropy = ((e1*less_rows)/tot + (e2*grea_rows)/tot)
        if(min>entropy):
            min = entropy
            idx = j
    return min, idx

#a helper traverse function to visualize the tree
def traverse(root):
    if len(root.children)==0:
        print "return root: ",root.root
        return
    
    print "Root: ",root.root
    
    for k,v in root.children.items():
        print "root: ",root.root, "key: ",k
        traverse(v)
# traverse(root)


# In[4]:


#prediction function
def predict2(row,root,num,categ, senior,default=0):
    
    if(root.isLeaf == True):
        if root.result == 1:
            year = row["time_spend_company"]
        
            senior[year] += 1
        return root.result
   
    col=root.root
    split_at = root.split
    val=row[col]
    less_key = ''
    grea_key = ''
    if col in num:
        for k,v in root.children.items():
            if k[0]=='l':
                less_key = k
            else:
                grea_key = k
            
        if val > split_at:
            return predict2(row, root.children[grea_key], num,categ,senior)
        else:
            return predict2(row, root.children[less_key], num,categ,senior)

    elif col in categ:
        if val in root.children.keys():
            return predict2(row,root.children[val],num,categ,senior)
        else:
            return root.result


# In[5]:


def helper(root, predict_col, df_sample,senior):
    df_sample[predict_col] = df_sample.apply(predict2, axis=1, args=(root,numerical,categorical,senior,0))
    return df_sample[predict_col]
    
#pred_label = predict(model,model_args,X) 
#where model = decision tree object, model_args = parameters to be passed, X = test sample.


# In[6]:


def predict(model,model_args,X):
    df_sample = pd.read_csv(X)
    senior =model_args[1]
    left_col = helper(model,model_args[0], df_sample,senior)
    return left_col


# In[7]:


#a helper function for making predictions, adds a new col of name predict_col to store the prediction    
def helper_validate(df,root,senior, predict_col):
    df[predict_col] = df.apply(predict2, axis=1, args=(root,numerical,categorical,senior,0))
    a,p,f,r = findMeasures(df, predict_col)
    


# In[8]:


#finds measures for tp, fp, tn, fn , accuracy,precision, recall
def findMeasures(df, predict_col):
    truePos=0
    trueNeg=0
    falsePos=0
    falseNeg=0
    
    for index, row in df.iterrows():
        
        if row[predict_col]==0 and row["left"]==0:
            trueNeg += 1
            
            
        elif row[predict_col]==0 and row["left"]==1:
    
            falseNeg += 1
    
        elif row[predict_col]==1 and row["left"]==1:
           
            truePos += 1
           
        
        elif row[predict_col]==1 and row["left"]==0:
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
    print "A, P, R, F: ",accuracy*100, precision*100, recall*100, f1_score
    return accuracy*100, precision*100, recall*100, f1_score


# In[9]:


#Step-4: Compute Information Gain (reduction in impurity score) provided by candidate attribute
#Step-5: Compare Information Gain provided by all candidates
#params: dataframe, numerical, categorical lists
#returns root and splitPoint (if numerical)

def findMaxInfoGain(dataframe, num, categ):
    unique_cols = dataframe.columns.tolist()
    label = 'left'
    
    entropy = entropyCalculate(dataframe, label)
    max = -sys.maxint - 1
    temp = 0
    attr_entropy=0
    split=0
    for i in unique_cols:
        if i != label:            
            if i in num:
                #this is a numerical attribute
                attr_entropy, idx = findSplit(dataframe, i)
                
            elif i in categ:
                attr_entropy = entropyAttribute(dataframe, i, label)
            
            temp = entropy - attr_entropy
            
            if temp>max:
                max=temp
                root=i
                if i in num:
                    split = idx
                else:
                    split = -1
    return root, split        


# In[10]:


def initSenior(df):
    col = df['time_spend_company']
    senior = {}
    uni = col.unique()
#     print uni
    for i in uni:
#         print type(i)
        senior[i]=0
    return senior


# In[11]:


#filename=raw_input('Enter filename: ')
filename = "../input_data/train.csv"

dataset = pd.read_csv(filename)
dataset = dataset.sample(frac=1)
train, validate = np.split(dataset, [int(.8*len(dataset))])

numerical = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']
categorical = ['Work_accident','promotion_last_5years','sales','salary','left']
root = buildTree(train, numerical, categorical)



# In[12]:


testFile = "../input_data/sample_test.csv"
senior ={}
senior = initSenior(dataset)
# print "senior: ", senior

model_args=['left', senior]

helper_validate(validate, root, senior,'prediction_ent')
print senior


# In[13]:


pred_label = predict(root, model_args, testFile )
print "Entropy:"
print pred_label

