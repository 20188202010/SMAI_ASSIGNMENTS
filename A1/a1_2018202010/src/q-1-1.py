#!/usr/bin/env python
# coding: utf-8

# # Part-1 Train decision tree only on categorical data.  Report precision,recall, f1 score and accuracy

# In[2]:


import pandas as pd
import numpy as np
import sys #for maxmin


# In[3]:


class Node(object):
    def __init__(self, root, pos, neg, tf):
        self.root = root
        self.pos = pos
        self.neg = neg
        self.children = {}
        self.isLeaf = tf
        if(pos>neg):
            self.result=1 #decision here
        else:
            self.result=0
    def add_child(self, key, val):
        self.children[key]=val


# In[4]:


def countYesNo(dataframe):
#     left_col = dataframe['play']
    left_col = dataframe['left']
    pos = dataframe[left_col == 1] 
    
    #no of rows of yes
    yes_r, yes_c = pos.shape

    neg = dataframe[left_col == 0]
    
    #no of rows of no
    no_r, no_c = neg.shape
    
    return yes_r, no_r


# In[5]:


def entropyCalculate(dataframe, col_name):
    total_len=len(dataframe)
    col=dataframe[col_name]
    s = pd.Series(col)
    count_arr=s.value_counts()
#     print count_arr[0], count_arr[1]
    entropy=0
    for counts in count_arr:
#         print counts
        prob = float(counts)/total_len
#         print prob
        entropy += -1*prob* np.log2(prob)
        
#     print entropy
    return entropy


# In[6]:


def entropyAttribute(dataframe, col_name, label):
    col=dataframe[col_name] 
    
    #total rows in this col, eg sales -8990
    col_len=len(col) 
#     print "colLen: ",col_len
    
    #convert the column to series
    s = pd.Series(col)
    
    #find all unique attr, eg low,med,high
    unique_val = s.unique() 
#     print "unique val: ",unique_val

    total_entropy=0
    
    for i in unique_val:
        #all rows in the col where sub attribute is i, eg for sales , i=accounting..
        ar = dataframe[col==i]
        
        #find no. of rows for that subattribute
        total_r, total_c = ar.shape
#         print "unique val, total r: ", i,total_r
        
        #made a dataframe for this attribute
        cur_df = pd.DataFrame(ar) 
        
        #now suppose attribute=sales, value=accounting, find entropy now for accounting
        curr_entropy = entropyCalculate(cur_df, label)
        
#         print "current entropy: ", curr_entropy
        fraction = float(total_r)/col_len
        total_entropy += fraction * curr_entropy
        
#         print "total entropy: ", total_entropy
    return total_entropy    


# In[7]:


def findMaxInfoGain(dataframe):
    unique_cols = dataframe.columns.tolist()
#     print "unique cols: ",unique_cols
#     label = 'play'
    label = 'left'
    
    entropy = entropyCalculate(dataframe, label)
#     print "entropy: ",entropy
    max = -sys.maxint - 1
    root=''
    for i in unique_cols:
#         print "col: ",i,
        if i != label:            
            attr_entropy = entropyAttribute(dataframe, i, label)
            
#             print "attr entropy: ", attr_entropy
            temp = entropy - attr_entropy
#             print "entropy: ",temp
            if temp>max:
                max=temp
                root=i
#     print "root: ",root
    return root, max        


# In[8]:


def buildTree(dataframe):
    yes, no = countYesNo(dataframe)
    
#     percent = (yes/no)*100
    
    if no==0: #node is yes->1
        return Node(1,yes,0,True)
        
    elif yes==0: #node is no->0
        return Node(0,0,no,True)
    
    if len(dataframe.keys())==1: 
        if yes<no:
            return Node(0,yes,no,True)
        else:
            return Node(1,yes,no,True)
    
    root_node, gain = findMaxInfoGain(dataframe) #work_acc
    
    tree_root = Node(root_node, yes, no, False)    
    root_col = dataframe[root_node] #work_acc col
    
    s = pd.Series(root_col) 
    unique_val = s.unique() #0,1
    
    for i in unique_val:
        array = dataframe[root_col == i] 
        
        #dataframe for current unique value
        curr = pd.DataFrame(array)
        
        #now drop this col
        curr = curr.drop(root_node , 1)
        
        recursive_root = buildTree(curr)
        
        tree_root.add_child(i,recursive_root)
    
    return tree_root


# In[9]:


def traverse(root):
    if len(root.children)==0:
#         print "return root: ",root.root
        return
    
#     print "Root: ",root.root
    
    for k,v in root.children.items():
#         print "root: ",root.root, "key: ",k
        traverse(v)


# In[10]:


def predict2(row,root,default=0):
    
    if(root.isLeaf == True):
#         print root.root
        return root.result
    col=root.root

    val=row[col]
#     print "val: ",val
    if val in root.children.keys():
        return predict2(row,root.children[val])
    else:
#         val=0
        return root.result
    
def helper(root, predict_col, df_sample):
    df_sample[predict_col] = df_sample.apply(predict2, axis=1, args=(root,0))
    return df_sample[predict_col]

def predict(model,model_args,X):
    df_sample = pd.read_csv(X)
    left_col = helper(model, model_args, df_sample)
    return left_col


# In[11]:


#a helper function for making predictions, adds a new col of name predict_col to store the prediction    
def helper_validate(df,root, predict_col):
    df[predict_col] = df.apply(predict2, axis=1, args=(root,0))
    a,p,f,r = findMeasures(df, predict_col)


# In[12]:


def findMeasures(df, predict_col):
    truePos=0
    trueNeg=0
    falsePos=0
    falseNeg=0
    
    for index, row in df.iterrows():
        
#         print "index, predict, left,",index, row["prediction"], row["left"]
        if row[predict_col]==0 and row["left"]==0:
            trueNeg += 1
            
        elif row[predict_col]==0 and row["left"]==1:
            falseNeg += 1
        
        elif row[predict_col]==1 and row["left"]==1:
            truePos += 1
        elif row[predict_col]==1 and row["left"]==0:
            falsePos += 1
#     print "TP, TN, FP, FN: ", truePos, trueNeg, falsePos, falseNeg
    sumtotal = truePos + trueNeg + falsePos + falseNeg
    accuracy = ((float)(truePos + trueNeg))/sumtotal
    try:
        precision = ((float)(truePos))/(truePos + falsePos)
    except:
        precision = 0
    try:
        recall = ((float)(truePos))/(truePos + falseNeg)
    except:
        recall = 0
    try:
        f1_score_den = 1.0/recall + 1.0/precision
        f1_score = 2.0/f1_score_den
    except:
        f1_score=0
    print "A, P, R, F: ", accuracy*100, precision*100, recall*100, f1_score
    return accuracy*100, precision*100, recall*100, f1_score
        


# In[13]:


filename = "../input_data/train.csv"

dataset = pd.read_csv(filename)
dataset = dataset.sample(frac=1)
train, validate = np.split(dataset, [int(.8*len(dataset))])
# print train.shape, validate.shape

df_train = pd.DataFrame(train,columns=['Work_accident','promotion_last_5years','sales','salary','left'])


# In[14]:


root = buildTree(df_train)


# In[15]:


testFile = "../input_data/sample_test.csv"
pred_label = predict(root, 'left', testFile )
print "Entropy Categorical: "
print pred_label
helper_validate(train, root, 'prediction_ent')

