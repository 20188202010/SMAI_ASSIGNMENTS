#!/usr/bin/env python
# coding: utf-8

# # Part-3: Contrast  the effectiveness of Misclassification  rate,  Gini,  Entropy as impurity measures in terms of precision, recall and accuracy

# In[1]:


import pandas as pd
import numpy as np
import sys #max,min int


# In[2]:


class Node(object):
    def __init__(self, root, pos, neg, tf, split):
        self.root = root
        self.pos = pos #no of yes
        self.neg = neg #no of no
        self.children = {} #subtrees
        self.isLeaf = tf #whether a leaf not or not
        self.split = split #where to split for numerical
        if(pos>neg):
            self.result=1
        else:
            self.result=0
    def add_child(self, key, val):
        self.children[key]=val


# In[3]:


def countYesNo(dataframe):
    left_col = dataframe['left']
    pos = dataframe[left_col == 1] 
    
    #no of rows of yes
    yes_r = pos.shape[0]
    neg = dataframe[left_col == 0]
    #no of rows of no
    no_r = neg.shape[0]
    
    return yes_r, no_r


# ## GINI INDEX

# In[4]:


def giniEntire(dataframe, col_name): #gini whole dataset
    total_len=len(dataframe)
    col=dataframe[col_name]
    
    left0 = dataframe[col == 0]
    count0 = left0.shape[0]
    
    left1 = dataframe[col == 1]
    count1 = left1.shape[0]
    
    try:
        prob_y = float(count0)/total_len
    except:
        prob_y = 0
    
    try:
        prob_n = float(count1)/total_len
    except:
        prob_n = 0
    
    gini = 2 * prob_y * prob_n
    return gini


# In[5]:


def giniNumerical(df, col_name, split):
    col = df[col_name]
    col_len = len(col)

    less_than = df[col <= split]
    total_less = less_than.shape[0]

    grea_than = df[col > split]
    total_greater = grea_than.shape[0]

    yesL, noL = countYesNo(less_than)

    yesG, noG = countYesNo(grea_than)

    prob_y_l = float(yesL)/total_less
    prob_n_l = float(noL)/total_less
    
    gini_l = 2*prob_y_l*prob_n_l

    prob_y_g = float(yesG)/total_greater
    prob_n_g = float(noG)/total_greater

    gini_g = 2*prob_y_g*prob_n_g
    
    weight1 = ( float(total_less)/col_len ) * gini_l
    weight2 = ( float(total_greater)/col_len ) * gini_g
        
    gini_var = weight1 + weight2

    return gini_var


# In[6]:


def giniCategorical(df, col_name):
    col = df[col_name]
    col_len = len(col)
    
    unique = col.unique()
    
    gini_var = 0
    gini_i = 0
    for i in unique:
        temp_df = df[col == i]
        #suppose we get low, count yes and no for low
        total_temp = len(temp_df)

        yes_i, no_i = countYesNo(temp_df)
        
        prob_y_i = float(yes_i)/total_temp
        prob_n_i = float(no_i)/total_temp

        gini_i = 2 * prob_y_i * prob_n_i
        
        wt1 = ( float(total_temp)/col_len ) * gini_i
        
        gini_var += wt1
    return gini_var


# In[7]:


def findGiniSplit(dataframe, col): #finds the split for numerical attribute using minimum gini
    label = 'left'
    min = sys.maxint
    idx=0
    for j in pd.Series(dataframe[col]).unique():
        less_than = dataframe[dataframe[col] > j]
        grea_than = dataframe[dataframe[col] <= j]

        less_rows = less_than.shape[0]
        grea_rows = grea_than.shape[0]
        tot = less_rows + grea_rows
        e1 = giniEntire(less_than, label)
        e2 = giniEntire(grea_than, label)

        entropy = ( float(e1*less_rows)/tot ) + ( float(e2*grea_rows)/tot )
        if(min>entropy):
            min = entropy
            idx = j
    return min, idx


# In[8]:


def findGiniRoot(dataframe, num, categ):
    unique_cols = dataframe.columns.tolist()
    label = 'left'
    
    entire_gini = giniEntire(dataframe, label)
    max = -sys.maxint - 1
    temp = 0
    gini_index=0
    split=0
    for i in unique_cols:
        if i != label:            
            if i in num:
                gini_index, idx = findGiniSplit(dataframe, i)
                
            elif i in categ:
                gini_index = giniCategorical(dataframe, i)
            
            temp = entire_gini - gini_index
            
            if temp>max:
                max=temp
                root=i
                if i in num:
                    split = idx
                else:
                    split = -1
    return root, split        


# In[9]:


def buildTreeGini(dataframe, num, categ):
    yes, no = countYesNo(dataframe)
    
    if no==0: #node is yes->1
        return Node(1,yes,0,True,-1)
        
    elif yes==0: #node is no->0
        return Node(0,0,no,True,-1)
    
    root_node,split = findGiniRoot(dataframe, num, categ) #work_acc
    
    tree_root = Node(root_node, yes, no, False,split)
        
    root_col = dataframe[root_node] #work_acc col
    
    if root_node in num:

        grea_than = dataframe[dataframe[root_node] > split]
        less_than = dataframe[dataframe[root_node] <= split]
        
        less_than_tree = buildTreeGini(less_than, num, categ)
        grea_than_tree = buildTreeGini(grea_than, num, categ)
        
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

            recursive_root = buildTreeGini(curr, num, categ)

            tree_root.add_child(i,recursive_root)
    
    return tree_root


# In[10]:


def misClEntire(dataframe, col_name): #miscl whole dataset
    total_len=len(dataframe)
    col=dataframe[col_name]
    
    left0 = dataframe[col == 0]
    count0 = left0.shape[0]
    
    left1 = dataframe[col == 1]
    count1 = left1.shape[0]
    
    try:
        prob_y = float(count0)/total_len
    except:
        prob_y = 0
    
    try:
        prob_n = float(count1)/total_len
    except:
        prob_n = 0
    
    mis_cl = min(prob_y, prob_n)
    return mis_cl


# In[11]:


def misClNumerical(df, col_name, split):
    col = df[col_name]
    col_len = len(col)
    less_than = df[col <= split]
    total_less = less_than.shape[0]
    grea_than = df[col > split]
    total_greater = grea_than.shape[0]

    yesL, noL = countYesNo(less_than)
    
    yesG, noG = countYesNo(grea_than)
    
    prob_y_l = float(yesL)/total_less
    prob_n_l = float(noL)/total_less
    mis_l = min(prob_y_l,prob_n_l)

    prob_y_g = float(yesG)/total_greater
    prob_n_g = float(noG)/total_greater
    mis_g = min(prob_y_g,prob_n_g)

    w1 = ( float(total_less)/col_len ) * mis_l
    w2 = ( float(total_greater)/col_len ) * mis_g
    
    mis_var = w1 + w2
    return mis_var


# In[12]:


def misClCategorical(df, col_name):
    col = df[col_name]
    col_len = len(col)
    unique = col.unique()
    mis_var = 0
    mis_i = 0
    for i in unique:
        temp_df = df[col == i]
        #suppose we get low, count yes and no for low
        total_temp = len(temp_df)
        yes_i, no_i = countYesNo(temp_df)
        prob_y_i = float(yes_i)/total_temp
        prob_n_i = float(no_i)/total_temp
    
        mis_i = min( prob_y_i , prob_n_i)
        mis_var += (float(total_temp)/col_len) * mis_i
    return mis_var


# In[13]:


def findMisClSplit(dataframe, col): #finds the split for numerical attribute using minimum gini
    label = 'left'
    tot = len(dataframe[col])
    min=sys.maxint
    idx=0
    for j in pd.Series(dataframe[col]).unique():
        less_than = dataframe[dataframe[col] > j]
        grea_than = dataframe[dataframe[col] <= j]

        less_rows = less_than.shape[0]
        grea_rows = grea_than.shape[0]
        
        e1 = misClEntire(less_than, label)
        e2 = misClEntire(grea_than, label)

        entropy =  float(e1*less_rows)/tot 
        entropy += float (e2*grea_rows)/tot
        if(min>entropy):
            min = entropy
            idx = j
    return min, idx


# In[14]:


def findMisClRoot(dataframe, num, categ):
    unique_cols = dataframe.columns.tolist()
    label = 'left'
    
    entire_misCl = misClEntire(dataframe, label)
    max = -sys.maxint - 1
    temp = 0
    misCl=0
    split=0
    for i in unique_cols:
        if i != label:            
            if i in num:
                #this is a numerical attribute
                misCl, idx = findMisClSplit(dataframe, i)
                
            elif i in categ:
                misCl = misClCategorical(dataframe, i)
            
            temp = entire_misCl - misCl
            
            if temp>max:
                max=temp
                root=i
                if i in num:
                    split = idx
                else:
                    split = -1

    return root, split,max        


# In[15]:


def buildTreeMisCl(dataframe, num, categ):
    yes, no = countYesNo(dataframe)
    
    root_node,split,mis_cl = findMisClRoot(dataframe, num, categ) #work_acc
    
    if mis_cl == 0:
        if yes>no:
            return Node(1,yes,no,True,-1)
        else:
            return Node(0,yes,no,True,-1)
    
    tree_root = Node(root_node, yes, no, False,split)
        
    root_col = dataframe[root_node] #work_acc col
    
    if root_node in num:
        #numerical attribute
        grea_than = dataframe[dataframe[root_node] > split]
        less_than = dataframe[dataframe[root_node] <= split]
        
        less_than_tree = buildTreeMisCl(less_than, num, categ)
        grea_than_tree = buildTreeMisCl(grea_than, num, categ)
        
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

            recursive_root = buildTreeMisCl(curr, num, categ)

            tree_root.add_child(i,recursive_root)
    
    return tree_root


# In[16]:


def traverse(root):
    if len(root.children)==0:
        print "return root: ",root.root
        return
    
    print "Root: ",root.root
    
    for k,v in root.children.items():
        print "root: ",root.root, "key: ",k
        traverse(v)
# traverse(root)


# In[17]:


def predict2(row,root,num,categ,default=0):
    
    if(root.isLeaf == True):
        return root.root
    
    
    col=root.root
    split_at = root.split
    val=row[col]
    less_key = ''
    grea_key = ''
    if col in num:
#         print "numerical attr"
        for k,v in root.children.items():
            if k[0]=='l': #less than
                less_key = k
            else:
                grea_key = k
            
        if val > split_at:
            return predict2(row, root.children[grea_key], num,categ)
        else:
            return predict2(row, root.children[less_key], num,categ)

    elif col in categ:
        if val in root.children.keys():
            return predict2(row,root.children[val],num,categ)
        else:
            return root.result

        
def helper(root, predict_col, df_sample):
    df_sample[predict_col] = df_sample.apply(predict2, axis=1, args=(root,numerical,categorical,0))
    return df_sample[predict_col]
    
#pred_label = predict(model,model_args,X) 
#where model = decision tree object, model_args = parameters to be passed, X = test sample.

def predict(model,model_args,X):
    df_sample = pd.read_csv(X)
    left_col = helper( model, model_args,df_sample)
    return left_col


# In[18]:


def helper_validate(df,root, predict_col):
    df[predict_col] = df.apply(predict2, axis=1, args=(root,numerical,categorical,0))
    a,p,f,r = findMeasures(df, predict_col)


# In[19]:


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


# In[20]:


#filename=raw_input('Enter filename: ')
filename = "../input_data/train.csv"

dataset = pd.read_csv(filename)
dataset = dataset.sample(frac=1)
train, validate = np.split(dataset, [int(.8*len(dataset))])

numerical = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']
categorical = ['Work_accident','promotion_last_5years','sales','salary','left']


# In[21]:


root_gini = buildTreeGini(train, numerical, categorical)


# In[22]:


testFile = "../input_data/sample_test.csv"
pred_label = predict(root_gini, 'left', testFile )
print "Gini : "
print pred_label
helper_validate(validate, root_gini, 'prediction_gini')


# In[23]:


root_misCl = buildTreeMisCl(train,numerical, categorical)


# In[24]:


pred_label = predict(root_misCl, 'left', testFile )
print "MisClassification: "
print pred_label
helper_validate(validate, root_misCl, 'prediction_misCl')

