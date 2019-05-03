#!/usr/bin/env python
# coding: utf-8

# ## Q2. Problem of Generating New Data

# - handwritten digitsdataset available in sklearn

# In[1]:


from sklearn.datasets import load_digits


# In[2]:


digits = load_digits()


# In[3]:


x = digits.data
y = digits.target
print(digits.data.shape)
print (type(digits.data))
# print y
classes = len(set(y))
print "classes: ",classes


# **Apply dimensionality reduction using PCA**

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[5]:


def doPCA(x, num_features):
    pca = PCA(n_components=num_features)
    x = pca.fit_transform(x)
    return x, pca


# In[6]:


x_31, pca_31 = doPCA(x,31)
x_15, pca_15 = doPCA(x,15)
x_41, pca_41 = doPCA(x,41)
print x_15.shape


# ### PART 1: Kernel Density Estimation: Use grid search cross validation on the reduced feature data to optimize bandwidth. Compute Kernel Density Estimate

# In[7]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
def GridCVKDE(x):
    param_grid = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), param_grid, cv=5).fit(x)
    
    kde = grid.best_estimator_
    print kde
    return kde


# KDE for 15 dimensions

# In[11]:


kde_15 = GridCVKDE(x_15)


# KDE for 31 dimensions

# In[12]:


kde_31 = GridCVKDE(x_31)


# KDE for 41 dimensions

# In[13]:


kde_41 = GridCVKDE(x_41)


# ### PART 2: Gaussian Mixture Model based Density Estimation: Use Bayesian Information Criteria to find  the number of GMM components we should use apply GMM using the using the above number of components.

# In[14]:


def plot(x, y, x_label, y_label, title):
    fig, axes = plt.subplots(figsize=(12,3))
    axes.plot(x, y, 'r')
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title);
    axes.grid(True)


# In[15]:


from sklearn.mixture import GaussianMixture
def doBIC(x):
    components_list = np.arange(50, 210,10)
    model_list = []
    for n in components_list:
        model = GaussianMixture(n, covariance_type='full', random_state=0)
        model_list.append(model)
    
    bic_list = []
    for one_model in model_list:
        bic_ = one_model.fit(x).bic(x)
        bic_list.append(bic_)
    
    plot(components_list, bic_list, 'number of components', 'bic', 'component vs bic')


# BIC for 15 dimensions

# In[16]:


doBIC(x_15)


# BIC for 31 dimensions

# In[17]:


doBIC(x_31)


# BIC for 41 dimensions

# In[18]:


doBIC(x_41)


# In[19]:


def BIC_GMM(x, min_components, cov_type):
    gmm = GaussianMixture(min_components, covariance_type=cov_type, random_state=0).fit(x)
    print gmm
    return gmm


# GMM for 15 dimensions

# In[20]:


gmm_15 = BIC_GMM(x_15, 200, 'full')


# GMM for 31 dimensions

# In[21]:


gmm_31 = BIC_GMM(x_31, 100, 'full')


# GMM for 41 dimensions

# In[22]:


gmm_41 = BIC_GMM(x_41, 65, 'full')


# ### PART 3: Draw  48  new points  in  the  projected  spaces using  both  the above generative models.  Use Inverse transform of PCA to construct new digits. Plot these points from both the models

# In[23]:


def plot_digits(data):
    fig, ax = plt.subplots(8, 6, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap = 'Greys')
        im.set_clim(0, 16)


# - KDE

# In[32]:


def reconstructKDE(kernel,pca):
    data_new = kernel.sample(48, random_state=0)
    
    digits_new = pca.inverse_transform(data_new)
    print "Reconstructed using: ",kernel
    plot_digits(digits_new)


# In[33]:


reconstructKDE(kde_15, pca_15)


# In[34]:


reconstructKDE(kde_31, pca_31)


# In[35]:


reconstructKDE(kde_41, pca_41)


# - GMM

# In[36]:


def generateGMM(kernel, pca):
    data_new,y = kernel.sample(48)
    digits_new = pca.inverse_transform(data_new)
    print "Reconstructed using: ",kernel
    plot_digits(digits_new)


# In[37]:


generateGMM(gmm_15, pca_15)


# In[38]:


generateGMM(gmm_31, pca_31)


# In[39]:


generateGMM(gmm_41, pca_41)

