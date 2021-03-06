#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:50:36 2020

@author: xiaoshou
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

"""

This helper function creates approximate variable values for SGMM-SGD methods.


"""

def SGD_init(train_data = None, n_clusters = None, percent_sample = None):
    """
    train_data: training data in dataframe (to be consistent with SGMM_SGD )
    n_clusters : number of clusters, integer valued 
    percent_sample: percent of samples to choose from training data, i.e. 0.05 
    """
    
    train_sample = train_data.sample(frac = percent_sample, random_state=1512)

    Xtrain, ytrain = train_sample.iloc[:,:-1].values, train_sample.iloc[:,-1].values
    
    # Estimator coefficient from logistic gression
    logimodel = LogisticRegression(penalty='l1', solver='liblinear', C = 1000).fit(Xtrain, ytrain)

    # Sample mean/cluster center from kmeans
    kmeans = KMeans(n_clusters = n_clusters, random_state = 1512).fit( Xtrain )
    Centers = kmeans.cluster_centers_ #* ((kmeans.cluster_centers_ > 0)*1.0)

    # W is the logistic regression weights, perturbed
    W = np.tile(logimodel.coef_.T, [1,n_clusters]) + np.random.normal(loc=0., scale=0.1, size=(Xtrain.shape[1],n_clusters))
    # w0 is the logistic regression bias, perturbed
    w0 =  np.tile(logimodel.intercept_, [1,n_clusters])  + np.random.normal(loc=0., scale=0.1, size=(n_clusters,)) 
    # perturbed sample variance 
    Noisy_Var = 1/ (d_newborn_tr.shape[0]-1)* np.sum(np.square( d_newborn_tr.iloc[:,:-1].values - Centers[:,np.newaxis] ),axis=1) +  np.random.normal(loc=0., scale=0.01, size=(n_clusters,d_newborn_tr.shape[1]-1))
    # Beta is log of diagonal scale  
    Beta = np.log(np.sqrt(np.maximum(Noisy_Var, 0.000001)))

    inits = {'Mu':np.transpose(Centers),'W': W,'w0':w0, 'Beta':Beta.T }
    
    
    return inits
