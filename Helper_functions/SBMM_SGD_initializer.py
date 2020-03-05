#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:50:36 2020
@author: xiaoshou
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

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

    # Perturbed sample mean
    Centers = np.mean(Xtrain,axis=0)[:,np.newaxis] + np.random.normal(loc=0., scale=0.1, size=(Xtrain.shape[1],n_clusters))
    # project into positive unit box (0,1) for numerical stability for Bernoulli probability
    Centers =  np.minimum( np.maximum(Centers, 0.000001),0.99999) 

    # W is the logistic regression weights, perturbed
    W = np.tile(logimodel.coef_.T, [1,n_clusters]) + np.random.normal(loc=0., scale=0.1, size=(Xtrain.shape[1],n_clusters))
    # w0 is the logistic regression bias, perturbed
    w0 =  np.tile(logimodel.intercept_, [1,n_clusters])  + np.random.normal(loc=0., scale=0.1, size=(n_clusters,)) 

    inits = {'Mu':Centers,'W': W,'w0':w0 }
    
    
    return inits
