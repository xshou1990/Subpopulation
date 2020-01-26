#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:42:33 2019

@author: george & xiao
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB





def logisticRegressionCv2(Xtrain = None, ytrain = None, Xtest = None,
                                  ytest = None, Cs = [10], penalty = 'l1',
                                 solver = 'saga', scoring = 'neg_log_loss'):
    
    model = LogisticRegressionCV(Cs = Cs, penalty = penalty, random_state = 0,
                                 solver = solver, scoring = scoring,cv = 10)\
                                 .fit(Xtrain, ytrain)
                                 
    probTrain = model.predict_proba( Xtrain )[:, 1]
    probTest = model.predict_proba( Xtest )[:, 1]
    
    
    params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
    return params, probTest, probTrain



def neural_nets(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None, h_l_s = (5, 3, 2), cv = 10,
                          scoring = 'neg_log_loss'):
    
    
     sgd  =  MLPClassifier( hidden_layer_sizes = h_l_s, early_stopping = True,
                                                              random_state = 0)
     param_grid = {'alpha' : [0.1,  0.01, 0.001, 1, 2, 5, 10, 12, 100]}
            
     model = GridSearchCV( sgd, param_grid = param_grid, 
                                   n_jobs = -1, 
                               scoring = scoring, cv = cv).fit(Xtrain, ytrain)
            
     probTrain = model.predict_proba( Xtrain )[:, 1]
     probTest = model.predict_proba( Xtest )[:, 1]
     
     params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
     return params, probTest, probTrain


def kmeansLogRegr( Xtrain = None, ytrain = None, Xtest = None,
                  ytest = None, Cs = [10], penalty = 'l1', 
                  solver = 'saga', scoring = 'neg_log_loss', n_clusters = 2,
                  adaR = 1):
    
    #CLUSTER WITH KMEANS
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).\
             fit( np.concatenate(( Xtrain, Xtest ), axis = 0) )
             
    #TAKE THE LABELS
    labelsTrain = kmeans.labels_[0: Xtrain.shape[0]]
    labelsTest = kmeans.labels_[ Xtrain.shape[0]:]
    
    #TRAIN LOGISTIC REGRESSION
    models = [] 
    probTrain = []
    probTest = []
    for i in np.arange( n_clusters ):
        indxTr = np.where(labelsTrain == i)[0]
        indxTest = np.where( labelsTest == i)[0]
        
        if adaR == 1:
            Csnew = (np.array(Cs)/len(indxTr)).tolist()
        
        params, _, _ = logisticRegressionCv2(Xtrain = Xtrain[indxTr], 
                                       ytrain = ytrain[indxTr], 
                                       ytest = ytest[indxTest],
                                       Xtest = Xtest[indxTest], 
                                       Cs = Csnew, penalty = penalty,
                                       solver = solver, scoring = scoring)
        
        models.append( params['model'] ) 
        probTrain.append( params['probTrain'] )
        probTest.append( params['probTest'] )
    
    params = {'models': models,'labelsTrain': labelsTrain,
              'labelsTest': labelsTest, 'probTrain': probTrain,
              'probTest': probTest}
    
    return params
        
    
def kmeansBNB( Xtrain = None, ytrain = None, Xtest = None,
                  ytest = None, n_clusters = 2):
    
    #CLUSTER WITH KMEANS
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).\
             fit( np.concatenate(( Xtrain, Xtest ), axis = 0) )
             
    #TAKE THE LABELS
    labelsTrain = kmeans.labels_[0: Xtrain.shape[0]]
    labelsTest = kmeans.labels_[ Xtrain.shape[0]:]
    
    #TRAIN NaiveBNB
    models = [] 
    probTrain = [] 
    probTest = []
    for i in np.arange( n_clusters ):
        indxTr = np.where(labelsTrain == i)[0]
        indxTest = np.where( labelsTest == i)[0]
        
        bnb =BernoulliNB(alpha=1)
        bnb.fit(Xtrain[indxTr], ytrain[indxTr])
        probTrainNB,probTestNB = bnb.predict_proba(Xtrain[indxTr])[:,1], bnb.predict_proba(Xtest[indxTest])[:,1]
        
        models.append( bnb ) 
        probTrain.append( probTrainNB )
        probTest.append( probTestNB )
    
    params = {'models': models,'labelsTrain': labelsTrain,
              'labelsTest': labelsTest, 'probTrain': probTrain,
              'probTest': probTest}
    
    return params        
    

def randomforests(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None, cv = 10, scoring = 'neg_log_loss'):
    
    "RANDOM FOREST CLASSIFIER"
    
    param_grid = {'n_estimators' : [20,  40, 60, 80, 100, 120, 150,500,900,
                                    1100] }
    forest  = RandomForestClassifier()
    
    model = GridSearchCV( forest, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain) #fit model 
    
    
    probTrain = model.predict_proba( Xtrain )[:, 1]
    probTest = model.predict_proba( Xtest )[:, 1]
     
    params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
    return params, probTest, probTrain

def xboost(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None, cv = 10, scoring = 'neg_log_loss'):
    
    param_grid = {'n_estimators' : [20,  40, 60, 80, 100, 120, 150, 700,
                                    900, 1100] }
    ada = AdaBoostClassifier()
    
    model = GridSearchCV( ada, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain) #fit model 
    
    
    probTrain = model.predict_proba( Xtrain )[:, 1]
    probTest = model.predict_proba( Xtest )[:, 1]
     
    params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
    return params, probTest, probTrain
   
    
def gradboost(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None, cv = 10, scoring = 'neg_log_loss'):
    
    "RANDOM FOREST CLASSIFIER"
    
    param_grid = {'n_estimators' : [20,  40, 60, 80, 100, 120, 150, 300, 500,
                                    700, 800, 900]}
    grad  = GradientBoostingClassifier(subsample = 0.5, max_features = 'sqrt',
                                       learning_rate = 0.01, max_depth = 3)
    
    model = GridSearchCV( grad, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain) #fit model 
    
    
    probTrain = model.predict_proba( Xtrain )[:, 1]
    probTest = model.predict_proba( Xtest )[:, 1]
     
    params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
    return params, probTest, probTrain
    
    