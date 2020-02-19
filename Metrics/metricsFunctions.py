#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:36:54 2019

@author: george
"""

import sys

sys.path.append('..')
sys.path.append('../SGMM')
#sys.path.append('../metrics')
sys.path.append('../loaders')
sys.path.append('../oldCode')
sys.path.append('../visual')
sys.path.append('../testingCodes')
sys.path.append('../otherModels')

import numpy as np
import pandas as pd
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#from sklearn.model_selection import train_test_split
#from sklearn.cluster import KMeans
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
 balanced_accuracy_score, f1_score
#from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve






def CalculateSoftLogReg( models = [],  Xtrain = [], Xtest = [], ytrain = [],
                        ytest = [],  membTrain = [], membTest = []):
                         
            """
                This Function probably is not gonna be used in the
                future. I might modify it or leave it at some point
                The initial purpose was to calculate metrics for the
                Supervised Gaussian Mixtures Models. But I have incorporate
                the functionality inside the class
                
            """
            #NUMBER OF CLUSTERS  
            clusters = membTrain.shape[ 1 ]
                
            #INITIALIZE THE PROBABILITY MATRICES
            #FOR TRAIN AND TEST DATA
            probTest = np.zeros( [Xtest.shape[0] ] )
            probTrain = np.zeros( [Xtrain.shape[0] ] )
            
            for i in np.arange( clusters ): #FOR EACHH CLUSTER
                model = models[i]
                
                #PROBABILITY OF EACH POINT TO BELONG IN CLASS 1
                #TO USE FOR THRESHOLDING
                # FOR TRAIN AND TEST DATA
                probTest = probTest + model.predict_proba( Xtest )[ :, 1] \
                                                        * membTest[ :, i ]
                probTrain = probTrain + model.predict_proba( Xtrain )[ :, 1] \
                                                        * membTrain[ :, i ]
                
            
            #ALL THE METRICS WE WILL CALCULATE
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            
            #CALCULATE THE OPTIMAL THRESHOLD THAT MAXIMIZES THE F1
            #MEASURE
            
            
            tau = optimalTau( probTrain, ytrain)  #calculate optimal 
                                                  #tau on the training data
                
            #CALCULATING THE METRICS FOR TEST AND THE METRICS FOR TRAIN
            metricsTrain, roc1 =  calc_metrics(y = ytrain, tau = tau,
                                               custom_prob = probTrain)
            metricsTest, roc2 =   calc_metrics( y = ytest, tau = tau,
                                               custom_prob = probTest)
            
            #MAKING THE METRICS INTO DATAFRAMES WITH THEIR RESPECTIVE COLUMNS
            
            metricsTrain = pd.DataFrame( [metricsTrain], columns = columns)
            metricsTest = pd.DataFrame( [metricsTest], columns = columns)
            
            return metricsTrain, metricsTest, roc1, roc2, tau
        
        
def optimalTau(probabilities, ylabels, returnAll = 0, mode = 0, 
               targetMax = 0, targetValue = 0.2):
            
            """ Finds the Optimal tau based on the F1 score or precision or
                recall
                Input: Probabilities of train data of being class 1,
                ylabels of training data
                
                returnAll: 0 or 1 :Default 0, If 1 return a more complete
                set of parameters except tau
                mode: 0 (maximize F1 score), 1 (maximize precision), 2
                      (maximize) recall, 3(maximize accuracy)
                targetMax: We included one more parameter if targetMax is 0
                then our Target is precision  if our targetMax is 1 then the 
                target is recall
                targetValue:  maximize targetMax (precision or Recall) such that
                the target value (recall , precision) is at least "targetValue"
                """
            
            #STEP 1 SORT PROBABILITIES AND LABELS
            sortedIndexes = np.argsort( probabilities )
            probabilities1 = probabilities[ sortedIndexes ]
            ylabels1 = ylabels[ sortedIndexes ]
            
            #INITIALIZE THRESHOLD TO BE 0
            #SO EVERY POINT  IS PREDICTED AS CLASS 1
            
           # initialPrediction = np.ones( probabilities1.shape[0] ) #matrix with all 1's - INITIAL PREDICTION
            
            TP = len( np.where( ylabels1 == 1)[0] )  #AT THE BEGGINING THE TRUE POSITIVES ARE THE SAME 
                                                    #AS THE POSITIVE LABELS OF THE DATASET
            
            FN = 0 #AT THE BEGGINING  WE HAVE 0 POSITIVE POINTS  CLASSIFIED AS NEGATIVE
            TN = 0 #AT THE BEGINNING  WE HAVE 0 NEGATIVE POINTS
         
            FP = len( np.where( ylabels1 == 0)[0] ) # AT THE BEGINNING FALSE POSITIVES ARE THE SAME AS NEGATIVEES IN THE SET
            precision = TP/(TP + FP)
            try:
                recall = TP/ (TP + FN)
            except:
                recall = 0
            accuracy = (TP + TN)/(TP + FN + FP + TN)
            
#            print(precision, recall, TP, FN, FP)
#            return
            f1 = ( 2*precision*recall )/( precision + recall )   
            
            threshold = 0
            prob_F1 = [[threshold, f1]]
            prec = precision
            rec = recall
            #list with f1, precision , recall
            metOld = [f1, precision, recall, accuracy]
            #precision recall list of lists
            precRec = [precision, recall]
            #tau that maximizes target with given targetvalue
            tauTarget = -1
            
            for i, probability in enumerate( probabilities1 ):
                
               # print( " Iteration: {}".format(i))
                
                
                if ylabels1[i] == 1:
                    
                    TP -= 1
                    FN += 1
                
                if ylabels1[i] == 0: 
                    FP -= 1
                    TN += 1
                    
                if (TP + FP == 0):
                    
                    precision = 0
                    
                else:
                    precision = TP/(TP + FP)
                
                try:
                    recall = TP/ (TP + FN)
                except:
                    recall = 0
                    
                accuracy = (TP + TN)/(TP + TN + FP + FN)
                
                if (precision + recall) == 0:
                
                    f1new = 0
                    metNew = [f1new, precision, recall, accuracy]

                else:
                    
                    f1new = ( 2*precision*recall )/( precision + recall )  
                    metNew = [f1new, precision, recall, accuracy]
                    
                #thresholds with F1 scores if you want to draw a graph
                prob_F1.append( [probability, metNew[mode]] )  
                
                if targetMax == 0: #maximize precision with given recall
                    
                    if recall >= targetValue:
                        tauTarget = probability
                        precRec.append([precision, recall])
                
                else:
                    
                    if precision >=  targetValue:
                        tauTarget = probability
                        precRec.append([precision, recall])
                    
                
                #maximize  f1 precision or recall
                if metNew[mode] >= metOld[mode] :
                    threshold = probability
                    f1 = f1new
                    prec = precision  #you can return precision
                    rec = recall      #you can return  recall
                    acc = accuracy
                    metOld  = [f1, prec, rec, acc]
                   # print(acc, threshold)
                                        
            
            #OUTSIDE THE LOOP
            if returnAll == 1:
                params = {'tau': threshold, 'curve': np.array(prob_F1), 
                          'precision': prec, 'recall': rec, \
                          'precRec': precRec, 'tauTarget': tauTarget}
                return params
            
            return threshold #, f1, np.array(prob_F1), prec, rec
        
def calc_metrics(model = None, cluster = -1, y = None, tau = 0.5, 
                 custom_prob = None, putModels = 0 , X = None):
            
             """              
                 COMPUTES METRICS OF THE ALGORITHM
                 Acuraccy, Balanced acuraccy, Auc, Precision,
                 RSpecificity, Sensitivity,  TP, TN, FP, FN,
                 Percentage of High Cost Patients
                 Percentage of Low Cost Patients
                 
                  
                 y: training or testing labels 
                 tau: Threshold for probabilities
                 custom_prob: Probabilities produced by the model
                              based  on which you want to calculate
                              the class, these correspond
                              for a datapoint to belong to class 1
                 putModels:  Checks if you put  model to do the predictions
                             or the probabilities for each data point 
                             to belong to  class 1.
                     
             """
             if  putModels != 0 :
                 probabilities = model.predict_proba( X )[:,1]
            
             else:
                 
                 probabilities = custom_prob
                
             try:       
                 auc = roc_auc_score( y , probabilities)  
                 roc = roc_curve(y, probabilities)
             except:
                 print( "Problem in Calculating auc")
                 auc = 0
                 roc = []
             #Calculate tau if calc_tau is 1
             #Given we have provided probability matrix
            
             
             #THRESHOLDING BASED ON TAU IN ORDER TO GET THE 
             #ESTIMATED LABELS FOR EACH DATAPOINT
             probabilities[ np.where( probabilities > tau ) ] = 1
             probabilities[ np.where( probabilities <= tau ) ] = 0
             predictions = probabilities
              
             #METRICS CALCULATION
             precision =  precision_score(y, predictions) #CALCULATE THE PRECISION
             sensitivity = recall_score(y, predictions)  #CALCULATE THE RECALL
             accuracy = accuracy_score(y, predictions) #CALCULATE THE ACCURACY
             bal_acc = balanced_accuracy_score(y, predictions) #CALCULATE THE BALANCED ACCURACY
             f1 = f1_score(y, predictions)
             
             clusterSize = len( y )  #Cluster Size
             highCostPerc = len( np.where( y == 1)[0] )/clusterSize
             lowCostPerc = len( np.where( y == 0)[0] )/clusterSize
             
             
             TP = len( np.where(  (y == 1) * (predictions == 1) )[0] )
             TN = len( np.where(  (y == 0) * (predictions == 0) )[0] )
             
             FP = len( np.where(  (y == 0) * (predictions == 1) )[0] )
             
             FN = len( np.where(  (y == 1) * (predictions == 0) )[0] )
             
             #print(TP, TN, FP, FN, clusterSize)
             try:
                 specificity = TN/(FP + TN)
                 FPR = 1 - specificity
             except:
                 specificity = -1000
                 FPR = -1000
             #PUT ALL THE METRICS IN A LIST AND RETURN THEM
             metrics =  [cluster, clusterSize, highCostPerc, lowCostPerc,
                         TP, TN, FP, FN,
                         FPR, specificity, sensitivity, precision, 
                         accuracy, bal_acc, f1, auc]
             
             return np.array(metrics), roc
            
            
def predict_y(probabilities, tau = 0.5):
     """return the predictions given probabilities and thresholds """
    
     probabilities[ np.where( probabilities >= tau ) ] = 1
     probabilities[ np.where( probabilities < tau ) ] = 0
     predictions = probabilities
     
     return predictions
    
         
            
def metrics_cluster(models = None, ytrain = None, ytest = None,
                                          testlabels = None,
                                          trainlabels = None,
                                          Xtrain = None,  Xtest = None):
            """
             Calculates Metrics such as accuracy, balanced accuracy,
             specificity, sensitivity, precision, True Positives,
             True Negatives etc.
             
             These metrics are calculated for each cluster:
             models: predictive models trained in each cluster
             ytrain: Target labels of training set 
             ytest: target labels of test set
             testlabels: a matrix with numbers from 0 to c-1 number of clusters
                         indicating in which cluster each data point belongs
                         in the test set
             trainlabels: the same as testlabels but for training data
             Xtrain: trainiing data
             Xtest: testing data
                     
            
            
            """
           
           
            # matrix with metrics for each cluster
            metricsTrain = []   
            #metrics for test  data in each cluster
            metricsTest = []  
            
            
            
            
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            
            
            #Calculate the Metrics for Each Cluster
            for  cluster in np.arange( len( models ) ):  
                #INDEXES OF CLUSTER "cluster"
                inC = np.where( trainlabels == cluster )[0] 
                inCT = np.where( testlabels == cluster )[0]
                
                #predict probabilities  of data in cluster "cluster"
                #to be 1
                probTrain = models[cluster].predict_proba(Xtrain[inC])[:, 1]
                probTest = models[cluster].predict_proba(Xtest[inCT])[:, 1]
                
                #calculate optimal tau based on F1
                try:
                    tau = optimalTau(probTrain, ytrain[inC])
                except:
                    tau = 0.5
                    print(" Warning tau setted to 0.5 due to error(s) \
                          in <<optimalTau>> function" )
                #CALCULATE METRICS : ACCURACY, RECALL, PRECISION ,
                #BALANCED ACCURACY ETC
                metTrain , _= calc_metrics(  custom_prob = probTrain, 
                                                  y = ytrain[inC], 
                                                  cluster = cluster, 
                                                  tau = tau )
                
                metTest, _ = calc_metrics(  custom_prob = probTest,
                                                  y = ytest[inCT], 
                                                  cluster = cluster, 
                                                  tau = tau)
                
                metricsTrain.append( metTrain )
                metricsTest.append( metTest )
                    
                   
                   
                    
            #Create a dataframe with metrics for better Visualization         
            metricsTrain = pd.DataFrame ( metricsTrain, columns = columns )
            metricsTest = pd.DataFrame( metricsTest, columns = columns ) 
                
            return metricsTrain, metricsTest
        
def sgmmResults( model, probTest, probTrain, ytest, ytrain, tau = None,
                mode = 3, Xtest = None):
    #a Summary of predictions and interesting model parameters
    #mixing coef
    pis = model.pis
    #means
    means = model.means
    #covariances
    cov = model.cov
    #weights
    weights = model.weights
    #logistic regressions
    logRegr = model.LogRegr
    #train memberships
    if Xtest is None:
        mTest = model.mTest
    else:
        mTest = model.predict_GMMS(Xtest)
    #test memberships
    mTrain = model.mTrain
    
    best_alphas = []
    
    for logR in logRegr:
        best_alphas.append( logR.best_params_)
        
    #CALCULATE THE OPTIMAL TAU
    if tau is None:
        tau = optimalTau(probTrain, ytrain, mode = mode)
    
    metTest,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, 
                                                                     y = ytest)
    metTrain ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau,
                                                                     y = ytrain)
    
    predictTrain = predict_y( probTrain.copy(), tau )
    predictTest = predict_y( probTest.copy(), tau )
    
    columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']

    metTestSGMM = pd.DataFrame( [metTest], columns = columns)
    metTrainSGMM = pd.DataFrame( [metTrain], columns = columns)
    
    Ntrain = ytrain.shape[0]
    Npos = len( np.where( ytrain == 1)[0])
    Nneg = Ntrain - Npos
    posPerc = Npos/Ntrain
    negPerc = Nneg/Ntrain
    
    results = {"testMet": metTestSGMM, "trainMet": metTrainSGMM,
               "yTest": predictTest, "yTrain": predictTrain, "memberTr":
                   mTrain, "memberTest": mTest, "means": means, "weights":
                       weights, "cov": cov, "tau": tau, "best_alphas":
                           best_alphas, 'pis': pis, "posP": posPerc,
                           "negP": negPerc}
        
    return results
    
    
    
        
        
        