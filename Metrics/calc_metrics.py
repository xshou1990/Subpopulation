import numpy as np
import pandas as pd
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
  
def calc_metrics(model = [], cluster = -1, y = [], tau = 0.5, 
                 custom_prob = [], putModels = 0 , X = []):
            
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
                
                    
             auc = roc_auc_score( y , probabilities)  
             roc = roc_curve(y, probabilities)
             
             #Calculate tau if calc_tau is 1
             #Given we have provided probability matrix
            
             
             #THRESHOLDING BASED ON TAU IN ORDER TO GET THE 
             #ESTIMATED LABELS FOR EACH DATAPOINT
             probabilities[ np.where( probabilities >= tau ) ] = 1
             probabilities[ np.where( probabilities < tau ) ] = 0
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
             
             specificity = TN/(FP + TN)
             FPR = 1 - specificity
             
             #PUT ALL THE METRICS IN A LIST AND RETURN THEM
             metrics =  [cluster, clusterSize, highCostPerc, lowCostPerc,
                         TP, TN, FP, FN,
                         FPR, specificity, sensitivity, precision, 
                         accuracy, bal_acc, f1, auc]
             
             return metrics, roc