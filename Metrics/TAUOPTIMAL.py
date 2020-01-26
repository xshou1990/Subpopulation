import numpy as np
def optimalTau(probabilities, ylabels):
            
            """ Finds the Optimal tau based on the F1 score"""
            
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
            #XIAO HERE YOU WILL PUT  ylabels == -1
            FP = len( np.where( ylabels1 == -1)[0] )
            
            precision = TP/(TP + FP)
            recall = TP/ (TP + FN)
            
#            print(precision, recall, TP, FN, FP)
#            return
            f1 = ( 2*precision*recall )/( precision + recall )   
            
            threshold = probabilities1.min()-0.1
            prob_F1 = [[threshold, f1]]
            
            for i, probability in enumerate( probabilities1 ):
                
                #print( " Iteration: {}".format(i))
                
                
                if ylabels1[i] == 1:
                    
                    TP -= 1
                    FN += 1
                
                if ylabels1[i] == -1: #FOR XIAO HERE -1
                    FP -= 1
                    
                if (TP + FP == 0):
                    
                    precision = 0
                    
                else:
                    precision = TP/(TP + FP)
                    
                recall = TP/ (TP + FN)
                
                if (precision + recall) == 0:
                
                    f1new = 0
                    
                else:
                    
                    f1new = ( 2*precision*recall )/( precision + recall )  
                
                prob_F1.append( [probability, f1new] )   #thresholds with F1 scores if you want to draw a graph
                
                if f1new >= f1 :
                    threshold = probability
                    f1 = f1new
                    prec = precision
                    rec = recall
                    
            
            return threshold, f1, np.array(prob_F1), prec, rec
