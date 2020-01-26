#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:13:59 2019
@author: george
"""

import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#RELATED TO WORDCLOUDS

def CreateClouds( data = None, labels = None, names = None, 
                 n_clusters = None, save = 1, dirCreate = 1, 
                                                 filename = 'Clouds',
                                                 dirName = 'Cdir', TFIDF = 1 ):
    
    
    
     """
     GENERAL FUNCTION ON CREATING CLOUDS
     data: Dataset to create clouds on
     labels: the cluster label of each data point
     names:[LIST OF STRINGS] column names of the dataset data (features' names)
     n_clusters = number of clusters 
     save: save the wordclouds or not Default: 1(yes)
     dirCreate: create A new directory to save the clouds, Default: 1(yes)
     filename: name if the wordcloud, it will append indexes to the name
             based on the cluster
     dirName: the name of the directory to save the clouds
     TFIDF: Use TFIDF or RAW Counts: Default: 1(use TFIDF)
     returns the dictionaries created for each class
     """
     dictions = CalculateCounts(data = data, names = names, TFIDF = TFIDF,
                               labels = labels, n_clusters = n_clusters)
    
     for  i in np.arange( n_clusters ):
         
         filenamenew = filename + str(i)
         clouds(counts = dictions[i], filename = filenamenew,
                dirName = dirName, dirCreate = dirCreate)
         
     return dictions
        
    
def CreateCloudsWeights( weights = None, names = None, n_clusters = None,
                                    save = 1, dirCreate = 1, filename = 'WC',
                                                 dirName = 'WCC', number = 50 ):
    
    """SAME AS CreateClouds but now it takes as inputs a list of each class
       weights and makes the clouds based on them """
       
    dictP, dictN = CalculateWeights( weights = weights, names = names,
                                    n_clusters = n_clusters, number = number)
    
    
    for i in np.arange( n_clusters ):
        filenamePos = filename+'Pos'+str(i)
        filenameNeg = filename+'Neg'+str(i)
        
        clouds( counts = dictP[i], filename = filenamePos, dirName = dirName,
                                                       dirCreate = dirCreate)
        clouds( counts = dictN[i], filename = filenameNeg, dirName = dirName,
                                                       dirCreate = dirCreate)
    
    params = {'dictp':dictP, 'dictN': dictN}   
    
    
    return params
        
        
  
    

def clouds( counts = None, dirCreate = 1, save = 1, filename = None,
                                                     dirName = None,
                                                     ):
    
    
    """
        Takes a Dictionary with features names and their corresponding counts
        and creates a wordcloud. 
        
        You can specify to create a directory relative to the current 
        directory and save thw worldCloud there. Or you can chose to not
        create a directory and not save the wordCloud.
        Also you must specify the name of the world cloud if you decide to
        save it.
       
        
    """
    
    cloud = WordCloud(stopwords = STOPWORDS, min_font_size = 4,
                      background_color = 'skyblue',colormap="Dark2", margin = 1).\
                      generate_from_frequencies( counts )
    #plot the cloud                 
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    fig.set_size_inches(12,12)
    
    
    ax.imshow(cloud, interpolation = 'bilinear')
    ax.axis('off')
    
    
       
    
                      
    #save the file
    if save == 1:
        filename = filename+'.png'
        
        #create a directory with dirName to save the cloud
        if dirCreate == 1:
            #check if Dir exist
            dirname = './'+ dirName
            p = Path(dirname)
            
            if  p.is_dir():
                fig.savefig(dirname+'/'+filename, format = 'png', dpi = 300)
            else:
                p.mkdir()
                fig.savefig(dirname+'/'+filename, format = 'png', dpi = 300)
        
    else:
        fig.savefig(filename, format = 'png', dpi = 300)
    
    return
        
        
def CalculateCounts( data= None, names = None, TFIDF = 1, labels = None,
                    n_clusters = None):
    """
      Takes data matrix and the column names and returns a dictionary
      with the counts and columns names
      two choices: Either raw counts, or TFIDF counts
      it does this for all the clusters
     
      
    """
    #CALCULATE COUNTS WITH TFIDF
    cloudsdicts = []
    
    if TFIDF == 1:
        counts = calcTFIDF( data = data, labels = labels, 
                                   n_clusters = n_clusters )
        
        
        for i in np.arange( n_clusters ):
            clDict = dict( zip( names, counts[i] ) )
            cloudsdicts.append( clDict )
         
        return cloudsdicts
    
    #calculate the raw counts of each cluster
    else:
        for i in np.arange( n_clusters ):
            inC = np.where( labels == i )[0]
            rawcount_i  = np.sum( data[inC], axis = 0 )
            countDict = dict( zip( names, rawcount_i ))
            cloudsdicts.append( countDict )
            
    return cloudsdicts      
        
def CalculateWeights( weights = None, names = None, n_clusters = None,
                     number = None)  :
    
    """
        Creates Dictionaries for each cluster weights
        number: number of features to take
        
    """
    #if we input a number bigger than the number of features 
    #then confine the number to the number of features
    
      
    posDicts = []
    negDicts = []
    for i in np.arange( n_clusters ):
        
        #KEEP TRACK OF THE INITIAL NUMBER O FEATURES WE WANT TO KEEP
        numberPos = number
        numberNeg = number
        
        w = np.array( weights[i] )
        #TAKE POSITIVE AND NEGATIVE WEIGHTS (some might be 0)
        wpos = w[ np.where( w > 0)[0] ]
        namesPos =  np.array( names )[ np.where( w > 0)[0]].tolist()
        
        #wneg = np.abs( w[ np.where( w < 0)[0] ] )
        wneg = ( w[ np.where( w < 0)[0] ] )
        namesNeg =  np.array( names )[ np.where( w < 0)[0]].tolist()
        
        #SORT THE INDEXES OF THE POS AND NEG WEIGHTS
        spos = np.argsort( wpos )
        sneg = np.argsort( wneg )
        
        if len( spos ) < number:
            numberPos = len( spos )
            
        if len( sneg ) < number:
            numberNeg = len( sneg )
        
        #TAKE THE TOP "NUMBER" NUMBER OF INDEXES
        indexPos = spos[-numberPos : ]
        #indexNeg = sneg[-numberNeg : ]    
        indexNeg = sneg[ 0: numberNeg  ]   
        
        #TAKE THE CORRESPONDING NAMES
        namesPos = np.array( namesPos )[indexPos].tolist()
        namesNeg =  np.array( namesNeg )[indexNeg].tolist()
        
        dPos = dict( zip( namesPos, wpos ))
        dNeg = dict( zip( namesNeg, wneg ))
        
        posDicts.append( dPos )
        negDicts.append( dNeg )
    
    return posDicts, negDicts
        
        
        
        
        
def calcTFIDF(  data = None, labels = None, n_clusters = None ):
       """
           HELPER FUNCTION ON CREATING THE CLOUDS
           IT RECALCULATES COUNTS WITH THE HELP OF TFIDF MEASURE
       """
    
       tfIdf = []
       idf = np.zeros( data.shape[1] )
       dataCount = np.sum( data, axis = 0)
            
       for  i in np.arange( n_clusters ):
            index = np.where( labels == i )[0]
            count = np.sum( data[index], axis = 0 )
            #account the Clusters percentage in data Points
            clusterWeight = len(index)/data.shape[0]  
            # totCount = np.sum(count)/ ( 2 * data.shape[1] )
                
            for j in np.arange( data.shape[1] ):
                  
                if count[j] > (dataCount[ j ] * clusterWeight) :
                    idf[j] += 1
                    
            tfIdf.append( count )
       #this is needed  when  we have little data  
       idf[ np.where(idf == 0) ] = n_clusters  
       for i in np.arange( n_clusters ):
           tfIdf[i] = np.ceil(tfIdf[i]  * 
                        (np.log( n_clusters /  idf )/ np.log( n_clusters )))
                                                     # /np.log( n_clusters ) ) )
                
#        #print(max(idf), n_clusters)
#        m = np.asarray(tfIdf)
#        l = m / (m.sum(axis=0)+1e-8)
#        variance = np.zeros(m.shape[1])
#        for i in range(m.shape[1]):
#            variance[i] = l[:,i].var()
#        t=variance<0.15
#        n=[i for i, x in enumerate(t) if x]
#        m[:,n] = 0
#        tfIdf = m.tolist()
       return tfIdf   

  
#PARALLEL COORDINATES    
       
def plot_parallel( data, columns, indx, scale = 0):
    """THIS CAN BE USED TO PLOT THE MEANS
    OF THE CLUSTERS AS PARALLEL COORDINATES
    Parallel Coordinates Plot for Multi Dimensional
    Features Visualization
    data: A data matrix of the form observations-features
    columns: the names of the columns corresponding  to features
    indx:  a list of the features to pick (ideal 3-10 above that it becomes
    messy) of the form of  index numbers ex [1,2, 10, 70, 5,...]
    range of indexes 0 to number of features -1
    scale: if you want to use a row wise standard scaler of the means
    and variance: Defauly scale = 0 which means no scale, scale = 1 means do
    scaling
    
    """
    
    #length = len(data[0])
    if scale == 1:
        dataNp = np.array(data)
        std = StandardScaler().fit_transform(dataNp)
        data  = pd.DataFrame( std, columns = columns)
    else:
         data  = pd.DataFrame( data, columns = columns)
    
    data['clusters'] = np.arange( data.shape[0] )
    dataIndex =  indx.copy()
    dataIndex.append( data.shape[1] -1)
    ax = parallel_coordinates( data.iloc[:, dataIndex], 'clusters')
                              #,use_columns = True)
                             # xticks = np.arange(len(dataIndex) -1) )
    ax.set_title('Parallel Coordinates Plot')
    ax.set_ylabel('Feature Value')
    ax.set_xlabel('Features')
    plt.show()
    #ax.grid(b = False)
    #plt.axis('off')
    #ax.set_xticks([])
   # ax.set_yticks([])
   
    
   
    return ax

#HEATMAPS
def heatmap( data, columns, index, annot = False, fmt = '.1f' ):
    
    data = pd.DataFrame( data, columns = columns)
    data = data.iloc[:, index]
    
    fig, ax = plt.subplots(1, 1, figsize = (9,6))
    ax = sns.heatmap( data, annot = annot, fmt = fmt, ax = ax)
    
    ax.set_title(' Heatmap of Clusters Versus Features ')
    ax.set_xlabel( 'Features ')
    ax.set_ylabel( 'Clusters' )
    plt.show()

    
    
#SOME HELPER FUNCTIONS
def findbinary( data, columnNames = None ):
    """
    Finds all the columns with binary values [0,1]
    data: Numpy array of data
    columnNames: optional give the names of the features
    """
    binIndex = []
    names = []
    
    for i in np.arange( data.shape[1] ):
        
        if np.isin([0,1],data[:, i]).all() == True:
            binIndex.append(i)
            
            if columnNames is not None:
                names.append( columnNames[i] )
    
    return binIndex, names