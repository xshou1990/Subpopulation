'''
This is just a wrapper function of seaborn clustmap.

@author: xiao shou

inputs:
1. means: 2darray (row: number of features, column: number of clusters)
2. variance associated with the means. 2darray (row: number of features, column: number of clusters)
3. featureslice: 1d array of features (dimension match the rows of means and variance)
4. clustpop : sample size for each clusters ,should be 1d array (dimension match the number of columns for means and variance)
5. normtype: integer, normalization(scale) type: none, row=0,column=1, 

(these 3 cases do not require input variances,clustpop, variances =None,clustpop = None)
                                    2-4:matrix entries as pvalues of t-test

    # 2 : 1- max (list of pvals), this means a given feature of a cluster has to differ from all of the rest of clusters;
    # 3 : 1- average (list of pvals), this means on average a given feature of a cluster has to differ from all of the rest of clusters;
    # 4 : 1 - min(list of pvals), this means a given feature of a cluster has to differ from any least one of the rest of clusters;

just added a feature cmap for selecting types values (positive ,negative, and zero) with 3 colors.
'''


import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import scipy.stats

# plot a simple nonscaled clustermap
def plotclustmap_simple(means,featureslice):
    cg = sns.clustermap(means,yticklabels = featureslice, xticklabels = np.arange(means.shape[0]),standard_scale=None,col_cluster=False, figsize =(14,14))
    cg.ax_row_dendrogram.set_visible(False)
    
    return 

def plotclustmap(means,variance,featureslice,clustpop,normtype):
    # map 3 levels
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue","#fed0fc","red"])
    if normtype == None:
        cg = sns.clustermap(means,yticklabels = featureslice, xticklabels = np.arange(means.shape[1]),standard_scale=None,col_cluster=False,cbar_kws={"ticks":[-1,0,1]}, figsize =(12,12),cmap=cmap) 
        #plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
        cg.ax_row_dendrogram.set_visible(False)
        cg.cax.set_visible(False)
    # row scaled    
    elif normtype == 0:
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=0,col_cluster=False)
    # column scaled    
    elif normtype == 1:
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=1,col_cluster=False)
    # 2 : 1- max (list of pvals)   
    elif normtype == 2:
        stdev = np.sqrt(variance)
        #pval =np.zeros(means.shape[1]-1,)
        pval_table = np.zeros((means.shape[0],means.shape[1]))
        for i in range(means.shape[0]):
            pval_temp = np.zeros(means.shape[1])
            pval_temp = np.diag(pval_temp)
            for j in range(means.shape[1]-1):
                for k in range(j+1,means.shape[1]):
                    tpval=scipy.stats.ttest_ind_from_stats(mean1=means[i,j], std1=stdev[i,j], nobs1=clustpop[j],
                                                           mean2=means[i,k], std2=stdev[i,k], nobs2=clustpop[k], equal_var=False)
            #print(tpval)
                    pval_temp[j,k]= tpval[1]
            pval_temp = pval_temp + pval_temp.T
            pval_table[i,:] = 1-pval_temp.max(axis=1)
        sns.clustermap(pval_table,yticklabels = featureslice, 
                       xticklabels =np.arange(means.shape[0]),standard_scale=None,col_cluster=False)
    # 3 : 1- average (list of pvals)
    elif normtype == 3:
        stdev = np.sqrt(variance)
        #pval =np.zeros(means.shape[1]-1,)
        pval_table = np.zeros((means.shape[0],means.shape[1]))
        for i in range(means.shape[0]):
            pval_temp = np.zeros(means.shape[1])
            pval_temp = np.diag(pval_temp)
            for j in range(means.shape[1]-1):
                for k in range(j+1,means.shape[1]):
                    tpval=scipy.stats.ttest_ind_from_stats(mean1=means[i,j], std1=stdev[i,j], nobs1=clustpop[j],
                                                           mean2=means[i,k], std2=stdev[i,k], nobs2=clustpop[k], equal_var=False)
            #print(tpval)
                    pval_temp[j,k]= tpval[1]
            pval_temp = pval_temp + pval_temp.T
            pval_table[i,:] = 1-pval_temp.mean(axis=1)
        sns.clustermap(pval_table,yticklabels = featureslice, 
                       xticklabels =np.arange(means.shape[0]),standard_scale=None,col_cluster=False)
    # 4 : 1 - min(list of pvals)
    elif normtype == 4:
        stdev = np.sqrt(variance)
        #pval =np.zeros(means.shape[1]-1,)
        pval_table = np.zeros((means.shape[0],means.shape[1]))
        for i in range(means.shape[0]):
            pval_temp = np.ones(means.shape[1])
            pval_temp = np.diag(pval_temp)
            for j in range(means.shape[1]-1):
                for k in range(j+1,means.shape[1]):
                    tpval=scipy.stats.ttest_ind_from_stats(mean1=means[i,j], std1=stdev[i,j], nobs1=clustpop[j],
                                                           mean2=means[i,k], std2=stdev[i,k], nobs2=clustpop[k], equal_var=False)
            #print(tpval)
                    pval_temp[j,k]= tpval[1]
            pval_temp = pval_temp + pval_temp.T
            pval_table[i,:] = 1-pval_temp.min(axis=1)
        sns.clustermap(pval_table,yticklabels = featureslice, 
                       xticklabels =np.arange(means.shape[0]),standard_scale=None,col_cluster=False)
    # 100 : print none ,row, column scaled
    elif normtype == 100:
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=None,col_cluster=False)
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=0,col_cluster=False)
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=1,col_cluster=False)
        
    return 
