import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


def plotclustmap(means,variance,featureslice,clustpop,normtype):
    # means 2darray should be row: number of features, column: number of clusters
    # variance 2darray should be row: number of features by column: number of clusters
    # featureslice: 1d array of features (dimension match the rows of means and variance)
    # normtype: integer, normalization type: none =0, row=1,column=2 , 3-5:matrix entries as pvalues of t-test
    # 4 : 1 - min(list of pvals), this means a given feature of a cluster has to differ from any least one of the rest of clusters;
    # 2 : 1- max (list of pvals), this means a given feature of a cluster has to differ from all of the rest of clusters;
    # 3 : 1- average (list of pvals), this means on average a given feature of a cluster has to differ from all of the rest of clusters;
    # clustpop : sample size for each clusters ,should be 1d array (dimension match the number of columns for means and variance)
    clustname = ['Upstate Mild','Upstate Urgent','Upstate Low BW','Downstate Healthy','Low BW ER','Sick Travelers I','Sick Travelers II']
    #clustname = np.arange(10)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue","gainsboro","red"])
    if normtype == None:
        cg = sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=None,col_cluster=False, figsize =(14,14),cmap=cmap) 
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
        cg.ax_row_dendrogram.set_visible(False)
        cg.cax.set_visible(False)
        
    elif normtype == 0:
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=0,col_cluster=False)
    elif normtype == 1:
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=1,col_cluster=False)
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
    elif normtype == 100:
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=None,col_cluster=False)
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=0,col_cluster=False)
        sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=1,col_cluster=False)
        
    return 