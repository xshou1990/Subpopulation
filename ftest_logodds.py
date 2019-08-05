""" 

@ author: Xiao Shou

ftest_corr: This function calculate the p-value for each feature for each class(cluster) 
by fisher's exact test and returns a table of log odds ratio for each feature for each class
that is significant at alpha = 0.01, along with the features themselves. 
For multi-hypothesis testing, it is FDR adjusted by Benjamini-Hochberg Procedure.


ftest_uncorr: (BH corrected, obsolete version: not BH corrected), all features are contained in final output matrix.
Similar to ftest_corr, but only returns 3 discrete levels for each feature and each class.
Three levels of values are returned: 1 if positive associated, 0 if not associated, -1 if negative associated.

ftest_mix: (BH corrected, obsolete version:not BH corrected)
Mixed levels are returned: continuous values if positive associated, 0 if not associated, negative continuous if negative associated.


Input parameters:
data: ndarray (binary)
labels: 1-d array for cluster number 0,1,2,3,...
features: 1-d array of features


restest: 
calculates the log odds ratios for highcost frequencies in each cluster.

additional input parameters:
response:1-d array of response 0,1



"""

import scipy.stats as stats
import numpy as np
import statsmodels.stats.multitest as smt


def ftest_corr(data,labels,features):
    data_label = np.append(data,labels,axis=1)
    # create log odds and pval table
    lor_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    pval_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    for i in range(data_label.shape[1]-1):
        for j in range(np.unique(labels).shape[0]):
            # feature i in cluster j  (a)
            feat_clustplus = np.count_nonzero(data_label[data_label[:,-1]==j][:,i]) + 1
            # feature i not in cluster j (b)
            feat_clustminus = data_label[data_label[:,-1]==j][:,i].shape[0] - feat_clustplus +1
       
            # feature i in cluster k != j (c)
            feat_noclustplus = np.count_nonzero(data_label[data_label[:,-1]!=j][:,i]) + 1
            # feature  i not in cluster k !=j (d)
            feat_noclustminus = data_label[data_label[:,-1]!=j][:,i].shape[0] - feat_noclustplus + 1
            # obtain pvalue frome fisher's exact
            __, pvalue = stats.fisher_exact([[feat_clustplus, feat_clustminus], [feat_noclustplus,feat_noclustminus]])
            pval_table[j,i] = pvalue
            # calculate log odds ratio = log(ad/bc)
            lor_table[j,i] = np.log(feat_clustplus+1)+np.log(feat_noclustminus+1)-np.log(feat_noclustplus+1)- np.log(feat_clustminus+1)
    # create pval_array for BH procedure        
    pval_array = pval_table.reshape(np.unique(labels).shape[0]*data.shape[1],)
    # apply hochberg benjamini correction at 0.99 confidence level
    test_array,__,__,__=smt.multipletests(pval_array, alpha=0.01, method='fdr_bh', is_sorted=False, returnsorted=False)
    test_table = (test_array*1).reshape(np.unique(labels).shape[0],data.shape[1])
    # select features that are significant
    sel_feat = features[test_table.sum(axis=0) !=0]
    sel_or_data = lor_table[:,np.in1d(features,sel_feat,assume_unique=True)]  
    # return selected features and log odds ratio which are significant
    return sel_feat, sel_or_data


def ftest_uncorr(data,labels,features):
    data_label = np.append(data,labels,axis=1)
    lor_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    feat_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    pval_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    for i in range(data_label.shape[1]-1):
        for j in range(np.unique(labels).shape[0]):
            feat_clustplus = np.count_nonzero(data_label[data_label[:,-1]==j][:,i]) +1
            feat_clustminus = data_label[data_label[:,-1]==j][:,i].shape[0] - feat_clustplus +1
            feat_noclustplus = np.count_nonzero(data_label[data_label[:,-1]!=j][:,i])+ 1
            feat_noclustminus = data_label[data_label[:,-1]!=j][:,i].shape[0] - feat_noclustplus+ 1
            __, pvalue = stats.fisher_exact([[feat_clustplus, feat_clustminus], [feat_noclustplus, feat_noclustminus]])
            pval_table[j,i] = pvalue
            lor_table[j,i] = np.log(feat_clustplus+1)+np.log(feat_noclustminus+1)-np.log(feat_noclustplus+1)- np.log(feat_clustminus+1)
    pval_array = pval_table.reshape(np.unique(labels).shape[0]*data.shape[1],)
    test_array,__,__,__=smt.multipletests(pval_array, alpha=0.01, method='fdr_bh', is_sorted=False, returnsorted=False)
    test_table = (test_array*1).reshape(np.unique(labels).shape[0],data.shape[1])
    # the above are similar to corr, the following select features significantly positively associated and set to 1;
    # negatively associated and set to -1, not significant set to 0. use feat_table for 1,0,-1 matrix.
    lor_table_pos,lor_table_neg = (lor_table > 0)*1 , (lor_table < 0)*(-1)
    lor_table_new = lor_table_pos + lor_table_neg
    feat_table = lor_table_new * test_table
    return feat_table,lor_table


def ftest_mix(data,labels,features):
    data_label = np.append(data,labels,axis=1)
    lor_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    feat_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    pval_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    for i in range(data_label.shape[1]-1):
        for j in range(np.unique(labels).shape[0]):
            feat_clustplus = np.count_nonzero(data_label[data_label[:,-1]==j][:,i]) +1
            feat_clustminus = data_label[data_label[:,-1]==j][:,i].shape[0] - feat_clustplus +1
            feat_noclustplus = np.count_nonzero(data_label[data_label[:,-1]!=j][:,i])+ 1
            feat_noclustminus = data_label[data_label[:,-1]!=j][:,i].shape[0] - feat_noclustplus+ 1
            __, pvalue = stats.fisher_exact([[feat_clustplus, feat_clustminus], [feat_noclustplus, feat_noclustminus]])
            pval_table[j,i] = pvalue
            lor_table[j,i] = np.log(feat_clustplus+1)+np.log(feat_noclustminus+1)-np.log(feat_noclustplus+1)- np.log(feat_clustminus+1)
    pval_array = pval_table.reshape(np.unique(labels).shape[0]*data.shape[1],)
    test_array,__,__,__=smt.multipletests(pval_array, alpha=0.01, method='fdr_bh', is_sorted=False, returnsorted=False)
    test_table = (test_array*1).reshape(np.unique(labels).shape[0],data.shape[1])
    # the above are similar to corr and uncorr, the following select features significantly associated whether positively or negatively . Set not significant to be 0. Use feat_table for plotting.  
    feat_table = lor_table * test_table 
    
    return feat_table,lor_table


def restest(labels, response):
    lab_res = np.append(labels,response,axis=1)
    or_table = np.zeros(np.unique(labels).shape[0])
    for j in range(np.unique(labels).shape[0]):
        res_clustplus = np.count_nonzero(lab_res[lab_res[:,0]==j][:,1])+1
        res_clustminus = lab_res[lab_res[:,0]==j][:,1].shape[0] - res_clustplus+1 
        res_noclustplus = np.count_nonzero(lab_res[lab_res[:,0]!=j][:,1])+1
        res_noclustminus = lab_res[lab_res[:,0]!=j][:,1].shape[0] - res_noclustplus+1
        oddsratio, __ = stats.fisher_exact([[res_clustplus, res_clustminus], [res_noclustplus, res_noclustminus]])
        #print(oddsratio)
        or_table[j] = oddsratio
    return np.log(or_table)



'''
obsolete. use updated.

def ftest_uncorr(data,labels,features):
    data_label = np.append(data,labels,axis=1)
    #odds_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    lor_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    #pval_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    feat_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    #feat_test = np.array([])
    for i in range(data_label.shape[1]-1):
        for j in range(np.unique(labels).shape[0]):
            feat_clustplus = np.count_nonzero(data_label[data_label[:,-1]==j][:,i]) +1
            feat_clustminus = data_label[data_label[:,-1]==j][:,i].shape[0] - feat_clustplus +1
            #feat_test = np.append(feat_test,feat_clustminus)
            
            feat_noclustplus = np.count_nonzero(data_label[data_label[:,-1]!=j][:,i])+ 1
            feat_noclustminus = data_label[data_label[:,-1]!=j][:,i].shape[0] - feat_noclustplus+ 1
            oddsratio, pvalue = stats.fisher_exact([[feat_clustplus, feat_clustminus], [feat_noclustplus, feat_noclustminus]])
            #pval_table[j,i] = pvalue
            lor_table[j,i] = np.log(oddsratio)
            if pvalue < 0.01:
                if lor_table[j,i] > 0:
                    feat_table[j,i] = 1
                else:
                    feat_table[j,i] = -1
            
            #odds_table[j,i] = feat_clustplus/feat_clustminus
#     pval_array = pval_table.reshape(np.unique(labels).shape[0]*data.shape[1],)
#     #test_array,__,__,__=smt.multipletests(pval_array, alpha=0.01, method='fdr_bh', is_sorted=False, returnsorted=False)
#     test_table = (test_array*1).reshape(np.unique(labels).shape[0],data.shape[1])
#             #if feat_clustminus ==0:
#             #    odds_table[j,i] = feat_clustplus/(feat_clustminus+1e-2)
#             #else:
#             #    odds_table[j,i] = feat_clustplus/feat_clustminus

#     sel_feat = features[test_table.sum(axis=0) !=0]
#     sel_or_data = or_table[:,np.in1d(features,sel_feat,assume_unique=True)]  
#     sel_odds_data = odds_table[:,np.in1d(features,sel_feat,assume_unique=True)]
    return feat_table,lor_table

def ftest_mix(data,labels,features):
    data_label = np.append(data,labels,axis=1)
    #odds_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    lor_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    #pval_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    feat_table = np.zeros((np.unique(labels).shape[0],data.shape[1]))
    #feat_test = np.array([])
    for i in range(data_label.shape[1]-1):
        for j in range(np.unique(labels).shape[0]):
            feat_clustplus = np.count_nonzero(data_label[data_label[:,-1]==j][:,i]) +1
            feat_clustminus = data_label[data_label[:,-1]==j][:,i].shape[0] - feat_clustplus +1
            #feat_test = np.append(feat_test,feat_clustminus)
            
            feat_noclustplus = np.count_nonzero(data_label[data_label[:,-1]!=j][:,i])+ 1
            feat_noclustminus = data_label[data_label[:,-1]!=j][:,i].shape[0] - feat_noclustplus+ 1
            oddsratio, pvalue = stats.fisher_exact([[feat_clustplus, feat_clustminus], [feat_noclustplus, feat_noclustminus]])
            #pval_table[j,i] = pvalue
            lor_table[j,i] = np.log(oddsratio)
            if pvalue < 0.01:
                # if lor_table[j,i] > 0:
                 #   feat_table[j,i] = 1
                #else:
                feat_table[j,i] = lor_table[j,i]
            
            #odds_table[j,i] = feat_clustplus/feat_clustminus
#     pval_array = pval_table.reshape(np.unique(labels).shape[0]*data.shape[1],)
#     #test_array,__,__,__=smt.multipletests(pval_array, alpha=0.01, method='fdr_bh', is_sorted=False, returnsorted=False)
#     test_table = (test_array*1).reshape(np.unique(labels).shape[0],data.shape[1])
#             #if feat_clustminus ==0:
#             #    odds_table[j,i] = feat_clustplus/(feat_clustminus+1e-2)
#             #else:
#             #    odds_table[j,i] = feat_clustplus/feat_clustminus

#     sel_feat = features[test_table.sum(axis=0) !=0]
#     sel_or_data = or_table[:,np.in1d(features,sel_feat,assume_unique=True)]  
#     sel_odds_data = odds_table[:,np.in1d(features,sel_feat,assume_unique=True)]
    return feat_table,lor_table

'''    