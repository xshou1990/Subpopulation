import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


def plotclustmap(means,variance,featureslice,clustpop,normtype):
    # means 2darray should be row: number of features, column: number of clusters
    # variance 2darray should be row: number of features by column: number of clusters
    # featureslice: 1d array of features (dimension match the rows of means and variance)
    # normtype: None
    # clustpop : sample size for each clusters ,should be 1d array (dimension match the number of columns for means and variance)
    clustname = ['Upstate Low BW','Non-NYC Healthy','Non-NYC Travelers','HV+LI Healthy','NYC Healthy','NYC Mild','Upstate Severe']
    #clustname = np.arange(10)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue","gainsboro","red"])
    if normtype == None:
        cg = sns.clustermap(means,yticklabels = featureslice, xticklabels = clustname,standard_scale=None,col_cluster=False, figsize =(16,16),cmap=cmap) 
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
        cg.ax_row_dendrogram.set_visible(False)
        cg.cax.set_visible(False)
        
    return 