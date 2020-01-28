## utility.py
## various utility functions used by all SCM functions

import numpy as np
import tensorflow as tf

from itertools import product
from scipy.special import xlogy

def eNet(alpha, lam, v):
    """Elastic-net regularization penalty"""
    return lam * (alpha * tf.reduce_sum(tf.abs(v)) + 
                  (1-alpha) * tf.reduce_sum(tf.square(v)))

def calcMargiProb(cadId, M):
    """Returns p(M=j) in vector form"""
    return np.array([np.sum(cadId == m) for m in range(M)]) / cadId.shape[0]

def calcJointProb(G, cadId, M):
    """Returns p(M=j, x in C_i) in matrix form"""
    jointProbMat = np.zeros((M,M)) # p(M=j, x in C_i)
    for i,j in product(range(M), range(M)):
        jointProbMat[i,j] = np.sum(G[cadId==i,j])
    jointProbMat /= G.shape[0]
    return jointProbMat
    
def calcCondiProb(jointProb, margProb):
    """Returns p(M = j | x in C_i)"""
    return np.divide(jointProb, margProb[:,None], out=np.zeros_like(jointProb), where=margProb[:,None]!=0)

def estEntropy(condProb):
    """Returns estimated entropy for each cadre"""
    return -np.sum(xlogy(condProb, condProb), axis=1) / np.log(2)

def entropy(G,M):
    """Returns estimated entropy for each cadre"""
    m = np.argmax( G, axis = 1 )
    marg = calcMargiProb(m, M)
    jont = calcJointProb(G, m, M)
    cond = calcCondiProb(jont, marg)
    return estEntropy(cond)


def oce_entropy(prob,weights,class_prior):
    """Off-Centered Entropy (OCE) (Lenca et al. 2010) uses a transformation function in order to convert the maximum uncertainty from an imbalanced to a balanced situation."""
    prob_scaled1 = prob/(2.0*class_prior) * (prob < class_prior) # p/2theta, if 0 <= p <= theta
    prob_scaled2 = (prob + 1.0 - 2*class_prior)/(2*(1-class_prior)) * (prob >= class_prior) # p+1-2theta/(2(1-theta) ) if theta <= p <= 1
    prob_scaled = prob_scaled1 + prob_scaled2
    cluster_entropy = - (prob_scaled*np.log2(prob_scaled)+ (1-prob_scaled)*np.log2(1-prob_scaled))
    weighted_entropy = np.sum(weights * cluster_entropy)/np.sum(weights)
    average_entropy = np.mean(cluster_entropy)
    return average_entropy, weighted_entropy, cluster_entropy
    
def asymm_entropy(prob,weights,class_prior):
    """Asymmetric Entropy (Marcellin et al. 2006) """
    cluster_entropy = prob * (1.0 - prob) /((1.0 - 2.0 * class_prior)*prob + class_prior**2)
    weighted_entropy = np.sum(weights * cluster_entropy)/np.sum(weights)
    average_entropy = np.mean(cluster_entropy)
    return average_entropy, weighted_entropy, cluster_entropy