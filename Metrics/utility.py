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

def weighted_class_entropy(prob,weights):
    cluster_entropy = - (prob*np.log2(prob)+(1-prob)*np.log2(1-prob))
    weighted_entropy = np.sum(weights * cluster_entropy)/np.sum(weights)
    return weighted_entropy, cluster_entropy

    