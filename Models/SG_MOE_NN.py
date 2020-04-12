#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:18:58 2020

@author: xiaoshou
"""

## Prediction Driven SPARSELY-GATED MIXTURE-OF-EXPERTS Neural Network 


from __future__ import division, print_function, absolute_import

import time
import numpy as np
#import pandas as pd
import tensorflow as tf


class binaryCadreModel(object):
    
    def __init__(self, M=2, lambda_W=0.001,lambda_Theta = 0.01, topK= 2,
                 alpha_W=0.9, Tmax=10000, record=100, 
                 eta=4e-2, Nba=64, eps=1e-5, termination_metric='loss'):
        ## hyperparameters / structure
        self.M = M                # number of cadres
        self.topK = topK          # softkmax ,should be between 1 and M
        self.lambda_W = lambda_W  # regularization weights on predictor
        self.alpha_W = alpha_W    # elastic net weights
        self.lambda_Theta = lambda_Theta  # weights on gating network
        self.fitted = False
        ## optimization settings
        self.Tmax = Tmax     # maximum iterations
        self.record = record # record points
        self.eta = eta       # initial stepsize
        self.Nba = Nba       # minibatch size
        self.eps = eps       # convergence tolerance 
        self.termination_metric = termination_metric
        ## parameters
        self.Theta = 0 # Weights of first layer
        self.Theta0 = 0 # Bias of first layer
        self.W = 0     # regression weights
        self.W0 = 0    # regression biases
        ## data
        self.data = None       # pd.DataFrame containing features and response
        self.cadreFts = None   # pd.Index of column-names giving features used for cadre assignment
        self.predictFts = None # pd.Index of column-names giving features used for target-prediction
        self.targetCol = None  # string column-name of response variable
        ## outputs
        self.metrics = {'training': {'loss': [],
                                     'accuracy': [],
                                     'ROC_AUC': [],
                                     'PR_AUC': []},
                        'validation': {'loss': [],
                                      'accuracy': [],
                                      'ROC_AUC': [],
                                      'PR_AUC': []}}
        self.time = [] # times
        self.proportions = [] # cadre membership proportions during training
        self.termination_reason = None # why training stopped
    
    def get_params(self, deep=True):
        return {'M': self.M, 
                'lambda_W': self.lambda_W,  'lambda_Theta':self.lambda_Theta,
                'alpha_W': self.alpha_W, 'Tmax': self.Tmax, 'record': self.record, 
                'eta': self.eta, 'Nba': self.Nba, 'eps': self.eps}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, data, targetCol, cadreFts=None, predictFts=None, dataVa=None, inits=None, 
            seed=16162, store=False, progress=False):
        np.random.seed(seed)
        """Fits binary classification cadre model"""
        ## store categories of column names
        self.targetCol = targetCol
        if cadreFts is not None:
            self.cadreFts = cadreFts
        else:
            self.cadreFts = data.drop(targetCol, axis=1).columns
        if predictFts is not None:
            self.predictFts = predictFts
        else:
            self.predictFts = data.drop(targetCol, axis=1).columns
        ## get dataset attributes
        self.fitted = True
        if store:
            self.data = data
        Pcadre, Ppredict, Ntr = self.cadreFts.shape[0], self.predictFts.shape[0], data.shape[0]
            
        ## split data into separate numpy arrays for faster access
        ## features for cadre-assignment
        dataCadre = data.loc[:,self.cadreFts].values
        ## features for target-prediction
        dataPredict = data.loc[:,self.predictFts].values
        ## target feature
        dataTarget = data.loc[:,[self.targetCol]].values

        
        if dataVa is not None:
            dataCadreVa = dataVa.loc[:,self.cadreFts].values
            dataPredictVa = dataVa.loc[:,self.predictFts].values
            dataTargetVa = dataVa.loc[:,[self.targetCol]].values

                
        ############################################
        ## tensorflow parameters and placeholders ##
        ############################################
        tf.reset_default_graph()
    
        ## cadre centers parameter
        # 1st layer weights
        if inits is not None and 'Theta' in inits:
            Theta = tf.Variable(inits['Theta'], dtype=tf.float32, name='Theta')
        else:
            Theta = tf.Variable(np.random.normal(loc=0., scale=1, size=(Ppredict,self.M)), 
                            dtype=tf.float32, name='Theta')
        
        if inits is not None and 'Theta0' in inits:
            Theta0 = tf.Variable(inits['Theta0'], dtype=tf.float32, name='Theta0')
        else:
            Theta0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='Theta0')
        ## regression hyperplane weights parameter
        if inits is not None and 'W' in inits:
            W = tf.Variable(inits['W'], dtype=tf.float32, name='W')
        else:
            W = tf.Variable(np.random.normal(loc=0., scale=1, size=(Ppredict,self.M)), 
                            dtype=tf.float32, name='W')
        ## regression hyperplane bias parameter
        if inits is not None and 'W0' in inits:
            W0 = tf.Variable(inits['W0'], dtype=tf.float32, name='W0')
        else:
            W0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='W0')
    
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,Pcadre), name='Xcadre')
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,Ppredict), name='Xpredict')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        eta = tf.placeholder(dtype=tf.float32, shape=(), name='eta')
        lambda_Ws = tf.placeholder(dtype=tf.float32, shape=(self.M,), name='lambda_Ws')
        
        # gating function:        
        if self.topK == 1:
            softk = tf.contrib.seq2seq.hardmax( tf.add(tf.matmul(Xcadre, Theta) , Theta0) ) #hardmax
        elif self.topK == self.M:
            softk = tf.nn.softmax(tf.add(tf.matmul(Xcadre, Theta), Theta0),axis=1) # softmax 
        else: # topkmax
            softmax = tf.nn.softmax(tf.add(tf.matmul(Xcadre, Theta), Theta0),axis=1)
            idx = tf.contrib.framework.argsort(softmax, direction='DESCENDING')  # sorted indices
            ranks = tf.contrib.framework.argsort(idx, direction='ASCENDING')  # ranks
            topk = ranks < self.topK
            softk_unnorm = tf.multiply(softmax,tf.cast(topk,dtype= tf.float32))
            softk = tf.divide(softk_unnorm,tf.tile(tf.expand_dims(tf.reduce_sum(softk_unnorm,axis = 1),axis=1), 
                                                tf.constant([1,softmax.get_shape()[1].value])))

        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(Xpredict, W), W0, name='E')
        ## component P(y=1 | x, m =k)
        Z = tf.divide(1.0,1.0 + tf.exp(-E), name = 'Z')
        ## probability P(y=1|x)
        F = tf.reduce_sum(softk * Z, name='F', axis=1, keepdims=True)
        ## hard membership
        bstCd = tf.argmax(softk, axis=1, name='bestCadre')
        
        ## cross entropy loss
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=F)
        loss = -tf.reduce_mean(tf.multiply(Y, tf.log(F)) + tf.multiply(1-Y,tf.log(1.0-F)),name = 'loss')
        
        ## regularization
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(lambda_Ws * W**2)
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(lambda_Ws * tf.abs(W))
        l2_Theta = self.lambda_Theta *  tf.reduce_sum(W**2)
        
        ## loss that is fed into optimizer
        loss_opt = loss + l2_W + l2_Theta 
        optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_opt)   
        
        ## full loss, including l1 terms handled with proximal gradient
        loss_full = loss_opt + l1_W

        ## nonsmooth proximal terms
        thresh_W = tf.assign(W, tf.sign(W) * (tf.abs(W) - eta * self.lambda_W * lambda_Ws * self.alpha_W) * tf.cast(tf.abs(W) > eta * self.lambda_W * self.alpha_W, tf.float32))
        
        ####################
        ## learning model ##
        ####################
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            if progress:
                if dataVa is not None:
                    print('numbers being printed:', 
                          'SGD iteration, training loss, validation loss, time')
                else:
                    print('numbers being printed:',
                          'SGD iteration, training loss, time')

            t0 = time.time()
            ## perform optimization
            for t in range(self.Tmax):
                inds = np.random.choice(Ntr, self.Nba, replace=False)
                ## calculate adaptive regularization parameter
                cadres = bstCd.eval(feed_dict={Xcadre: dataCadre[inds,:], Xpredict: dataPredict[inds,:]})
                cadre_counts = np.zeros(self.M)
                for m in range(self.M):
                    cadre_counts[m] = np.sum(cadres == m) + 1
                cadre_counts = cadre_counts.sum() / cadre_counts
                
                ## take SGD step
                sess.run(optimizer, feed_dict={Xcadre: dataCadre[inds,:],
                                               Xpredict: dataPredict[inds,:],
                                               Y: dataTarget[inds,:],
                                               lambda_Ws: cadre_counts,
                                               eta: self.eta / np.sqrt(t+1)})
                ## take proximal gradient step
                sess.run([thresh_W], feed_dict={eta: self.eta / np.sqrt(t+1), lambda_Ws: cadre_counts})
                # record-keeping        
                if not t % self.record:
                    if progress:
                        if len(self.time) and dataVa is not None:
                            print(t,
                                  self.metrics['training']['loss'][-1], 
                                  self.metrics['validation']['loss'][-1], 
                                  self.time[-1])
                        elif len(self.time):
                            print(t,
                                  self.metrics['training']['loss'][-1], 
                                  self.time[-1])
                        else:
                            print(t)
                    self.time.append(time.time() - t0)

                    l = sess.run([loss], 
                                                 feed_dict={Xcadre: dataCadre,
                                                            Xpredict: dataPredict,
                                                            lambda_Ws: cadre_counts,
                                                            Y: dataTarget})
                    
                    self.metrics['training']['loss'].append(l)

                    
                    if dataVa is not None:
                        cadres = bstCd.eval(feed_dict={Xcadre: dataCadre, Xpredict: dataPredict})
                        cadre_counts = np.zeros(self.M)
                        for m in range(self.M):
                            cadre_counts[m] = np.sum(cadres == m) + 1
                        cadre_counts = cadre_counts / cadre_counts.sum()
                        l = sess.run([loss_full], feed_dict={Xcadre: dataCadreVa,
                                                                        Xpredict: dataPredictVa,
                                                                        lambda_Ws: cadre_counts,
                                                                        Y: dataTargetVa})
                        self.metrics['validation']['loss'].append(l)
 
                    if dataVa is not None:
                        if len(self.time) > 1:
                            last_metric = self.metrics['validation'][self.termination_metric][-1]
                            second_last_metric = self.metrics['validation'][self.termination_metric][-2]

                            if abs(last_metric[0] - second_last_metric[0])/last_metric[0] < self.eps:
                                self.termination_reason = 'lack of sufficient decrease in validation ' + self.termination_metric
                                break
                    else:
                        if len(self.time) > 1:
                            last_metric = self.metrics['training'][self.termination_metric][-1]
                            second_last_metric = self.metrics['training'][self.termination_metric][-2]
                            if abs(last_metric[0] - second_last_metric[0])/last_metric[0] < self.eps:
                                self.termination_reason = 'lack of sufficient decrease in training ' + self.termination_metric
                                break
                                       
            if self.termination_reason == None:
                self.termination_reason = 'model took ' + str(self.Tmax) + ' SGD steps'
            if progress:
                print('training has terminated because: ' + str(self.termination_reason))
                
            self.Theta,self.Theta0, self.W, self.W0 = Theta.eval(), Theta0.eval(), W.eval(), W0.eval()
            
            ## clean up output for easier analysis
#            self.metrics['training'] = pd.DataFrame(self.metrics['training'])
#            if dataVa is not None:
#                self.metrics['validation'] = pd.DataFrame(self.metrics['validation'])

            
        return self
    
    def predictFull(self, Dnew):
        """Returns classification scores/margins, predicted labels, cadre membership scores, predicted cadres, and loss"""
        if not self.fitted: print('warning: model not yet fit')
        
        tf.compat.v1.reset_default_graph()
        Theta  = tf.Variable(self.Theta, dtype=tf.float32, name='Theta') 
        Theta0 = tf.Variable(self.Theta0, dtype=tf.float32, name='Theta0')
        W  = tf.Variable(self.W, dtype=tf.float32, name='W')
        W0 = tf.Variable(self.W0, dtype=tf.float32, name='W0')
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,self.cadreFts.shape[0]), name='Xcadre')
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,self.predictFts.shape[0]), name='Xpredict')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')

        # gating function:        
        if self.topK == 1:
            softk = tf.contrib.seq2seq.hardmax( tf.add(tf.matmul(Xcadre, Theta) , Theta0) ) #hardmax
        elif self.topK == self.M:
            softk = tf.nn.softmax(tf.add(tf.matmul(Xcadre, Theta), Theta0),axis=1) # softmax 
        else: # topkmax
            softmax = tf.nn.softmax(tf.add(tf.matmul(Xcadre, Theta), Theta0),axis=1)
            idx = tf.contrib.framework.argsort(softmax, direction='DESCENDING')  # sorted indices
            ranks = tf.contrib.framework.argsort(idx, direction='ASCENDING')  # ranks
            topk = ranks < self.topK
            softk_unnorm = tf.multiply(softmax,tf.cast(topk,dtype= tf.float32))
            softk = tf.divide(softk_unnorm,tf.tile(tf.expand_dims(tf.reduce_sum(softk_unnorm,axis = 1),axis=1), 
                                                tf.constant([1,softmax.get_shape()[1].value])))
        ## cadre-wise prediction scores
        E = tf.add(tf.matmul(Xpredict, W), W0, name='E')
        Z = tf.divide(1.0,1.0 + tf.exp(-E), name = 'Z')
        ## prob(y=1|x)
        #F = tf.reduce_sum(softmax * Z, name='F', axis=1, keepdims=True)
        F = tf.reduce_sum(softk * Z, name='F', axis=1, keepdims=True)

        #bstCd = tf.argmax(softmax, axis=1, name='bestCadre')
        bstCd = tf.argmax(softk, name='bestCadre',axis=1)
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            Fnew, Mnew, Lnew = sess.run([F,softk,bstCd ], 
                                                     feed_dict={Xcadre: Dnew.loc[:,self.cadreFts].values,
                                                                Xpredict: Dnew.loc[:,self.predictFts].values,
                                                                Y: Dnew.loc[:,[self.targetCol]]})
        return Fnew,Mnew,Lnew

    def predictClust(self, Membership, Dnew):
        """Returns classification scores for each cadre (hard cluster) """
         
        if not self.fitted: print('warning: model not yet fit')
        
        tf.compat.v1.reset_default_graph()
        
        W  = tf.Variable(self.W, dtype=tf.float32, name='W')
        W0 = tf.Variable(self.W0, dtype=tf.float32, name='W0')
        
        Xpredict = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,self.predictFts.shape[0]), name='Xpredict') 
        
        E = tf.add(tf.matmul(Xpredict, W), W0, name='E')
        Z = tf.divide(1.0,1.0 + tf.exp(-E), name = 'Z')
        
        mem_max = tf.constant((Membership == Membership.max(axis=1)[:,None]), dtype =tf.float32, name = 'mem_max')
        
        F_clust = tf.multiply(mem_max, Z, name = 'F_clust')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            F_clust = sess.run([F_clust], feed_dict={Xpredict: Dnew.loc[:,self.predictFts].values})
            
        return F_clust[0]   
