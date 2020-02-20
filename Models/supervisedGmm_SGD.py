#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:09:17 2020

@author: xiaoshou
"""

## supervised binary clustering: simultaneuiys detection subpopulation (cadres) with respect to a binary target.
## parameters have both l1 and l2 regularization, and l1 terms are handled with
## proximal gradient steps
## input is a pandas.DataFrame containing both target and input features
## loss function is the upper bound of negative likelihood function 
## - \frac{1}{N} \sum_{i=1}^{N}  \sum_{k=1}^{K} \pi_{k} \{ \text{log} ~  N (x_{i}| \mu_{k},Sigma^{k} ) + 
## y_{i} \text{log} ~  \sigma(w_{k}^{T}x_{i}) + (1-y_{i}) \text{log} ~ [1-\sigma(w_{k}^{T}x_{i})] \} + \lambda ||w_{k}||


from __future__ import division, print_function, absolute_import

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


class supervisedGmm_SGD(object):
    
    def __init__(self, M=2, lambda_W=0.001,alpha_W=0.99, Tmax=10000, record=100, 
                 eta=2e-3, Nba=128, eps=1e-2, termination_metric='loss'):
        ## hyperparameters / structure
        self.M = M                # number of cadres
        self.lambda_W = lambda_W  # regularization strength  
        self.alpha_W = alpha_W   # regularization proportionality alpha*||W||_1 
        self.fitted = False
        ## optimization settings
        self.Tmax = Tmax     # maximum iterations
        self.record = record # record points
        self.eta = eta       # initial stepsize
        self.Nba = Nba       # minibatch size
        self.eps = eps       # convergence tolerance 
        self.termination_metric = termination_metric
        ## parameters
        self.gamma = 0 # reparam for pi, pi_k = exp(gamma_k)/ sum_j (exp(gamma_j))
        self.pi = 0 # mixing proportions
        self.W = 0     # regression weights
        self.w0 = 0    # regression biases
        self.Mu = 0     # Gaussian centers
        self.Beta = 0 # Gaussian diagnoval covariances: beta^2

        ## data
        self.data = None       # pd.DataFrame containing features and response
        self.cadreFts = None   # pd.Index of column-names giving features used for cadre assignment
        self.predictFts = None # pd.Index of column-names giving features used for target-prediction
        self.targetCol = None  # string column-name of response variable
        ## outputs
        self.metrics = {'training': {'loss': []},
                        'validation': {'loss': []}}
        self.time = [] # times
        self.termination_reason = None # why training stopped
    
        
    def fit(self, data, targetCol, cadreFts=None, predictFts=None, dataVa=None, inits=None, 
            seed=16162, store=False, progress=False, posMu = None  ):
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
        
        # cadre weights/ mixing proportions
        if inits is not None and 'gamma' in inits:
            gamma = tf.Variable(inits['gamma'], dtype=tf.float32, name='gamma')
        else:
            gamma = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='gamma')  
        
        ## cadre centers parameter
        if inits is not None and 'Mu' in inits:
            Mu = tf.Variable(inits['Mu'], dtype=tf.float32, name='Mu')
        else:
            if posMu == 1:
                Mu = tf.Variable(np.random.beta(a = 2.0, b=2.0, size=(Pcadre,self.M)), 
                            dtype=tf.float32, name='Mu')
            else:
                Mu = tf.Variable(np.random.normal(loc=0.5, scale=0.1, size=(Pcadre,self.M)), 
                            dtype=tf.float32, name='Mu')


        ## regression hyperplane weights parameter
        if inits is not None and 'W' in inits:
            W = tf.Variable(inits['W'], dtype=tf.float32, name='W')
        else:
            W = tf.Variable(np.random.normal(loc=0., scale=0.1, size=(Ppredict,self.M)), 
                            dtype=tf.float32, name='W')
            
        ## regression hyperplane bias parameter
        if inits is not None and 'w0' in inits:
            w0 = tf.Variable(inits['w0'], dtype=tf.float32, name='w0')
        else:
            w0 = tf.Variable(tf.zeros(shape=(self.M,), dtype=tf.float32), 
                             dtype=tf.float32, name='w0')
        
        ## cadre diagonal covariance
        if inits is not None and 'Beta' in inits:
            Beta = tf.Variable(inits['Beta'], dtype=tf.float32, name='Beta')
        else:
            Beta = tf.Variable(np.random.normal(loc= 0, scale= 1, size=(Pcadre,self.M)), 
                            dtype=tf.float32, name='Beta')

            
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,Pcadre), name='Xcadre')
        rep_Xcadre = tf.keras.backend.repeat_elements(Xcadre, self.M, axis=0) 
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,Ppredict), name='Xpredict')
        Y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='Y')
        rep_Y = tf.keras.backend.repeat_elements(Y, self.M, axis=1)
        eta = tf.placeholder(dtype=tf.float32, shape=(), name='eta')
        lambda_Ws = tf.placeholder(dtype=tf.float32, shape=(self.M,), name='lambda_Ws')
        
###################### Membership ########################################   
        pi =  tf.divide(tf.exp(gamma),tf.reduce_sum(tf.exp(gamma)), name = 'pi') 
        
        mvn = tfp.distributions.MultivariateNormalDiag(loc= tf.tile(tf.transpose(Mu),[tf.shape(Xcadre)[0],1]),
                        scale_diag = tf.tile(tf.transpose(tf.exp(Beta)),[tf.shape(Xcadre)[0],1]))
        
        # log N(x|mu_k,Sigma_k)
        LogN_ik = tf.reshape(mvn.log_prob(rep_Xcadre),(tf.shape(Xcadre)[0],self.M), name = 'LogN_ik') 
        
        # max log N(x|mu_k,Sigma_k), column-wise reduce max
        MaxlogN_ik = tf.expand_dims(tf.reduce_max(LogN_ik, axis =1),axis=1, name ='MaxlogN_ik')
        
        # log N(x|mu_k,Sigma_k) - max log N(x|mu_k,Sigma_k)
        Norm_LogN_ik = tf.subtract(LogN_ik, tf.tile(MaxlogN_ik, 
                                                tf.constant([1,LogN_ik.get_shape()[1].value])),  name = 'Norm_LogN_ik') 
        # N(x|mu_k,Sigma_k) 
        N_ik = tf.exp( Norm_LogN_ik , name = 'N_ik')
        
        # f(x,z) = pi * N_ik
        joint_pdf = tf.multiply(pi, N_ik, name = 'joint_pdf')  + 1e-9
        # f(x,z) --> normalized pmf
        Mem_pmf = tf.divide(joint_pdf, tf.tile(tf.expand_dims(tf.reduce_sum(joint_pdf,axis = 1),axis=1), 
                                                tf.constant([1,joint_pdf.get_shape()[1].value])),  name = 'mem_pdf') 
        # best cluster (cadre)
        bstCd = tf.argmax(Mem_pmf, axis=1, name='bestCadre')

###################### Emission ########################################        
        # XW+w0
        Z_ik = tf.add(tf.matmul(Xpredict, W), w0, name='Z_ik')
        # 1/(1+e^(-Z_ik))
        S_ik = tf.divide(1.0,1.0 + tf.exp(-Z_ik), name = 'S_ik')
        # log p(y_i | x_i,z_k)
        Loglogiprob = tf.add(tf.multiply(tf.log(S_ik + 1e-9),rep_Y), tf.multiply(tf.log(1.0-S_ik + 1e-9), 1 - rep_Y), name = 'Loglogiprob')
                 
###################### Objective: NLL ########################################  
        
        ## regularization
        l2_W = self.lambda_W * (1 - self.alpha_W) * tf.reduce_sum(lambda_Ws * W**2)
        l1_W = self.lambda_W * self.alpha_W * tf.reduce_sum(lambda_Ws * tf.abs(W))
        
        
        ## loss, full losss including l1 terms handled with proximal gradient
        loss = tf.divide( - tf.reduce_sum(tf.multiply(pi, tf.add(Norm_LogN_ik,Loglogiprob))) , tf.dtypes.cast(tf.shape(Xcadre)[0], dtype = tf.float32), name ='loss')
        loss_opt = loss + l2_W
        loss_full = loss_opt + l1_W
        optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_opt)
        
        ## projection Mu to positive orthant 
        proj_Mu = tf.assign(Mu, tf.maximum(Mu, 0) )
        
        ## nonsmooth proximal terms 
        thresh_W = tf.assign(W, tf.sign(W) * (tf.abs(W) - eta * self.lambda_W * lambda_Ws * self.alpha_W) * tf.cast(tf.abs(W) > eta * self.lambda_W * self.alpha_W, tf.float32))
        
        ####################
        ## learning model ##
        ####################
        with tf.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()

                    
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
                ## project Mu to positive orthant
                if posMu == 1:
                    sess.run([proj_Mu])    
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
                
            self.Mu, self.gamma, self.W, self.w0, self.Beta = Mu.eval(), gamma.eval(), W.eval(), w0.eval(), Beta.eval()
            
            ## clean up output for easier analysis
            self.metrics['training'] = pd.DataFrame(self.metrics['training'])
            if dataVa is not None:
                self.metrics['validation'] = pd.DataFrame(self.metrics['validation'])

            
        return self
    
    def predictFull(self, Dnew):
        """Returns classification scores/margins,cadre membership scores, predicted cadres """
        if not self.fitted: print('warning: model not yet fit')
        
        tf.reset_default_graph()
        gamma  = tf.Variable(self.gamma, dtype=tf.float32, name='gamma')
        Mu  = tf.Variable(self.Mu, dtype=tf.float32, name='Mu')
        Beta  = tf.Variable(self.Beta, dtype=tf.float32, name='Beta')
        W  = tf.Variable(self.W, dtype=tf.float32, name='W')
        w0 = tf.Variable(self.w0, dtype=tf.float32, name='w0')
        
        Xcadre = tf.placeholder(dtype=tf.float32, shape=(None,self.cadreFts.shape[0]), name='Xcadre')
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,self.predictFts.shape[0]), name='Xpredict')        
        rep_Xcadre = tf.keras.backend.repeat_elements(Xcadre, self.M, axis=0) 
        
        # preparing parameters 
        pi =  tf.divide(tf.exp(gamma),tf.reduce_sum(tf.exp(gamma)), name = 'pi') 
        
        mvn = tfp.distributions.MultivariateNormalDiag(loc= tf.tile(tf.transpose(Mu),[tf.shape(Xcadre)[0],1]),
                        scale_diag = tf.tile(tf.transpose(tf.exp(Beta)),[tf.shape(Xcadre)[0],1]))
           
        LogN_ik = tf.reshape(mvn.log_prob(rep_Xcadre),(tf.shape(Xcadre)[0],self.M), name = 'N_ik') 
        
        MaxlogN_ik = tf.expand_dims(tf.reduce_max(LogN_ik, axis =1),axis=1, name ='MaxlogN_ik')
        
        Norm_LogN_ik = tf.subtract(LogN_ik, tf.tile(MaxlogN_ik, 
                                                tf.constant([1,LogN_ik.get_shape()[1].value])),  name = 'Norm_LogN_ik') 
        
        N_ik = tf.exp( Norm_LogN_ik , name = 'N_ik')
        
        joint_pdf = tf.multiply(pi, N_ik, name = 'joint_pdf')  + 1e-9
        
        Mem_pmf = tf.divide(joint_pdf, tf.tile(tf.expand_dims(tf.reduce_sum(joint_pdf,axis = 1),axis=1), 
                                                tf.constant([1,joint_pdf.get_shape()[1].value])),  name = 'mem_pdf') 
        
        Z_ik = tf.add(tf.matmul(Xpredict, W), w0, name='Z_ik')
        
        S_ik = tf.divide(1.0,1.0 + tf.exp(-Z_ik), name = 'S_ik')
                                     
        ## F[n] = f_k(x^n): membership weighted logistic regression probability score
        F = tf.reduce_sum(Mem_pmf * S_ik, name='F', axis=1, keepdims=True)
        
        ## predicted cluster
        bstCd = tf.argmax(Mem_pmf, axis=1, name='bestCadre')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            F, Mem_pmf,hard_m = sess.run([F, Mem_pmf,bstCd], 
                                                     feed_dict={Xcadre: Dnew.loc[:,self.cadreFts].values,
                                                                Xpredict: Dnew.loc[:,self.predictFts].values})

        return F, Mem_pmf, hard_m
    
    def predictClust(self, Mem_pmf, Dnew):
         
        if not self.fitted: print('warning: model not yet fit')
        
        tf.reset_default_graph()
        W  = tf.Variable(self.W, dtype=tf.float32, name='W')
        w0 = tf.Variable(self.w0, dtype=tf.float32, name='w0')
        
        Xpredict = tf.placeholder(dtype=tf.float32, shape=(None,self.predictFts.shape[0]), name='Xpredict') 
        
        Z_ik = tf.add(tf.matmul(Xpredict, W), w0, name='Z_ik')
        S_ik = tf.divide(1.0,1.0 + tf.exp(-Z_ik), name = 'S_ik')
        
        mem_max = tf.constant((Mem_pmf == Mem_pmf.max(axis=1)[:,None]), dtype =tf.float32, name = 'mem_max')
        
        F_clust = tf.multiply(mem_max, S_ik, name = 'F_clust')
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            F_clust = sess.run([F_clust], feed_dict={Xpredict: Dnew.loc[:,self.predictFts].values})
            
        return F_clust[0]
        
        
        
        