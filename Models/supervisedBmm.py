#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:14:37 2019
Updated on Sat Jan 25 10:02:27 2020
@author: george & Xiao 
"""

import sys 

sys.path.append('..')
#sys.path.append('../SGMM')
sys.path.append('../metrics')
sys.path.append('../loaders')
sys.path.append('../oldCode')
sys.path.append('../visual')
sys.path.append('../testingCodes')
sys.path.append('../otherModels')


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  SGDClassifier
from sklearn.model_selection import GridSearchCV
from  scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
#from cvxopt import matrix
#from cvxopt.solvers import qp
from sklearn.linear_model import LogisticRegression
import random as rdm


class SupervisedBMM():
    
    """ 
        THIS CLASS IMPLEMENTS THE SUPERVISED BMM ALGORITHM 
        IT USES SOME PARTS OF SCIKIT LEARN TO ACCOMPLISH THIS    
    """
    
    
    def __init__(self, max_iter = 1000, cv = 10, mix = 0.2, 
                 C = [20,50,200,500,2000,5000], 
                 alpha = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ],
                 max_iter2 = 10, penalty = 'l1', scoring = 'neg_log_loss',
                 solver = 'saga', n_clusters = 2, tol = 10**(-3 ) , 
                 mcov = 'diag', tol2 = 10**(-3), transduction = 0, adaR = 1,
                 verbose = 0, warm = 0, m_sparse = 0, m_sparseL = 10**(-3),
                 m_sp_it1 = 2, m_sp_it2 = 2, m_choice = 0, 
                 m_LR = 0.001, m_mix = 1, altern = 0, log_reg = 'LG'):
        
        
        
        """ MODEL  PARAMETERES:
            
            max_iter:[INT] #Number of epochs of SGD default 1000
            cv:[INT] Cross Validation: Default 5 Fold
            mix:{FLOAT] In what Percentages to Upadate Memberships in respect with
            the previous iteration: Def --> 0.2
            C: [LIST] Inverse of Regularization Parameter: DEF: 1000
           
            alpha:[LIST] regularization parameters, for the stochastic 
            gradient descend cross validation
            max_iter2: Maximum # of EM Iterations, DEF: 10 
            penalty:[STRING] Regularization type ( Default L1 )
            scoring:[STRING] score to optimize in cross validation: DEF: 
                'negative_log_loss
            solver: [STRING] DEF: 'saga', Solvers used by scikit learn
            for logistic Regression 
            n_clusters:{INTEGER] #of Soft Clusters: DEF: 2
            tol:[FLOAT] memberships convergence tolerance
            tol2 =[FLOAT] stochastic gradient descent tolerance: Def 10^(-3)
            mcov =[STRING] 'full' or 'diag', 'full' means full covariance,
                   'diag' means diagonal covariance
                   
            transduction:[BINARY] 
            If to use transductional logic or not: Default: 1(TR)
            adaR:[BINARY] Adaptive regularization , regularize according to the
            cluster size
            verbose:[BINARY] information on the fitting of the algorithm and 
            other information
            warm:[BINARY], if this is 1 we need to give during the fitting the
            memberships, warm start, given memberships
            m_sparse:{BINARY] take sparse means in the gaussians centers or not
            m_sparseL: [FLOAT] the lambda for the means regularization
            m_sp_it1: iterations to run the first sparse means algorith if 
            chosen
            m_sp_it2: iterations to run the second sparse algorithm if chosen
            m_choice:[BINARY] choice of sparse algorithm QP or Gardient Descend
            m_LR:  if the choice if Gradient descend  pick the learning rate
            m_mix: 
            altern: [BINARY] start using  prediction driven approach when
                            memberships have converged with just mixture models
            
            log_reg: [STRING], "SGD" stochastic gradient descend,
                               "LG" Logistic Regression
            
                
    
        """
        ######################################################################
        # CLASS ATTRIBUTES
        ######################################################################
        #ind1 and ind2 are optional feature selection parameters that might
        #be specified in the fit method
        self._ind1 = None
        self._ind2 = None
        #idx1, idx2 are indexes created if we do a split and explained in
        #the split method
        self._idx1 = None 
        self._idx2 = None
        
        #maximum number of epochs for SGD
        self._max_iter = max_iter
        #Number of Folds for Cross Validation
        self._cv = cv
        #Mixing Coefficient
        self._mix = mix
        #NOT USED ANY MORE
        self._C = C
        #List with regularization parameters for cross validation
        self._alpha = alpha
        #Nuber of iterations of the EM algorithm
        self._max_iter2 =  max_iter2
        #type of penalty for logistic regression
        self._penalty = penalty
        #type of scoring  for cross validation
        self._scoring = scoring
        #type of scikit learn solver for SGD
        self._solver = solver
        #nimber of clusters to use 
        self._n_clusters = n_clusters
        #tolerance for the SGD agorithm
        self._tol = tol
        #tolerance for the membership convergence
        self._tol2 = tol2
        #variable for the type pf covariance for the gaussians, Bernoullis' only have diagonal
        self._mcov = mcov
        #use transuction or not
        self._trans = transduction
        #use adaptive regularization of not
        self._adaR = adaR
        #verbose or not
        self._vb = verbose
        #warm : warm start or not
        self._warm = warm
        
        self._m_sparse = m_sparse
        self._m_sparseL = m_sparseL
        self._m_sp_it1 = m_sp_it1
        self._m_sp_it2 = m_sp_it2
        self._m_choice = m_choice 
        self._m_LR =  m_LR
        self._m_mix = m_mix
        self._altern = altern
        self._log_reg = log_reg
        
        #FOR FIT INITIALIZE WITH KMEANS THE MEMBERSHIPS
        self._KMeans = None
        
        ######################################################################
        
        #THE FOLLOWING ATTRIBUTES ARE SETTED AFTER FITTING THE ALGORITHM
        #TO DATA
        
        #PARAMETER TO BE SETTED AFTER THE MODELS IS FITTED
       
        #A list of the Gaussians After fitting the data
         
        #when we fit a mixture of bernulis the means of the bernullis           
        self.Bers = None 
        #mixture coefficients of Bernullis (to do )
        self.mixesB = None
        
        #A list of Logistic regression predictors one for each class
        self.LogRegr = None 
        
        #PARAMETERS OF THE GAUSSIANS MODELS
        #list of covariances matrixes, list of means, list of mixes
        #probability matrix "Gauss membershipfor train and test if test iused
        #Gmms list of Gaussian predictors from Scipy Class
        self.params = None  
        
        #PARAMETERS AFTER FITTING THE MODEL
        #fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
        #             'labTrain' : trainlabels }
        #memberships for training data and testing data
        #hard cluster labels for training and testing data
        self.fitParams = None 
                             
                             
       
        #GAUSSIAN'S MEANS AND COVARINACES, Bernoullis only have means
        self.means = None 
        self.cov = None
        #LOGISTIC REGRESSION WEIGHTS
        self.weights = None
        
        #TRAIN AND TEST MEMBERSHIPS SOFT
        self.mTrain = None
        self.mTest = None
        
        #IF MODEL IS FITTED OR NOT
        self.fitted = None
        
        ######################################################################
        #TO DO 
        # INCORPORATE BERNULLIS AND GAUSSIANS TOGETHER IN THE SAME FIT 
        # FUNCTION
        #GIVE THE BINARY DATA COLUMNS AND USE THESE FOR BERNULLIS AND THE
        #REST WITH GAUSSIANS
        #######################################################################
    #HELPER   
    def split(self, data = None, X = None, y = None, split = 0.2):
        """
        A helper function to split data into training and test set
        There are 2 choices, either Input a data numpy Array with the last 
        column be its labels or  The data and the labels separately
        data: Data with last column labels
        X: Data
        y: labels
        split: the percentage of test data
        
        returns: Xtrain, Xtest, ytrain, ytest, idx1, idx2
        idx1:  indexes taken for training data
        idx2:  indexes taken for test data
        
        
        """
        
       # if (data and X and y ) is None:
          #  return "Input data or X and y "
        
        if (X is None) and (y is None):
            Xtrain, Xtest, ytrain, ytest , idx1, idx2  = \
                                train_test_split(data[:,:-1], data[:,-1], 
                                 np.arange( data.shape[0] ), 
                                 test_size = split, random_state = 1512,
                                 stratify = y)
        else:
            Xtrain, Xtest, ytrain, ytest, idx1, idx2 = \
                                train_test_split(X, y, 
                                 np.arange( X.shape[0] ), 
                                 test_size = split, random_state = 1512,
                                 stratify = y)
        self.idx1 = idx1
        self.idx2 = idx2
        
        return Xtrain, Xtest, ytrain.astype(int), ytest.astype(int)
        
        
    def initializeMemb( self, warm, kmeans, dimXtrain, n_clusters,
                       regk, trans, simple, dimXtest, Xtrain, Xtest, mTrain1,
                       mTest1):
        
        """ Function to initialize memberships,
        warm: if we want a warm start ((provide mTrain1, mTest1))
        kmeans: [binary] kmeans initialization or not
        dimXtrain: number of training data
        n_clusters: number of clusters we use
        regk: amount of regularization for the divisions
        trans: use transduction or not (if yes we need test data too )
        simple: prediction driven or not
        if we have trunsduction give the dimension of test data
        Xtrain: training data
        Xtest: testing data
        mTrain1: given thatwe want a warm start give the initial memeberhsips
        mTest1: given that we want a warm start give the initial memeberships
        of test data
        """
        
        
        
        if warm == 0: #IF WE DONT HAVE WARM START
            
            if kmeans == 0: #NO KMEANS INITIALIZATION (RANDOM INIT)
               mTrain = np.random.rand( dimXtrain, n_clusters) + regk
               
               if trans == 1:
                   mTest = np.random.rand( dimXtest, n_clusters )  + regk
                   
               else:
                   mTest = []
            
            else: #KMEANS INITIALIZATION
                
                km = KMeans( n_clusters = n_clusters, random_state = 0)
                
                if simple == 0:
                    rdm.seed(0)
                    randsamp_train = rdm.sample(range(Xtrain.shape[0]),np.int(Xtrain.shape[0]*0.5))
                
                    if trans == 1:
                        #FIT KMEANS IN DATA (TEST AND TRAIN IF TRANSDUCTION)
                        randsamp_test = rdm.sample(range(Xtest.shape[0]),np.int(Xtest.shape[0]*0.5))
                        Xtot = np.concatenate( (Xtrain[randsamp_train,:] , 
                                                Xtest[randsamp_test,:]), axis = 0)
                        km = km.fit( Xtot )
                        mAll = np.zeros([ dimXtrain +dimXtest, n_clusters])
                        labels = km.predict (np.concatenate( (Xtrain , Xtest), axis = 0))
                    else:
                        #FIT ONLY TRAIN IF NOT TRANSDUCTION, with 50% data
                        km.fit( Xtrain [randsamp_train,:])
                        mAll = np.zeros([ dimXtrain , n_clusters])
                        labels = km.predict ( Xtrain ) 
                else:
                    if trans == 1:
                        #FIT KMEANS IN DATA (TEST AND TRAIN IF TRANSDUCTION)
                        km = km.fit( np.concatenate( (Xtrain, Xtest), axis = 0))
                        mAll = np.zeros([ dimXtrain +dimXtest, n_clusters])
                    
                    else:
                        #FIT ONLY TRAIN IF NOT TRANSDUCTION
                        km.fit( Xtrain)
                        mAll = np.zeros([ dimXtrain , n_clusters])
                
                    #TAKE THE LABELS FROM KMEANS
                    labels = km.labels_ 
                for j in np.arange( labels.shape[0] ): #MAKE THE MEMBERSHIPS
                    mAll[j, labels[j]] = 1
                    
                mTrain = mAll[0: dimXtrain ]
                if trans == 1:
                    mTest = mAll[ dimXtrain :]
                    
                else:
                    mTest = []

        else: #IF WE HAVE WARM START, ASSIGN WITH THE GIVEN MEMBERSHIPS

            mTrain = mTrain1
            mTest = mTest1
        
        return mTrain, mTest
    
    
################### FITTING LOGISTIC REGRESSION MODEL #########################     
    
    def fitLogisticRegression(self, n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist,
                              param_grid, lg_regr, C,  hard_cluster):
        
        """ FIT LOGISTIC REGRESSION FOR EACH CLUSTER 
            n_clusters: number of gaussians -- clusters
            mTrain: train data membership,
            adaR: to use or not adaptive regularization
            alpha: regularization parameteres list
            max_iter : number of epochs to train the stochastic gradient
            descend algorithm
            tol2: tolerance of SGD training
            Xtrain: training data
            ytrain: training labels
            vb: to print some info at eout related to adaptive regularization
            such us cluster size, new alphas etc
            penalty: penalty to use for training , default L1 norm
            scoring: scoring to use for training , Default neg log loss
            cv: number of folds for cross validation
            regk: regularization when computing log probabilities
            ind1: indexes to use for training (feature columns)
            indexing: a list with the indexes of the training data
            logiProb: an initialized matrix to put the logistic regression
            probabilities
            logLogist: an initialized matrix to put the log probabilities
            lg_regr: Choice of SGD or FULL Logistic Regression
            C: regularization for logistic regression
            hard_cluster: hard_cluster memebrships before the fit of
                        logistic regressions
            
            returns: models-> logistic regresion models
                     logiProb--> probabilities of a data point to belong in 
                     in its class given the cluster
                     logLogist--> the same as above but log probabilities
                     
           
            """
            
        mTrain = self.hardCluster( mTrain.copy(), hard_cluster)
        models = []
        for clust in np.arange( n_clusters ): #FITLOG REGR
                #FIT THE L1 LOGISTIC REGRESSION MODEL
                #CROSS VALIDATION MAXIMIZING BE DEFAULT THE NEGATIVE LOG LIKEHOOD
                
                #ADAPTIVE REGULARIZATION
                Nclus = np.sum( mTrain[:, clust], axis = 0 )
                if adaR == 1:
                    if lg_regr == 'SGD':
                        alphanew = (np.array( alpha ) / Nclus).tolist()
                        param_grid = {'alpha': alphanew}
                    else:
                        
                        Cnew = (np.array( C ) / Nclus ).tolist()
                        param_grid = {'C': Cnew}
                # PRINT SOME INFORMATION  
                if vb == 1:
                    #print Cluster Size
                    print('\n Cluster {} has Size {} of {}'.format( clust,
                          Nclus, mTrain.shape[0]))
                    if adaR == 1:
                        if lg_regr == 'SGD':
                            print('alpha is {} alphaNew {}'.format(alpha, alphanew))
                        else:
                            print('C is {} CNew {}'.format(C, Cnew))
       
                #TRAIN LOGISTIC REGRESSION MODEL
                if lg_regr == 'SGD':
                    mf = SGDClassifier(loss = "log", penalty = penalty, 
                                      n_jobs = -1, max_iter = max_iter,
                                      random_state = 0, tol = tol2)
                else:
                
                    mf = LogisticRegression( penalty = penalty, tol = tol2,
                                             random_state = 0, 
                                        max_iter = max_iter, n_jobs = -1)
                 
                model = GridSearchCV( mf, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain,
                                      sample_weight = mTrain[:, clust] ) 
    
                #FOR EACH CLUSTER APPEND THE MODEL in MODELS
                models.append( model )  
               
                #PREDICT PROBABILITIES FOR BEING IN CLASS 1 or 0
                proba = model.predict_proba( Xtrain[:, ind1] )
                #log probabilities
                logproba = np.log( proba + regk)
                
                #FOR EACH DATA POINT TAKE THE PROB ON BEING IN CORRECT CLASS
                logiProb[:, clust]  = proba[ indexing, ytrain ] 
                logLogist[:, clust] = logproba[ indexing, ytrain]
                
                ######## END OF CODE FITTING LOGISTIIC REGRESSION ############
        return models, logiProb, logLogist
   
    
################ COMPUTE LIKELIHOOD ##########################################    
    def computeLikelihood( self, gmmLogprob, logLogist, dimXtrain, vb, trans,
                                                                simple, Qold):
        
           """COMPUTER THE AUXILARY FUNCTION Q IN EACH ITERATION 
           gmmLogprob: The log probabilities for all clusters from Mixture
           of Gaussians
           logLogist:  Log probabilities from logistic regressin
           dimXtrain: Train Data Dimension
           vb:  verbose output,
           trans: if trunsduction is used or not
           simple: if we use the MoG or the SGMM
           Qold: the previous calculated Q value
           """
           dimXtest = gmmLogprob.shape[0] - dimXtrain
           if trans == 0:
               Qf = gmmLogprob + logLogist*(1-simple)
               Qf2 = np.log( np.sum(np.exp( Qf ), axis = 1) )
               Qf3 = np.sum( Qf2 )/dimXtrain
                      
           else:
               Qft = gmmLogprob[0: dimXtrain,:] + logLogist*(1-simple)
               Qf2 = np.log( np.sum(np.exp( Qft ), axis = 1) )
               Qf31 = np.sum( Qf2 )           
                    
               Qftest =  gmmLogprob[dimXtrain:, :] 
               Qftest2 =  np.log( np.sum(np.exp( Qftest ), axis = 1) )
               Qftest3 = np.sum( Qftest2 ) 
                    
               Qf3 = (Qftest3 + Qf31)/(dimXtest + dimXtrain)
           
           Qdif = abs( Qf3 - Qold)
           if vb == 1:     
               print("\n Qnew is : {}".format( Qf3 ))
              # print("\n Qdif is : {}".format( Qdif ))
           
           
           return Qf3, Qdif

              
    
################ SPARSE MEANS FUNCTIONS #######################################
    ############  UNDER CONSTRUCTION    #######################################
    def objective(self, X, Nk, meank, mean, cinv, lambd, covk):
        
        t1 = Nk*0.5*np.linalg.det(cinv)
        t2 = 0
        for i in np.arange( X.shape[0] ):
            t2 += -0.5*(np.expand_dims(X[i,:]-meank, axis = 0))@\
                                cinv@np.expand_dims((X[i,:]-meank), axis = 1)
        
        t3 = -lambd*np.linalg.norm( mean - meank, ord = 1)
        obj = t1+t2+t3
        return obj
            
        
    def m_GD(self, X, memb, meank, covk, Nk, N, lambd, sp_it2, LR, reg, mcov):
        #Gradient Descend algorithm
        mean = np.sum( X, axis = 0)/N
        print( mean.shape )
       # print(mean)
        
        cinv = np.linalg.pinv( covk )
        for i in np.arange( sp_it2 ): #GRADIENT DESCEND LOOP
            #cinv = np.linalg.pinv( covk )
            # obj = self.objective(X, Nk, meank, mean, cinv, lambd, covk)
             #print( obj )
             mTerm1 = np.sum( (memb*(X-meank).T).T, axis = 0)
           # mTerm1 = np.expand_dims( mTerm1, axis = 1)
            #print(mTerm1.shape)
             mnew = meank + LR*( cinv@mTerm1-lambd*( -np.sign( mean - meank)))
            
          #  cTerm2 = -0.5*self.cCov(X, mnew, memb, reg, Nk, mcov)
           # Snew = covk + LR*( 0.5*Nk*covk + cTerm2 )
            
             meank = mnew
          #  covk = Snew 
        covk = self.cCov(X, mnew, memb, reg, Nk, mcov)
        
        return meank, covk
    
    def spMeans(self, X, memb, covk, Nk, N, lambd ):
        """ Calculates the Sparse means by optimizing the l1 norm 
        
        X: data Matrix
        memb: membership for Gaussian k
        covk: covariance matrix
        Nk: data in cluster k
        N: data
        lambd: regularization
        """
        #Number of Features
        Nf = X.shape[1]
        
        #data mean
        mean = np.expand_dims( np.sum(X, axis = 0)/N, axis = 1)
        
        #inverse covariance
        cinv = np.linalg.pinv( covk )
        
        #Form P matrix [I*Nk 0]
        zeros = np.zeros(shape = [Nf, Nf])
        onesD =  cinv*Nk 
        first = np.concatenate((onesD, zeros), axis = 1)
        second = np.concatenate(( zeros, zeros ), axis = 1)
        P = np.concatenate(( first, second), axis = 0)
        Po = matrix( P )
        
        #form q [coef 1^T *lambda]
        print(memb.shape, X.shape)
        wsum = np.expand_dims( np.sum( (memb*X.T).T, axis = 0), axis = 1 )
        fq = ( -wsum.T@cinv).T
        sq = np.ones(shape = [Nf, 1] )*lambd
        q = np.concatenate( (fq, sq), axis = 0)
        qo = matrix( q )
        
        #form G
        eye = np.eye(Nf)
        firstG = np.concatenate( ( eye, -eye ), axis = 1)
        secondG = np.concatenate(( -eye, -eye), axis = 1)
        thirdG = np.concatenate( (zeros, -eye), axis = 1)
        G = np.concatenate( (firstG, secondG, thirdG), axis = 0)
        Go = matrix( G )
        
        #forming matrix h
        zerosh = np.zeros(shape = [Nf, 1] )
        h = np.concatenate(( mean, -mean, zerosh))
        ho = matrix( h )
        
        slv = qp(Po, qo, G = Go, h = ho)
        
        meank = np.array( slv['x'] )
        meank = np.squeeze( meank[0:Nf], axis = 1)
        
        print(meank)
        
        return meank
    
######### END OF SPARSE MEANS FUNCTIONS #######################################        
                            
    


################# #FITTING THE Bernoulli MIXTURE MODEL #########################  
##################   BERNULLI  STARTS UNDER HERE    #############################
    def fitB(self, Xtrain = None, ytrain = None, Xtest = None, ind1 = None,
                    ind2 = None, mTrain1 = None, mTest1 = None, 
                    kmeans = 0, mod = 1, simple = 0, comp_Lik = 0,
                    memb_mix = 0.0, hard_cluster = 0):
        """ 
            Fit the Supervised Mixtures of Bernoulli Model
            
            ind1: chose the features to use in the training of the Ml model
            ind2: chose the fetures to use in the training of the Gaussians
            Xtrain: training data
            ytrain: labels of training data
            Xtest: testing data if tranduction is on
            kmeans: kmeans initialization  of memberships
            mod: mode of computing the probabilities for gaussians, default
            mod = 1
            simple : binary variable to decide if you will use simple 
            mixture of gaussians plus classification [simple = 1], if 
            simple is 0 [simple = 0] then use prediction driven gaussians
            for training ( the proposed model )
            a third choice is use simple  = 0  and set the altern variable from
            the model to 1 this will use no prediction driven results till one 
            point and then  it will alter to prediction driven
            comp_Lik: (UNSTABLE) Compute Likelihood  or not 
            memb_mix: parameter on how to mix the supervised along with the 
                      memberships
            
            hard_cluster: hard cluster memberships before the logistic regression
                          fitting.
            
            
        """
        #CHECK IF ALL DATA ARE GIVEN
        self.ind1 = ind1
        self.ind2 = ind2
        self.fitted = 1
        self._KMeans = kmeans
        
        if Xtrain is None or ytrain is None  :
            print(" Please Give Xtrain, ytrain, Xtest  data ")
            return
        
        if ind1 is None:
            #ALL FEATURES FOR PREDICTION IF ind1 NOT SPECIFIED
            self.ind1 = np.arange( Xtrain.shape[1] )
            ind1 = self.ind1
            
        if ind2 is None:
            #ALL FEATURES FOR CLUSTERING IF ind2 NOOT SPECIFIED
            self.ind2 = np.arange( Xtrain.shape[1] )
            ind2 = self.ind2
        
        #PARAMETERES TO BE USED BY THE FIT FUNCTION
        n_clusters = self._n_clusters
        max_iter = self._max_iter
        cv = self._cv
        mix = self._mix
        penalty = self._penalty
        scoring = self._scoring
        #solver = self._solver
        max_iter2 = self._max_iter2
        trans = self._trans
        C = self._C
        alpha = self._alpha
        tol = self._tol
        tol2 = self._tol2
        adaR = self._adaR
        vb = self._vb
        warm = self._warm
        altern = self._altern
        lg_regr = self._log_reg
        
        dimXtrain = Xtrain.shape[0]
        dimXtest = 0
        
        if trans == 1:
            dimXtest = Xtest.shape[0]
            
        #regularize the sums  for numerical instabilities
        reg = 10**(-5)
        #regularization to be added to every memebership entry
        regk = reg/n_clusters
        
        
        
        
        #INITIALIZE MEMBERSHIP FUNCTIONS
        #WE KEEP SEPARATED TRAIN AND TEST MEMBERSHIPS
        #BECAUSE TRAIN IS SUPERVISED MEMBERSHIP
        #TEST IS UNSUPERVISED
        
        mTrain, mTest = self.initializeMemb(warm, kmeans, dimXtrain, n_clusters,
                       regk, trans, simple, dimXtest, Xtrain, Xtest, mTrain1,
                       mTest1)
        

        #NORMALIZE MEMBERSHIPS SO EACH ROW SUMS TO 1
        sumTrain = np.sum( mTrain, axis = 1) 
        mTrain = ( mTrain.T / sumTrain ).T
        
        if trans == 1:
            sumTest = np.sum( mTest, axis = 1 )
            mTest = ( mTest.T / sumTest ).T    
        
       
        
        #SET SOME  PARAMETERES FOR USE IN THE FOR LOOP OF EM
        indexing = np.arange( dimXtrain )
        
        #MATRIX WITH LOGISTIC REGRESSION PROBABILITIES FOR EACH CLASS
        logiProb = np.zeros([dimXtrain, n_clusters])
        
        #MATRIX WITH LOG PROBABILITIES
        logLogist = np.zeros([dimXtrain, n_clusters])
       
        #variable related  to the altern parametere
        gate = 0
        ###START FITTING ALGORITHM ##################
       
        #setting the cross validation grid
        if lg_regr == 'SGD':
            param_grid = {'alpha': alpha}
        else:
            param_grid = {'C':  C}
            
        Qold = 0 #initial likelihood (if we are to calculate it)
        
        #STARTING EM ALGORITHM
        for iter2 in np.arange( max_iter2 ): #OUTER EM ITERATIONS
            #FITING THE L1 LOGISTIC REGRESSIONS
            if simple == 0:
                models, logiProb, logLogist = self.fitLogisticRegression( 
                              n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist, 
                              param_grid,  lg_regr, C,
                                              hard_cluster = hard_cluster )
                
            else: #IF WE USE SIMPLE MIXTURES OF GAUSSIANS JUST FIT AT LAST ITER
                if iter2 == ( max_iter2 - 1):
                    models, logiProb, logLogist = self.fitLogisticRegression( 
                              n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist, 
                              param_grid, lg_regr, C,  hard_cluster = hard_cluster )
                    
            #WE TAKE THE MEMBERSHIPS AND ALL THE DATA
            #TO FIT THE GAUSSIANS USING THE EM ALGORITHM FOR GMM 
            
            if trans == 1: #if we hve transduction
                data = np.concatenate((Xtrain[:, ind2], Xtest[:, ind2]), 
                                                                      axis = 0)
                mAll = np.concatenate( (mTrain, mTest ), axis = 0 )
            
            else:
                 data =  Xtrain[:, ind2]
                 mAll = mTrain
               
            #take the parameters of the GMM models
            #THIS PIECE OF CODE WILL BE REMOVED IN THE FUTURE
            params = self.berModels( data, mAll )
#            print('test here')
#            print(params)
            #bmmProb = params['probMat']
            ###################################################################
            #THIS IS AFTER MODIFICATIONS#######################################
            if mod == 1: #THIS IS THE MOD WE WILL KEEP IN THE FUTURE
                bmmProb = params['probMat2']
                bmmLogprob = params['logProb']  
                self.mixesB = params['pis']
                self.params = params 
            #END OF MODIFICATION ##############################################
            
           #CALCULATE NEW MEMBERSHIPS FOR TRAIN AND TEST
            if simple and gate == 0: #NO PREDICTION DRIVEN (MoG + LR + L1)
                mNewTrain =  bmmProb[0: dimXtrain, :] + regk
                
            else: #PREDICTION DRIVEN (SGMM)
                mNewTrain = logiProb * bmmProb[0: dimXtrain, :] + regk
                
                simple = 0
            
            ###################################################################
            if trans:
                mNewTest = bmmProb[dimXtrain :, :] + regk
             
            #COMPUTE LIKELIHOOD IF COMP_LIK == 1
            if mod  and comp_Lik:
                 Qnew, Qdif = self.computeLikelihood( bmmLogprob, logLogist,
                                    dimXtrain, vb, trans, simple, Qold)
                 Qold = Qnew
            #END OF MODIFICATION ##############################################
            
            #NORMALIZE NEWMEMBERSHIPS
            sumTrain = np.sum( mNewTrain, axis = 1)
            if trans == 1:
                sumTest = np.sum( mNewTest, axis = 1 )
          
            mNewTrain = ( mNewTrain.T / sumTrain ).T
            if trans == 1:
                mNewTest = ( mNewTest.T / sumTest ).T  
                
                
            #EVALUATE ERROR
            errorTr = np.sum( np.abs( mTrain - mNewTrain) )
            if trans == 1:
                errorTst = np.sum( np.abs( mTest - mNewTest ) )
                error = ( errorTr + errorTst )/( (dimXtrain + dimXtest)\
                                                             *n_clusters )
            else:
                error = errorTr/( dimXtrain * n_clusters )
            
            if (error < 5*10**(-8)) and altern:
                gate = 1
                altern = 0
           

            #MAKE A SOFT CHANGE IN MEMEBRSHIPS MIXING OLD WITH NEW 
            #MEMBERSHIPS WITH DEFAULT MIXING OF 0.5
            mNewTrain = mNewTrain*(1-memb_mix) + \
                                            self.predict_BMMS(Xtrain)*memb_mix
                                            
            mTrain = mNewTrain*(1-mix) + mTrain*(mix)
            if trans == 1:
                mTest = mNewTest*(1-mix) + mTest*(mix)
        
            
            print("BMM iteration: {}, error: {}".format(iter2, error))
            if error < tol:
                
                 break
        ############ END OF EM UPDATES #######################################
       #if simple  and error < tol:
        models, logiProb, logLogist = self.fitLogisticRegression( 
                              n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist, 
                              param_grid, lg_regr, C,
                                                  hard_cluster = hard_cluster )

        #self.mixesB = params['pis']
        self.LogRegr = models
      
        #TAKING HARD CLUSTERS IN CASE WE WANT TO USE LATER  
        if trans == 1:
            testlabels = np.argmax( mTest, axis = 1 )
            
        else:
            testlabels = []
            
        trainlabels = np.argmax( mTrain, axis = 1 )
        fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
                     'labTrain' : trainlabels }
        
        self.mTrain = mTrain
        
        if trans == 1:
            self.mTest = mTest
            
        self.fitParams = fitParams
        #set the weights of LOGREG MEANS AND COVARIANCES OF GAUSSIANS
        self.setWeights()
        
        self.setBernoulli( params )
        
        return self

        
       
    def berModels(self, X, members):
            
            """
                Calculates the Mixtures of Bernullis Parameters
                
                
                X : Train and Test data together
                members: Posterior Probabibilities for each cluster
                             and each data point (memberships)
                             
                Returns: a list with the mean matrices of the Bernullis,
                a list with the mixing parameteres,
                
                the probability matrix with the posteriors for each data
                point and each cluster,
                
                All these it returns in the form of a dictionary
                
            """
                
            clusters = members.shape[1]
            regk = (10**(-5)/clusters)
           
            means = [] #list of means Probabilities of clusteres
            pis = [] #list of mixing coefficients
            probMat = np.zeros( [X.shape[0], clusters] )
           
            logprobaMatrix = np.zeros([X.shape[0], clusters])
                    
            for cl in np.arange( clusters ):
               
                # FOR EACH CLUSTER USE THE EM ALGORITHM
                # TO CREATE THE NEW MEMBERSHIP MATRIX OF THE GAUSSIANS
                #IT IS NOT EXACTLY THE MEMBERSHIP BECAUSE IT IS
                # NORMALIZED  AFTER THIS FUNCTION ENDS
                mCl, piCl, logproba = self.calcBerPar( X, 
                                                       members[:,cl])
                   
                logprobaMatrix[:,cl] = logproba 
                
               
                means.append( mCl )
                pis.append( piCl )
                
            
            
            #FOR EACH DATA POINT FIND THE MAXIMUM LOGPROBABILITY 
            #THIS IS DONE FOR REGULARIZATION PURPOSES
            maxLog = np.max( logprobaMatrix, axis = 1 )
            logprobaMatrix2 = ( logprobaMatrix.T - maxLog).T

            
            #### NEXT 4 LINES BEFORE  WILL BE DELETED IN FUTURE
            probMat = np.exp( logprobaMatrix2 ) + regk
            sumRel = np.sum( probMat, axis = 1)
            probMat = (probMat.T / sumRel).T
            probMat = probMat*np.array(pis)
 
            #THIS WILL BE KEPT IN THE FUTURE -->p(x/z_i)p(z_i)
            probMat2 = np.exp( logprobaMatrix2 )*np.array( pis ) + regk
            totLog = logprobaMatrix + np.log( np.array( pis ) )
            
            params = {'means': means, 'pis' : pis, 
                          'probMat':probMat,  'probMat2': probMat2,
                          'logProb': totLog}
            
            return params
      
           
    def calcBerPar(self, X, memb):
        #CALCULATES PARAMETERS FOR EACH Bernulli product
        #FOR EACH CLUSTER
        #RETURNS:
       
        #meank : mean vector of Bernulli of class k
        #pk: mixing coefficient of Bernulli of class k
        
        #proba: the posterior probabilities, i.e probabilities of being
        #in class k given X 
        
        Nk = np.sum(memb) 
        N = X.shape[0] 
        #mixing coefficient
        pk = Nk/N  
        #print("Minimum of memb {} max{}".format(np.min( memb), np.max( memb)))
        meank = np.sum( ( X.T * memb ).T, axis = 0) / Nk +10**(-6)
       
        meankOne = 1-meank  + 10**(-6)
        meanklog = np.log(meank)
        meankOnelog = np.log(meankOne)
        
        #full covarinace
        logProbTerm1 = np.sum( X * meanklog, axis = 1 )
        logProbTerm2 = np.sum( (1-X) * meankOnelog, axis = 1)
        
        
        logproba = logProbTerm1 + logProbTerm2 
       #print( logproba )
      
       # print("min log prob {}, max of log prob {}".format( np.min(proba ),
       #np.max(proba )))
        #proba = model.cdf(X)*pk
        return  meank, pk, logproba    

    
###  PREDICTIONS  #############################################################      
    def setBernoulli( self, params ):
        #SETTING MEANS AND COVARIANCES OF THE GAUSSIANS
        if self.fitted == None:
            print("MODEL IS NOT FITTED YET")
            
        self.means = params['means']
        self.pis = params['pis']
        return        
    
    def predict_BMMS( self, X):
        """
        Given a Data matrix X it returns the Membership matrix 
        for each data point in X based on the Bernoulli's already fitted
        
        """
        
        if self.fitted == 0:
            print("Warning: There is no fitted model ")
            return []

        
        nclusters = self._n_clusters 
        mixes = self.params['pis']
        meanB = np.zeros( [X.shape[1], nclusters] )
        for i in range( len(self.params['means']) ):
            meanB[:,i] = self.params['means'][i] 
        regk = 10**(-5)/nclusters
        meanBOne = 1 - meanB + regk
        meanBlog = np.log( meanB + regk )
        meanBOnelog = np.log( meanBOne )

        
        membership = np.zeros( [X.shape[0], nclusters] )
        logmembership = np.zeros( [X.shape[0], nclusters] )
        for i in np.arange( nclusters ):
            
            logProbTerm1 = np.sum( X[:, self.ind2] * meanBlog[:,i], axis = 1 )
            logProbTerm2 = np.sum( (1-X[:, self.ind2]) * meanBOnelog[:,i], axis = 1)
        
        
            logmembership[:,i] = logProbTerm1 + logProbTerm2 
            
        maxlog = np.max( logmembership, axis = 1)
        logmembership = (logmembership.T - maxlog).T
        probMat = np.exp( logmembership )* np.array( mixes ) + regk
        sumRel = np.sum( probMat, axis = 1)
        membership = (probMat.T / sumRel).T 
       
        return membership
    
    def predict_prob_int_B(self, Xtest = None, Xtrain = None):  
        
        """
          AFTER FITTING THE MODEL, PREDICTS THE PROBABILITIES OF TRAIN AND TEST
          DATA TO BE 1, USING THE MEMBERSHIPS THAT HAVE BEEN CALCULATED DURING
          TRAINING
           
        """
        #CHECKING IF THERE IS TRANSUCTION
        trans = self._trans
        if trans == 1:
            if self.mTest is None:
                print("The Model is not fitted or some other error might have\
                              occured")
                return
        
        logisticModels = self.LogRegr  #TAKE LOGISTIC REGRESSION MODELS
       
        if trans == 1:
            pMatrixTest = np.zeros( (Xtest.shape[0]) )
            
        pMatrixTrain = np.zeros( (Xtrain.shape[0]) )
        #FOR EACH MODEL CALCULATE THE PREDICTION FOR EACH DATA POINT
        for i, model in enumerate( logisticModels ):
            #probability each test point
            #to be in class 1
            if trans == 1:
                probsTest = model.predict_proba( Xtest )[:,1] 
                pMatrixTest += probsTest*self.mTest[:, i]
               
            #probability each  training point
            #to be in class 1                                       
            probsTrain = model.predict_proba(Xtrain)[:,1] 
            pMatrixTrain += probsTrain*self.mTrain[:, i]
            
        if trans == 0:   
            pMatrixTest = self.predict_proba_B(Xtest)   
        
        return pMatrixTest, pMatrixTrain
    
    def predict_proba_B(self, X = None):
        "Predicts the Probabity of  data X to be in class 1"""
        
        models = self.LogRegr
        memb = self.predict_BMMS( X )   #PREDICT MEMBERSHIP OF X  
        totalProb = np.zeros( [X.shape[0]])
        for i in np.arange( memb.shape[1] ):
            #probability  points of X belong in class 1
            model = models[i]
            probModel = model.predict_proba( X )
            proba = probModel[:, 1]
           # totalProb += models[i].predict_proba( X )[:, 1]*memb[:, i]
            totalProb += proba*memb[:, i]
        
        return totalProb 
            
    def getweightsL1(self, models ):
        """GIVEN THE LOGISTIC REGRESSION MODELS,
        RETURN THE SUM OF THE WEIGHTS PLUS REGULARIZATION """
        sumW = 0
        
        for i, model in enumerate( models ):
            weights = model.best_estimator_.coef_.tolist()[0]
            alphDict = model.best_params_
            alph = alphDict['alpha']
            weights = np.array( weights )
            weights = np.abs( weights )
            sumW += alph*np.sum(weights)
            
        return -sumW
         
    def setWeights( self ):
        """ setting logistic regression weights for each cluster """
        if self.fitted == None:
            print("MODEL IS NOT FITTED YET")
            
        models = self.LogRegr
        
        weights = []
        for model in models:
            weight = model.best_estimator_.coef_.tolist()[0]
            intercept = model.best_estimator_.intercept_[0]
            weight.insert(0, intercept)
            weights.append( weight )
        
        self.weights = weights
        
        return
    
        
    def hardCluster( self, mTrain, hard_cluster):
        """takes the memeberships assigns 1 at the max element of each row
         and 0 to all the other elements of the row
        """
        if hard_cluster:
            mTrain2 = np.zeros_like( mTrain )
            mTrain2[ np.arange(len(mTrain)), np.argmax( mTrain, axis = 1)] = 1
            return mTrain2
        
        return mTrain