#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:03:29 2020

@author: xiaoshou
"""

from tensorflow.keras.layers import Input, Dense,Lambda,Dropout
from tensorflow.keras.models import Model
import tensorflow.keras as keras
from numpy.random import seed
from tensorflow import set_random_seed



def supclustNN(Xtr,ytr,n_clusters,linear,epochs):
    seed(0)
    set_random_seed(0)
    
    kernel_init=keras.initializers.glorot_uniform(seed=0)
    # This returns a tensor
    inputs = Input(shape=(Xtr.shape[1],))
    
    # first neural network for binary predictions
    bin_output_1 = Dense(128, activation='relu',kernel_initializer= kernel_init)(inputs)
    bin_output_1 = Dropout(0.2,seed = 0)(bin_output_1)
    bin_output_2 = Dense(16, activation='relu',kernel_initializer= kernel_init)(bin_output_1)
    
    # second neural network for multi class clustering
    multi_output_1 = Dense(64, activation='relu',kernel_initializer= kernel_init)(inputs)
    multi_output_1 = Dropout(0.2,seed = 0)(multi_output_1)
    multi_output_2 = Dense(16, activation='relu',kernel_initializer= kernel_init)(multi_output_1)

    # with linear predictors, only hidden layer used, sigma(WX+B), weights are readily interpretable
    if linear == 1 :
        bin_predictions = Dense(n_clusters, activation='sigmoid',kernel_initializer= kernel_init)(inputs)
        multi_predictions = Dense(n_clusters, activation='softmax',kernel_initializer= kernel_init)(inputs)   
    else:
        bin_predictions = Dense(n_clusters, activation='sigmoid',kernel_initializer= kernel_init)(bin_output_2)
        multi_predictions = Dense(n_clusters, activation='softmax',kernel_initializer= kernel_init)(multi_output_2)
    
    # sum g(x) e(x) , weighted prediction of p(y|x)
    weighted_prob = keras.layers.Multiply() ([bin_predictions, multi_predictions ])
    summed_prob = Lambda(lambda x:keras.backend.sum(x, axis=1, keepdims=True))(weighted_prob)
    
    # This creates a model that includes the Input layer and three Dense layers of two weighted neural networks
    model = Model(inputs=inputs, outputs=summed_prob)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Xtr, ytr, epochs=epochs, batch_size=64, verbose=1, shuffle=False)
    
    # output for softmax layer from the first neural network
    multi_model = Model(model.input,  multi_predictions)    
    bin_model = Model(model.input,  bin_predictions)
    
    
    return model,multi_model,bin_model
