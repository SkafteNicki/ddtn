# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:24:41 2018

@author: nsde
"""

#%% Packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer, MaxPool2D

from ddtn.transformers.construct_localization_net import get_loc_net
from ddtn.transformers.keras_layers import SpatialAffineLayer
from ddtn.transformers.keras_layers import SpatialAffineDiffioLayer
from ddtn.transformers.keras_layers import SpatialHomografyLayer
from ddtn.transformers.keras_layers import SpatialCPABLayer
from ddtn.transformers.keras_layers import SpatialTPSLayer

import argparse

#%% Argument parser for comman line input
def argument_parser():
    parser = argparse.ArgumentParser(description='''This program will train a 
                                     neural network mnist dataset.''')
    parser.add_argument('-lr', action="store", dest="learning_rate", type=float, default = 0.0001,
                        help = '''Learning rate for optimizer. Default: 1e-4''')
    parser.add_argument('-ne', action="store", dest="num_epochs", type=int, default = 10,
                        help = '''Number of epochs. Default: 10''')
    parser.add_argument('-bs', action="store", dest="batch_size", type=int, default = 100,
                        help = '''Batch size: Default: 100''')
    parser.add_argument('-tt', action="store", dest="transformer_type", type=str, default='no',
                        help = '''Transformer type to use. Choose between: no, 
                                  affine, cpab or affine_cpab. Default: no''')
    res = parser.parse_args()
    args = vars(res)
    
    # Print input
    print(50*'-')
    print('Running script with arguments:')
    print('   learning rate:            ', args['learning_rate'])
    print('   number of epochs:         ', args['num_epochs'])
    print('   batch size:               ', args['batch_size'])
    print('   transformer type:         ', args['transformer_type'])
    print(50*'-')
    
    return args

#%%
if __name__ == '__main__':
    args = argument_parser()
    
    input_shape=()
    
    # Construct localization network
    loc_net = get_loc_net(input_shape=,
                          transformer_name=args['transformer_type'])
    
    # Construct keras model
    model = Sequential()
    model.add(in_layer)
    model.add(loc_net)
    model.add(Conv2D())
    model.add(Conv2D())
    model.add(MaxPool2D())
    model.add(Conv2D())
    model.add(Conv2D())
    
    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Fit model
    model.fit(X_train, y_train,
              batch_size=args['batch_size'],
              epochs=args['num_epochs'],
              verbose=1,
              validation_data=(X_test, y_test))
    
    # Calculate final score
    score = model.evaluate(X_test, y_test, verbose=0)
