# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:01:13 2018

@author: nsde
"""

#%% Packages
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from ddtn.transformers.transformer_layers import get_transformer_dim
from ddtn.transformers.transformer_layers import get_transformer_init_weights

#%%
def get_loc_net(input_shape, transformer_name = 'affine'):
    """ Example on how a localization layer can look like """
    # Get dimension for the last layer
    dim = get_transformer_dim(transformer_name)
    
    # Get weights for identity transformer. Note 50=#unit in second last layer
    weights = get_transformer_init_weights(50, transformer_name)
    locnet = Sequential()    
    locnet.add(Convolution2D(20, (3,3), activation='tanh', input_shape=input_shape))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Convolution2D(20, (3,3), activation='tanh'))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Flatten())
    locnet.add(Dense(50, activation='tanh'))
    locnet.add(Dense(dim, activation='linear', weights=weights))
    return locnet    
    
#%%
if __name__ == "__main__":
    locnet = get_loc_net((28, 28), transformer_name='affine')