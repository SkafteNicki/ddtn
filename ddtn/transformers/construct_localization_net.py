# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:01:13 2018

@author: nsde
"""

#%% Packages
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPool2D

from ddtn.transformers.transformer_util import get_transformer_init_weights
from ddtn.transformers.transformer_util import get_transformer_dim

#%%
def get_loc_net(input_shape, transformer_name = 'affine'):
    """ Example on how a localization layer can look like """
    # Get dimension for the last layer
    dim = get_transformer_dim(transformer_name)
    
    # Get weights for identity transformer. Note 50=#unit in second last layer
    weights = get_transformer_init_weights(50, transformer_name)
    locnet = Sequential()    
    locnet.add(Conv2D(16, (3,3), activation='tanh', input_shape=input_shape))
    locnet.add(MaxPool2D(pool_size=(2,2)))
    locnet.add(Conv2D(32, (3,3), activation='tanh'))
    locnet.add(MaxPool2D(pool_size=(2,2)))
    locnet.add(Conv2D(32, (3,3), activation='tanh'))
    locnet.add(MaxPool2D(pool_size=(2,2)))
    locnet.add(Flatten())
    locnet.add(Dense(50, activation='tanh'))
    locnet.add(Dense(dim, activation='linear', weights=weights))
    return locnet    
    
#%%
if __name__ == "__main__":
    pass