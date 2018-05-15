# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:35:01 2018

@author: nsde
"""

#%%
from keras.layers.core import Layer

#%%
class SpatialAffineTransformer(Layer):
    def __init__(self, localization_net, output_size):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialAffineTransformer, self).__init__()
    
    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None, int(output_size[0]), int(output_size[1]), int(input_shape[-1]))
    
    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
        self.constraints = self.locnet.constraints
        
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Affine_transformer(theta, X, self.output_size)
        return output
