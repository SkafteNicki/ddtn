#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:46:10 2018

@author: nsde
"""

#%%
import tensorflow as tf
from tensorflow.python.keras import backend as K

#%%
class KerasTrainingLogger(tf.keras.callbacks.Callback):
    """
    
    """
    def __init__(self, trans_layer=1):
        super(KerasTrainingLogger, self).__init__()
        self.trans_layer = trans_layer
        self.validation = None
        self.trans_func = None
        
    def on_train_begin(self, logs=None):
        self.validation = self.params['do_validation']
        self.trans_func = K.function([self.model.inputs], [self.model.layers[self.trans_layer].output])
        
    def on_batch_end(self, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
        
#%%
if __name__ == '__main__':
    own_callback = KerasTrainingLogger()        
