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
    Arguments
        data:
        
        log_dir:
            
        trans_layer: integer determining where the transformation layer is in
        the model
    """
    def __init__(self, data, log_dir='./logs', trans_layer=1):
        super(KerasTrainingLogger, self).__init__()
        self.data = data
        self.N = data.shape[0]
        self.log_dir = log_dir
        self.tl = trans_layer
        self.validation = None
        self.trans_func = None
        self.model = None
        self.sess = None
        self.step = 0
        
    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        
    def on_train_begin(self, logs=None):
        self.validation = self.params['do_validation']
        self.trans_func = K.function([self.model.inputs], [self.model.layers[self.tl].output])
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train')
        if self.validation:
            self.val_writer = tf.summary.FileWriter(self.log_dir + '/val')
        
        
    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        tf_loss = tf.summary.scalar('loss', tf.cast(logs.get('loss'), tf.float32))
        tf_acc = tf.summary.scalar('acc', tf.cast(logs.get('acc'), tf.float32))
        tf_summ = tf.summary.merge([tf_loss, tf_acc])
        summary = self.sess(tf_summ)
        self.train_writer.add_summary(summary, global_step=self.step)
        self.step += 1
        
    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        if self.validation:
            tf_loss = tf.summary.scalar('loss', tf.cast(logs.get('val_loss'), tf.float32))
            tf_acc = tf.summary.scalar('acc', tf.cast(logs.get('val_acc'), tf.float32))
            tf_summ = tf.summary.merge([tf_loss, tf_acc])
            summary = self.sess(tf_summ)
            self.val_writer.add_summary(summary, global_step=self.step)
            
        imgs = self.trans_func([self.data])
        tf_img = tf.summary.image(tf.cast(imgs, tf.float32))
        summary = self.sess(tf_img)
        self.train_writer.add_summary(summary, global_step=self.step)
        
    def on_train_end(self, logs=None):
        self.train_writer.close()
        if self.validation: self.val_writer.close()
        
#%%
if __name__ == '__main__':
    own_callback = KerasTrainingLogger()
