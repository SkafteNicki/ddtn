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
    """ A Keras callback class that can be given to any Keras model.fit(...)
        method. This class does two things:
            * Logs the loss and acc of the model after each batch
            * Transform some input imgs after each epoch and logs these
    
    Arguments
        data: 4D-`Tensor` [n_imgs, height, width, n_channel] with images that
            are transformed in the end of each epoch
            
        name: string, results are stored in a folder called 'log_dir/name'. 
            Thus to easily compare the results of different runs in tensorboard, 
            keep the logdir the same but change the name for each run
        
        log_dir: string, directory to store the tensorboard files
            
        trans_layer: integer determining where the transformation layer is in
            the model (usually translayer=0, because it is the first layer)
    """
    def __init__(self, data, name, log_dir='./logs', trans_layer=0, **kwargs):
        super(KerasTrainingLogger, self).__init__(**kwargs)
        
        # Initialize variables
        self.data = data
        self.log_dir = log_dir + '/' + name
        self.tl = trans_layer
        self.step = 0
        self.validation = self.trans_func = self.model = self.sess = None
               
        # Placeholders for summary
        self.loss_p = tf.placeholder(tf.float32)
        self.acc_p = tf.placeholder(tf.float32)
        self.img_p = tf.placeholder(tf.float32, shape=data.shape)
        
        # Summary operations
        self.summ_op = tf.summary.merge([tf.summary.scalar('loss', self.loss_p),
                                         tf.summary.scalar('acc', self.acc_p)])
        self.summ_val_op = tf.summary.merge([tf.summary.scalar('loss_val', self.loss_p),
                                             tf.summary.scalar('acc_val', self.acc_p)])
    
        self.img_op = tf.summary.image('trans_img', self.img_p)
        
    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        
    def on_train_begin(self, logs=None):
        self.validation = self.params['do_validation']
        self.trans_func = K.function([self.model.input], [self.model.layers[self.tl].output])
        self.train_writer = tf.summary.FileWriter(self.log_dir)
        if self.validation:
            self.val_writer = tf.summary.FileWriter(self.log_dir)
                
    def on_batch_end(self, batch, logs=None):
        logs = {} if logs is None else logs
        summary = self.sess.run(self.summ_op, feed_dict={
                                self.loss_p: logs.get('loss'),
                                self.acc_p: logs.get('acc')})
        self.train_writer.add_summary(summary, global_step=self.step)
        self.step += 1
        
    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        if self.validation:
            summary = self.sess.run(self.summ_val_op, feed_dict={
                                    self.loss_p: logs.get('val_loss'),
                                    self.acc_p: logs.get('val_acc') })
            self.val_writer.add_summary(summary, global_step=self.step)

        imgs = self.trans_func([self.data])[0]
        summary = self.sess.run(self.img_op, feed_dict={self.img_p: imgs})
        self.train_writer.add_summary(summary, global_step=self.step)
        
    def on_train_end(self, logs=None):
        #self.sess.close()
        self.train_writer.close()
        if self.validation: self.val_writer.close()
        
#%%
if __name__ == '__main__':
    own_callback = KerasTrainingLogger()
