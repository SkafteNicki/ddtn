#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:13:53 2017

@author: nsde
"""
#%%
from .transformer.setup_CPAB_transformer import setup_CPAB_transformer
from .cuda.tf_CPAB_transformer import tf_CPAB_transformer, tf_CPAB_transformer_old
from .transformer.tf_CPAB_transformer import tf_CPAB_transformer as tf_CPAB_transformer_new
import tensorflow as tf
import numpy as np
import time

#%%
def profiler_gpu(bs = 50, img_s = 200):
    # Setup transformer
    s = setup_CPAB_transformer(ncx=2, 
                               ncy=2,
                               valid_outside = 1,
                               zero_trace = 0, 
                               zero_boundary = 0,
                           override = True)
    # Sample theta value
    theta_true = s.sample_theta_without_prior(bs)    
    
    # Sample grid
    points = s.sample_grid(img_s)
    
    # Start tensorflow seesion
    with tf.Session() as sess:
        tf_theta = tf.cast(theta_true, tf.float32)
        tf_points = tf.cast(points, tf.float32)
        start = time.time()
        newpoints1 = sess.run(tf_CPAB_transformer(tf_points, tf_theta))
        print("Time: ", time.time() - start)

#%%
def transformer_speed_test(bs = 50, img_s = 200, rep = 1, 
                           ncx = 2, ncy = 2, device = '/gpu:0'):
    # Setup transformer
    s = setup_CPAB_transformer(ncx=ncx, 
                               ncy=ncy,
                               valid_outside = 1,
                               zero_trace = 0, 
                               zero_boundary = 0,
                               override = True)
    # Sample theta value
    theta_true = s.sample_theta_without_prior(bs)    
    
    # Sample grid
    points = s.sample_grid(img_s)
    
    # Start tensorflow session
    with tf.Session() as sess:
        with tf.device(device):
            tf_theta = tf.cast(theta_true, tf.float32)
            tf_points = tf.cast(points, tf.float32)
            
            start = time.time()
            for _ in range(rep):    
                newpoints1 = sess.run(tf_CPAB_transformer_new(tf_points, tf_theta))
            t1 = (time.time() - start) / rep
            start = time.time()
            for _ in range(rep):
                newpoints2 = sess.run(tf_CPAB_transformer_old(tf_points, tf_theta))
            t2 = (time.time() - start) / rep
    
            print('Batch sitf_CPAB_transformerze: ', bs)
            print('Img. dim: ', (img_s, img_s))
            print('Number of reps: ', rep)    
            print('NEW. Time it took: ', round(t1, 3))
            print('OLD. Time it took: ', round(t2, 3))
            print('Speedup:           ', round(t2/t1, 3))
            print('Difference in output: ', np.linalg.norm(newpoints2 - newpoints1) / (img_s**2))

#%%
def new_transformer_speed_test(bs = 50, img_s = 200, rep = 1,
                               ncx = 2, ncy = 2, device = '/gpu:0'):
    
    # Setup transformer
    s = setup_CPAB_transformer(ncx=ncx, 
                               ncy=ncy,
                               valid_outside = 1,
                               zero_trace = 0, 
                               zero_boundary = 0,
                               override = True)
    # Sample theta value
    theta_true = s.sample_theta_without_prior(bs)
    
    # Sample grid
    points = s.sample_grid(img_s)
    
    # Start tensorflow session
    with tf.Session() as sess:
        with tf.device(device):
            tf_theta = tf.cast(theta_true, tf.float32)
            tf_points = tf.cast(points, tf.float32)
            
            start = time.time()
            for _ in range(rep):    
                newpoints1 = sess.run(tf_CPAB_transformer(tf_points, tf_theta))
            t1 = (time.time() - start) / rep
            start = time.time()
            for _ in range(rep):
                newpoints2 = sess.run(tf_CPAB_transformer_old(tf_points, tf_theta))
            t2 = (time.time() - start) / rep
    
            print('Batch size: ', bs)
            print('Img. dim: ', (img_s, img_s))
            print('Number of reps: ', rep)    
            print('NEW. Time it took: ', round(t1, 3))
            print('OLD. Time it took: ', round(t2, 3))
            print('Speedup:           ', round(t2/t1, 3))
            print('Difference in output: ', np.linalg.norm(newpoints2 - newpoints1) / (img_s**2))

    
               
#%%
def gradient_speed_test(bs = 50, img_s = 200, rep = 1,
                        ncx = 2, ncy = 2, device = '/gpu:0'):
    # Setup transformer
    s = setup_CPAB_transformer(ncx=ncx, 
                               ncy=ncy,
                               valid_outside = 1,
                               zero_trace = 0, 
                               zero_boundary = 0,
                               override = True)
    # Sample theta value
    theta_true = s.sample_theta_without_prior(bs)    
    
    # Sample grid
    points = s.sample_grid(img_s)
    
    # Start tensorflow session
    with tf.Session() as sess:
        with tf.device(device):
            tf_theta = tf.cast(theta_true, tf.float32)
            tf_points = tf.cast(points, tf.float32)
            
            start = time.time()
            for _ in range(rep):
                grad = sess.run(tf.gradients(tf_CPAB_transformer(tf_points, tf_theta), [tf_theta])[0])    
            t1 = (time.time() - start) / rep
                
            start = time.time()
            for _ in range(rep):
                old_grad = sess.run(tf.gradients(tf_CPAB_transformer_old(tf_points, tf_theta), [tf_theta])[0])
            t2 = (time.time() - start) / rep
            
            print('Batch size: ', bs)
            print('Img. dim: ', (img_s, img_s))
            print('Number of reps: ', rep)    
            print('NEW. Time it took: ', round(t1, 3))
            print('OLD. Time it took: ', round(t2, 3))
            print('Speedup:           ', round(t2/t1, 3))
            print('Difference in output: ', np.linalg.norm(old_grad - grad) / (grad.size))
            