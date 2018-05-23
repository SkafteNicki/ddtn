#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:51:23 2017

@author: nsde
"""

#%%

#%%

#%%

        
#%%
if __name__ == '__main__':
    from ddtn.cuda.tf_CPAB_transformer import tf_CPAB_transformer as tf_cuda_transformer
    from ddtn.transformers.setup_CPAB_transformer import setup_CPAB_transformer
    import numpy as np
    from time import time
    
    # Setup transformer and sample grid + parametrization
    s = setup_CPAB_transformer()
    points = s.sample_grid(20)
    theta = s.sample_theta_without_prior(5)
    
    points_tf = tf.cast(points, tf.float32)
    theta_tf = tf.cast(theta, tf.float32)
    
    trans_points1 = tf_pure_CPAB_transformer(points, theta)
    trans_points2 = tf_cuda_transformer(points, theta)
    
    sess = tf.Session()
    
    start = time()
    res1 = sess.run(trans_points1)
    time1 = time() - start
    
    start = time()
    res2 = sess.run(trans_points2)
    time2 = time() - start
    
    # Print res
    print('Pure tf implementation: ', time1)
    print('Cuda implementation:    ', time2)
    print('Difference: ', np.linalg.norm(res1-res2))
    
    
    
    