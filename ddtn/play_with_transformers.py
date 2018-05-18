# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:27:39 2018

@author: nsde
"""
#%%
from ddtn.transformers.setup_CPAB_transformer import setup_CPAB_transformer
from ddtn.transformers.transformer_util import get_transformer, get_random_theta
from ddtn.helper.utility import get_cat, show_images
import numpy as np
import tensorflow as tf

#%%
if __name__ == '__main__':
    # Set this
    transformer_name = 'TPS' # transformer to use
    N = 15 # number of transformations
    
    # Special for CPAB
    if transformer_name=='CPAB': s = setup_CPAB_transformer()
    
    # Get transformer and random transformation
    transformer = get_transformer(transformer_name)
    theta = get_random_theta(N, transformer_name)
    
    # Load im and create a batch of imgs
    im = get_cat()
    im = np.tile(im, (N, 1, 1, 1))
        
    # Cast to tensorflow
    im_tf = tf.cast(im, tf.float32)
    theta_tf = tf.cast(theta, tf.float32)
    
    # Transformer imgs
    trans_im = transformer(im_tf, theta_tf, (1200, 1600))

    # Run computations
    sess = tf.Session()
    out_im = sess.run(trans_im)
    
    show_images(out_im, title=transformer_name + ' transformations', cols=3)