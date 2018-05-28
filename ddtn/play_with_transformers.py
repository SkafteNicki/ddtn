# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:27:39 2018

@author: nsde
"""
#%%
from ddtn.transformers.setup_CPAB_transformer import setup_CPAB_transformer
from ddtn.transformers.transformer_util import get_transformer_layer, get_random_theta
from ddtn.helper.utility import get_cat, show_images
import numpy as np
import tensorflow as tf
import argparse

#%%
def _argument_parser():
    parser = argparse.ArgumentParser(description='''This program will deform a
                                     image of a cat using different transformations''')
    # Argument for transformer type
    parser.add_argument('-t', action="store", dest="transformer", type=str, 
                        default='CPAB', help='''Transformer type to use. 
                        Choose between: affine, cpab, affine_diffio, homografy
                        or TPS''')
    # Argument for number of transformations
    parser.add_argument('-n', action="store", dest="n_img", type=int, default = 15,
                        help = '''Number of images to transform. Default 15''')
    res = parser.parse_args()
    args = vars(res)
    return args


#%%
if __name__ == '__main__':
    # Get command line arguments
    args = _argument_parser()
    
    # Set this
    transformer_name = args['transformer'] # transformer to use
    N = args['n_img'] # number of transformations
    
    # Special for CPAB
    if transformer_name=='CPAB': s = setup_CPAB_transformer()
    
    # Get transformer and random transformation
    transformer = get_transformer_layer(transformer_name)
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
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    sess = tf.Session()
    out_im = sess.run(trans_im, options=run_options)
    
    # Show the transformed images
    show_images(out_im, title=transformer_name + ' transformations')