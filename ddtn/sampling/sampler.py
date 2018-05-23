#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:38:44 2018

@author: nsde
"""

#%%
import numpy as np

from ddtn.helper.transformer_util import get_transformer, get_transformer_dim
from ddtn.helper.transformer_util import get_transformer_init_weights
import tensorflow as tf

#%%
class image_registration(object):
    def __init__(self, transformer='affine'):
        self.trans_func = get_transformer(transformer)
        self.dim = get_transformer_dim(transformer)
        self.init = get_transformer_init_weights(1, transformer)[1]
        self.sess = tf.Session()

    def transform(self, im, theta):
        im = im[np.newaxis, :, :, :]
        theta = theta[np.newaxis, :]
        trans_im = self.trans_func(im, theta, im.shape[1:])
        return self.sess.run(trans_im)[0]
        
    def error_func(self, x, y):
        return np.linalg.norm(x - y)
    
    def proposal(self, theta):
        return np.random.multivariate_normal(mean=theta, cov=np.eye(self.dim)).astype('float32')
    
    def sampler(self, im1, im2, lm1, lm2, N=1000):
        im1 = im1.astype('float32')
        im2 = im2.astype('float32')
        
        # Initial sample
        current_samp = self.init.astype('float32')
        current_trans = self.transform(im2, current_samp)
        current_error = -self.error_func(im1, current_trans)
        
        accepted_samples = np.zeros((N, self.dim), dtype=np.float32)
        accepted_count = 0
        for i in range(N):
            
            # Proposal sample
            proposal_samp = self.proposal(current_samp)
            proposal_trans = self.transform(im2, proposal_samp)
            proposal_error = -self.error_func(im1, proposal_trans)
            diff_error = proposal_error - current_error
            accept = np.log(np.random.uniform()) < diff_error
            if accept:
                current_samp = proposal_samp
                current_trans = proposal_trans
                current_error = proposal_error
                accepted_samples[accepted_count] = proposal_samp
                accepted_count += 1
            print(i, accept, diff_error, current_error, proposal_error)  
            
        return accepted_samples
    
#%%
if __name__ == "__main__":
    im1, im2, lm1, lm2 = np.load('george_boy.npy')
    ir = image_registration()
    ir.sampler(im1, im2)
