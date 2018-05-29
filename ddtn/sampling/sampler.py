#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:38:44 2018

@author: nsde
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

from ddtn.transformers.transformer_util import get_transformer
from ddtn.transformers.transformer_util import get_transformer_dim
from ddtn.transformers.transformer_util import get_transformer_init_weights
import tensorflow as tf

#%%
class image_registration(object):
    """ NOT AT ALL SURE IF THIS WORK
        MCMC sampling for image registration using different transformers
    """
    def __init__(self, transformer='TPS'):
        self.trans_func = get_transformer(transformer)
        self.dim = get_transformer_dim(transformer)
        self.init = get_transformer_init_weights(1, transformer)[1]
        self.sess = tf.Session()

    def transform_img(self, im, theta):
        im = im[np.newaxis, :, :, :]
        theta = theta[np.newaxis, :]
        trans_im = self.trans_func(im, theta, im.shape[1:])
        return self.sess.run(trans_im)[0]
        
    def transform_lm(self, lm, theta):
        theta = np.reshape(theta, (1, 16, 2))
        trans_lm = self.trans_func(lm, theta)
        return self.sess.run(trans_lm)[0]
        
    def error_func(self, x, y):
        return np.linalg.norm(x - y)
    
    def proposal(self, theta):
        return np.random.multivariate_normal(mean=theta, cov=(1.0/self.dim**2)*np.eye(self.dim)).astype('float32')
    
    def sampler(self, im1, im2, lm1, lm2, N=1000):
        im1 = im1.astype('float32')
        im2 = im2.astype('float32')
        
        
        # Landmark transformation
        mu1 = np.mean(lm1, axis=1)
        s1 = np.std(lm1)
        mu2 = np.mean(lm2, axis=1)
        s2 = np.std(lm2)
        lm2 = ((lm2.T - mu2) * (s1 / s2) + mu1.T).T
        
        lm1 = lm1.astype('float32')
        lm2 = np.concatenate([lm2, np.ones((1, lm2.shape[1]))], axis=0).astype('float32')
        
        # Initial sample
        current_samp = self.init.astype('float32')
        current_trans = self.transform_lm(lm2, current_samp)
        current_error = -self.error_func(lm1, current_trans)
        
        accepted_samples = np.zeros((N, self.dim), dtype=np.float32)
        accepted_count = 0
        for i in range(N):
            
            # Proposal sample
            proposal_samp = self.proposal(current_samp)
            proposal_trans = self.transform_lm(lm2, proposal_samp)
            proposal_error = -self.error_func(lm1, proposal_trans)
            diff_error = proposal_error - current_error
            accept = np.log(np.random.uniform()) < diff_error
            print(i, accept, diff_error, current_error, proposal_error)  
            if accept:
                current_samp = proposal_samp
                current_trans = proposal_trans
                current_error = proposal_error
                accepted_samples[accepted_count] = proposal_samp
                accepted_count += 1
                
            plt.plot(lm1[0], lm1[1], 'b.')
            plt.plot(current_trans[0], current_trans[1], 'r.')
            plt.show()
            
            
        return accepted_samples
    
#%%
if __name__ == "__main__":
    im1, im2, lm1, lm2 = np.load('george_boy.npy')
    lm1 = np.reshape(lm1, (2, 68)).astype('float32')
    lm2 = np.reshape(lm2, (2, 68)).astype('float32')
    ir = image_registration()
    acs = ir.sampler(im1, im2, lm1, lm2)
