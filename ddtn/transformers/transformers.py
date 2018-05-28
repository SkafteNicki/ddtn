# -*- coding: utf-8 -*-
"""
Created on Fri May 25 09:41:11 2018

@author: nsde
"""

#%%
import tensorflow as tf
from ddtn.helper.tf_funcs import tf_meshgrid, tf_TPS_meshgrid
from ddtn.helper.tf_funcs import tf_TPS_system_solver, tf_expm3x3_analytic

#%%
from sys import platform as _platform
from ddtn.helper.utility import check_for_gpu, check_cuda_support
# This will load the fast cuda version of the CPAB transformer and gradient for
# linux and MAC OS X and load the slower pure tensorflow implemented CPAB 
# transformer for windows

gpu = check_for_gpu() and check_cuda_support()
if (_platform == "linux" or _platform == "linux2" \
    or _platform == "darwin") and gpu: # linux or MAC OS X
   from ddtn.cuda.CPAB_transformer import tf_cuda_CPAB_transformer as tf_CPAB_transformer
else: # Windows 32 or 64-bit or no GPU
   from ddtn.cuda.CPAB_transformer import tf_pure_CPAB_transformer as tf_CPAB_transformer
   
#%%
def tf_Affine_transformer(points, theta):
    """
    Arguments:
        points: `Matrix` [2, np] of grid points to transform
        theta: `Matrix` [bs, 2, 3] with a batch of transformations
    """
    with tf.name_scope('Affine_transformer'):
        num_batch = tf.shape(theta)[0]
        grid = tf.tile(tf.expand_dims(points, 0), [num_batch, 1, 1])
    
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        return T_g

#%%
def tf_Affinediffeo_transformer(points, theta):
    """
    Arguments:
        points: `Matrix` [2, np] of grid points to transform
        theta: `Matrix` [bs, 2, 3] with a batch of transformations
    """
    with tf.name_scope('Affinediffio_transformer'):
        num_batch = tf.shape(theta)[0]
        grid = tf.tile(tf.expand_dims(points, 0), [num_batch, 1, 1])
    
        # Take matrix exponential -> creates invertable affine transformation
        theta = tf_expm3x3_analytic(theta)
        theta = theta[:,:2,:]
    
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        return T_g

#%%
def tf_Homografy_transformer(points, theta):
    """
    Arguments:
        points: `Matrix` [2, np] of grid points to transform
        theta: `Matrix` [bs, 3, 3] with a batch of transformations
    """
    with tf.name_scope('Homografy_transformer'):
        num_batch = tf.shape(theta)[0]
        grid = tf.tile(tf.expand_dims(points, 0), [num_batch, 1, 1])
        
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s, z_s)
        T_g = tf.matmul(theta, grid)
        
        # Make non-homo cordinates (x_s, y_s, z_s) -> (x_s / z_s, y_s / z_s, 1)
        T_g = T_g[:,:2,:] / tf.expand_dims(T_g[:,2,:], 1)
        return T_g

#%%    
def tf_TPS_transformer(points, theta, tps_size=[4,4]):
    """
    Arguments:
        points: `Matrix` [2, np] of grid points to transform
        theta: `Matrix` [bs, tps_size[0]*tps_size[1], 2] with a batch of transformations
    """
    with tf.name_scope('TPS_transformer'):
        num_batch = tf.shape(theta)[0]  
        
        # Solve TPS system
        source = tf.transpose(tf_meshgrid(tps_size[0], tps_size[1])[:2,:]) # [np, 2]
        source = tf.tile(tf.expand_dims(source, 0), [num_batch, 1, 1]) # [bs, np, 2]
        T = tf_TPS_system_solver(source, theta) # [bs, 2, np]
    
        # Calculate TPS grid
        grid = tf_TPS_meshgrid(points, source) 
    
        # Transform points with TPS kernel
        T_g = tf.matmul(T, grid)
        return T_g

#%%
if __name__ == '__main__':
    pass