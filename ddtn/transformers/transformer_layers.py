#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:04:35 2017

@author: nsde
"""

#%%
import tensorflow as tf
from ddtn.helper.tf_funcs import tf_meshgrid, tf_interpolate
from ddtn.helper.tf_funcs import tf_TPS_system_solver, tf_expm3x3_analytic
from ddtn.cuda.tf_CPAB_transformer import tf_CPAB_transformer

#%%
def ST_Affine_transformer(U, theta, out_size):
    """ Spatial transformer using affine transformations
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        theta: `Matrix` [n_batch, 6]. Parameters for the transformation. Each
            row specify a transformation for each input image.
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
            
    Output:
        V: 4D-`Tensor` [n_batch, out_size[0], out_size[1], n_channels]. Tensor
            with transformed images.
    """
    with tf.name_scope('ST_Affine_transformer'):
        num_batch = tf.shape(theta)[0]
   
        # Reshape theta
        theta = tf.reshape(theta, (-1, 2, 3))
    
        # Create grid of points
        out_height = out_size[0]
        out_width = out_size[1]
        grid = tf_meshgrid(out_height, out_width)
        grid = tf.tile(tf.expand_dims(grid, 0), [num_batch, 1, 1])
        
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        
        # Slice and reshape
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1] )
        y_s_flat = tf.reshape(y_s, [-1])
        
        # Interpolate values
        V = tf_interpolate(U, x_s_flat, y_s_flat, out_size)
        return V

#%%
def ST_Affine_diffio_transformer(U, theta, out_size):
    """ Spatial transformer using diffeomorphic affine transformations
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        theta: `Matrix` [n_batch, 6]. Parameters for the transformation. Each
            row specify a transformation for each input image.
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
            
    Output:
        V: 4D-`Tensor` [n_batch, out_size[0], out_size[1], n_channels]. Tensor
            with transformed images.
    """
    with tf.name_scope('ST_Affine_diffeo_transformer'):
        num_batch = tf.shape(theta)[0]
   
        # Reshape theta
        theta = tf.reshape(theta, (-1, 2, 3))
    
        # Create grid of points
        out_height = out_size[0]
        out_width = out_size[1]
        grid = tf_meshgrid(out_height, out_width)
        grid = tf.tile(tf.expand_dims(grid, 0), [num_batch, 1, 1])
        
        # Take matrix exponential -> creates invertable affine transformation
        theta = tf.concat((theta, tf.zeros((num_batch, 1, 3))), axis=1)
        theta = tf_expm3x3_analytic(theta)
        theta = theta[:,:2,:]
        
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(theta, grid)
        
        # Slice and reshape
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1] )
        y_s_flat = tf.reshape(y_s, [-1])
        
        # Interpolate values
        V = tf_interpolate(U, x_s_flat, y_s_flat, out_size)
        return V

#%%
def ST_CPAB_transformer(U, theta, out_size):
    """ Spatial transformer using CPAB transformations
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        theta: `Matrix` [n_batch, d]. Parameters for the transformation. Each
            row specify a transformation for each input image. The number d is
            determined by tessalation. See transformer/setup_CPAB_transformer.py
            for more information.
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
            
    Output:
        V: 4D-`Tensor` [n_batch, out_size[0], out_size[1], n_channels]. Tensor
            with transformed images.
    """
    with tf.name_scope('ST_CPAB_transformer'):
        # Create grid of points
        out_height = out_size[0]
        out_width = out_size[1]
        grid = tf_meshgrid(out_height, out_width)
        
        # Transform grid
        T_g = tf_CPAB_transformer(grid[:2], theta)
        
        # Slice and reshape
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        
        # Interpolate values
        V = tf_interpolate(U, x_s_flat, y_s_flat, out_size)
        return V

#%%
def ST_Homografy_transformer(U, theta, out_size):
    """ Spatial transformer using homografy transformations
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        theta: `Matrix` [n_batch, 9]. Parameters for the transformation. Each
            row specify a transformation for each input image.
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
            
    Output:
        V: 4D-`Tensor` [n_batch, out_size[0], out_size[1], n_channels]. Tensor
            with transformed images.
    """
    with tf.name_scope('ST_Homografy_transformer'):
        num_batch = tf.shape(theta)[0]
   
        # Reshape theta
        theta = tf.reshape(theta, (-1, 3, 3))
    
        # Create grid of points
        out_height = out_size[0]
        out_width = out_size[1]
        grid = tf_meshgrid(out_height, out_width)
        grid = tf.tile(tf.expand_dims(grid, 0), [num_batch, 1, 1])
        
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s, z_s)
        T_g = tf.matmul(theta, grid)
        
        # Make non-homo cordinates (x_s, y_s, z_s) -> (x_s / z_s, y_s / z_s, 1)
        T_g = T_g[:,:2,:] / tf.expand_dims(T_g[:,2,:], 1)
        
        # Slice and reshape
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1] )
        y_s_flat = tf.reshape(y_s, [-1])
        
        # Interpolate values
        V = tf_interpolate(U, x_s_flat, y_s_flat, out_size)
        return V
        

#%%
def ST_TPS_transformer(U, theta, out_size, tps_size = [4,4]):
    """ Spatial transformer using thin plate spline transformations
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        theta: `Matrix` [n_batch, 2*tps_size[0]*tps_size[1]]. Parameters for 
            the transformation. Each row specify a transformation for each 
            input image. 
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
        tps_size: `list` where tps_size[0] is the number of points in the x
            direction and tps_size[1] is the number of points in the y direction.
            This should be set to match the dimension of theta.
    Output:
        V: 4D-`Tensor` [n_batch, out_size[0], out_size[1], n_channels]. Tensor
            with transformed images.
    """
    with tf.name_scope('ST_TPS_transformer'):
        num_batch = tf.shape(theta)[0]
        
        # Create grid of points
        out_height = out_size[0]
        out_width = out_size[1]
        grid = tf_meshgrid(out_height, out_width)
        grid = tf.tile(tf.expand_dims(grid, 0), [num_batch, 1, 1])
        
        # Solve TPS system
        target = tf.reshape(theta, (-1, tps_size[0]*tps_size[1], 2))
        source = tf_meshgrid(tps_size[0], tps_size[1])
        source = tf.tile(tf.expand_dims(source, 0), [num_batch, 1, 1])
        T = tf_TPS_system_solver(source, target)
        
        # Transform points with TPS kernel
        T_g = tf.matmul(T, grid)
        
        # Slice and reshape
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        
        # Interpolate values
        V = tf_interpolate(U, x_s_flat, y_s_flat, out_size)
        return V

#%%
def ST_Affine_transformer_batch(U, thetas, out_size):
    """ Batch version of the affine transformer. Applies a batch of affine
        transformations to each image in U.
        
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        thetas: 3D-`Tensor` [n_batch, n_trans, 6]. Parameters for 
            the transformation. Note that for each image, we expect [n_trans, 6]
            parameters, and thus each image is transformed uniquly n_trans times
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
    
    Output:
        V: 4D-`Tensor` [n_batch*n_trans, out_size[0], out_size[1], n_channels].
            Tensor with transformed images. Note that the number of output images,
            are not the same as the input images, since each image is transformed
            n_trans times.
    """
    with tf.name_scope('ST_Affine_transformer_batch'):
        num_batch, num_transformes = map(int, thetas.get_shape().as_list()[:2])
        
        # Repeat the input images n_trans times
        indices = [[i] * num_transformes for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        
        # Call transformer on repeated input
        V = ST_Affine_transformer(input_repeated, thetas, out_size)
        return V
    
#%%
def ST_Affine_diffio_transformer_batch(U, thetas, out_size):
    """ Batch version of the diffeomorphic affine transformer. Applies a batch 
        of affine transformations to each image in U.
        
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        thetas: 3D-`Tensor` [n_batch, n_trans, 6]. Parameters for 
            the transformation. Note that for each image, we expect [n_trans, 6]
            parameters, and thus each image is transformed uniquly n_trans times
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
    
    Output:
        V: 4D-`Tensor` [n_batch*n_trans, out_size[0], out_size[1], n_channels].
            Tensor with transformed images. Note that the number of output images,
            are not the same as the input images, since each image is transformed
            n_trans times.
    """
    with tf.name_scope('ST_Affine_diffeo_transformer_batch'):
        num_batch, num_transformes = map(int, thetas.get_shape().as_list()[:2])
        
        # Repeat the input images n_trans times
        indices = [[i] * num_transformes for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        
        # Call transformer on repeated input
        V = ST_Affine_diffio_transformer(input_repeated, thetas, out_size)
        return V

#%%
def ST_CPAB_transformer_batch(U, thetas, out_size):
    """ Batch version of the CPAB transformer. Applies a batch of CPAB
        transformations to each image in U.
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        thetas: 3D-`Tensor` [n_batch, n_trans, d]. Parameters for 
            the transformation. Note that for each image, we expect [n_trans, 6]
            parameters, and thus each image is transformed uniquly n_trans times.
            The number d is determined by tessalation. 
            See transformer/setup_CPAB_transformer.py for more information.
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
    
    Output:
        V: 4D-`Tensor` [n_batch*n_trans, out_size[0], out_size[1], n_channels].
            Tensor with transformed images. Note that the number of output images,
            are not the same as the input images, since each image is transformed
            n_trans times.
    """
    with tf.name_scope('ST_CPAB_transformer_batch'):
        num_batch, num_transformes = map(int, thetas.get_shape().as_list()[:2])
        
        # Repeat the input images n_trans times
        indices = [[i] * num_transformes for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        
        # Call transformer on repeated input
        V = ST_CPAB_transformer(input_repeated, thetas, out_size)
        return V 

#%%
def ST_Homografy_transformer_batch(U, thetas, out_size):
    """ Batch version of the homografy transformer. Applies a batch of homografy
        transformations to each image in U.
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        thetas: 3D-`Tensor` [n_batch, n_trans, 6]. Parameters for 
            the transformation. Note that for each image, we expect [n_trans, 6]
            parameters, and thus each image is transformed uniquly n_trans times
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
    
    Output:
        V: 4D-`Tensor` [n_batch*n_trans, out_size[0], out_size[1], n_channels].
            Tensor with transformed images. Note that the number of output images,
            are not the same as the input images, since each image is transformed
            n_trans times.
    """
    with tf.name_scope('ST_Homografy_transformer_batch'):
        num_batch, num_transformes = map(int, thetas.get_shape().as_list()[:2])
        
        # Repeat the input images n_trans times
        indices = [[i] * num_transformes for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        
        # Call transformer on repeated input
        V = ST_Homografy_transformer(input_repeated, thetas, out_size)
        return V 
    
#%%
def ST_TPS_transformer_batch(U, thetas, out_size, tps_size = [4,4]):
    """ Batch version of the TPS transformerST_Affine_transformer. Applies a batch of TPS
        transformations to each image in U.
    
    Arguments:
        U: 4D-`Tensor` [n_batch, height, width, n_channels]. Input images to
            transform.
        thetas: 3D-`Tensor` [n_batch, n_trans, 2*tps_size[0]*tps_size[1]]. Parameters for 
            the transformation. Note that for each image, we expect [n_trans, 6]
            parameters, and thus each image is transformed uniquly n_trans times
        out_size: `list` where out_size[0] is the output height and out_size[1]
            is the output width of each interpolated image.
        tps_size: `list` where tps_size[0] is the number of points in the x
            direction and tps_size[1] is the number of points in the y direction.
            This should be set to match the dimension of theta.
    
    Output:
        V: 4D-`Tensor` [n_batch*n_trans, out_size[0], out_size[1], n_channels].
            Tensor with transformed images. Note that the number of output images,
            are not the same as the input images, since each image is transformed
            n_trans times.
    """
    with tf.name_scope('ST_TPS_transformer_batch'):
        num_batch, num_transformes = map(int, thetas.get_shape().as_list()[:2])
        
        # Repeat the input images n_trans times
        indices = [[i] * num_transformes for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        
        # Call transformer on repeated input
        V = ST_TPS_transformer(input_repeated, thetas, out_size, tps_size)
        return V 

#%%
if __name__ == '__main__':
    import numpy as np
    from ddtn.helper.utility import get_cat, show_images
    
    # Load im and create a batch of imgs
    N = 15
    im = get_cat()
    im = np.tile(im, (N, 1, 1, 1))
   
    # Create transformation vector
    theta = np.tile(np.array([1,0,0,0,1,0], np.float32), (N, 1))
    theta[:,2] = np.random.normal(scale=0.5, size=N)
    theta[:,5] = np.random.normal(scale=0.5, size=N)
    
    # Cast to tensorflow and normalize values
    im_tf = tf.cast(im, tf.float32)
    theta_tf = tf.cast(theta, tf.float32)
    
    # Transformer imgs
    trans_im = ST_Affine_transformer(im_tf, theta_tf, (1200, 1600))

    # Run computations
    sess = tf.Session()
    out_im = sess.run(trans_im)
    
    show_images(out_im, cols=3)
    
    
    
    
