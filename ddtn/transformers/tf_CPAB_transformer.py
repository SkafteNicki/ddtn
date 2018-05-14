#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:51:23 2017

@author: nsde
"""

#%%
import tensorflow as tf
from ddtn.helper.tf_funcs import tf_mymin, tf_repeat_matrix, tf_expm3x3_analytic
from ddtn.helper.utility import load_basis

#%%
def tf_findcellidx(points, ncx, ncy, inc_x, inc_y):
    """ Computes the cell index for some points and a given tessalation 
    
    Arguments:
        points: 3D-`Tensor` [n_points,3,1], with points in homogeneous coordinates
        ncx, ncy: `integer`, with the number of cells in the x and y direction
        inc_x, inc_y: `floats`, the size of the cells in the x and y direction
    
    Output:
        idx: 1D-`Tensor` [n_points,], with the cell idx for each input point
    """
    with tf.name_scope('findcellidx'):
        p = tf.transpose(tf.squeeze(points)) # 2 x n_points
        ncx, ncy = tf.cast(ncx, tf.float32), tf.cast(ncy, tf.float32)
        inc_x, inc_y = tf.cast(inc_x, tf.float32), tf.cast(inc_y, tf.float32)
    
        # Move according to lower bounds
        p = tf.cast(p + 1, tf.float32)
        
        p0 = tf.minimum((ncx*inc_x - 1e-8), tf.maximum(0.0, p[0,:]))
        p1 = tf.minimum((ncy*inc_y - 1e-8), tf.maximum(0.0, p[1,:]))
            
        xmod = tf.mod(p0, inc_x)
        ymod = tf.mod(p1, inc_y)
            
        x = xmod / inc_x
        y = ymod / inc_y
        
        # Calculate initial cell index    
        cell_idx =  tf_mymin((ncx - 1) * tf.ones_like(p0), (p0 - xmod) / inc_x) + \
                    tf_mymin((ncy - 1) * tf.ones_like(p0), (p1 - ymod) / inc_y) * ncx 
        cell_idx *= 4
    
        cell_idx1 = cell_idx+1
        cell_idx2 = cell_idx+2
        cell_idx3 = cell_idx+3

        # Conditions to evaluate        
        cond1 = tf.less_equal(p[0,:], 0) #point[0]<=0
        cond1_1 = tf.logical_and(tf.less_equal(p[1,:], 0), tf.less(p[1,:]/inc_y, 
            p[0,:]/inc_x))#point[1] <= 0 && point[1]/inc_y<point[0]/inc_x
        cond1_2 = tf.logical_and(tf.greater_equal(p[1,:], ncy*inc_y), tf.greater(
            p[1,:]/inc_y - ncy, -p[0,:]/inc_x))#(point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx
        cond2 = tf.greater_equal(p[0,:], ncx*inc_x) #point[0] >= ncx*inc_x
        cond2_1 = tf.logical_and(tf.less_equal(p[1,:],0), tf.greater(-p[1,:]/inc_y,
            p[0,:]/inc_x-ncx))#point[1]<=0 && -point[1]/inc_y > point[0]/inc_x - ncx
        cond2_2 = tf.logical_and(tf.greater_equal(p[1,:],ncy*inc_y), tf.greater(
            p[1,:]/inc_y - ncy,p[0,:]/inc_x-ncx))#point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx
        cond3 = tf.less_equal(p[1,:], 0) #point[1] <= 0
        cond4 = tf.greater_equal(p[1,:], ncy*inc_y) #point[1] >= ncy*inc_y
        cond5 = tf.less(x, y) #x<y
        cond5_1 = tf.less(1-x, y) #1-x<y
    
        # Take decision based on the conditions
        idx = tf.where(cond1, tf.where(cond1_1, cell_idx, tf.where(cond1_2, cell_idx2, cell_idx3)),
              tf.where(cond2, tf.where(cond2_1, cell_idx, tf.where(cond2_2, cell_idx2, cell_idx1)),
              tf.where(cond3, cell_idx, 
              tf.where(cond4, cell_idx2,
              tf.where(cond5, tf.where(cond5_1, cell_idx2, cell_idx3), 
              tf.where(cond5_1, cell_idx1, cell_idx))))))
    
        return idx

#%%
def tf_CPAB_transformer(points, theta):
    """ CPAB transformer in pure tensorflow. Transform the input points by
        repeatly appling the matrix-exponentials parametrized by theta. This
        function should automatic be able to calculate the gradient of the
        output w.r.t. theta.
    
    Arguments:
        points: `Matrix` [2, n_points]. 2D input points to transform
        theta: `Matrix` [n_theta, dim]. Parametrization to use. 
            
    Output:
        trans_points: 3D-`Tensor` [n_theta, 2, n_points]. The transformed points
            for each parametrization in theta.
    """
    with tf.name_scope('CPAB_transformer'):
        # Make sure that both inputs are in float32 format
        points = tf.cast(points, tf.float32) # format [2, nb_points]
        theta = tf.cast(theta, tf.float32) # format [n_theta, dim]
        n_theta = tf.shape(theta)[0]
        n_points = tf.shape(points)[1]
        
        # Repeat point matrix, one for each theta
        newpoints = tf_repeat_matrix(points, n_theta) # [n_theta, 2, nb_points]
        
        # Reshape into a [nb_points*n_theta, 2] matrix
        newpoints = tf.reshape(tf.transpose(newpoints, perm=[0,2,1]), (-1, 2))
        
        # Add a row of ones, creating a [nb_points*n_theta, 3] matrix
        newpoints = tf.concat([newpoints, tf.ones((n_theta*n_points, 1))], axis=1)
        
        # Expand dims for matrix multiplication later -> [nb_points*n_theta, 3, 1] tensor
        newpoints = tf.expand_dims(newpoints, 2)
        
        # Load file with basis
        file = load_basis()
        
        # Tessalation information
        nC = tf.cast(file['nC'], tf.int32)
        ncx = tf.cast(file['ncx'], tf.int32)
        ncy = tf.cast(file['ncy'], tf.int32)
        inc_x = tf.cast(file['inc_x'], tf.float32)
        inc_y = tf.cast(file['inc_y'], tf.float32)
        
        # Steps sizes
        nStepSolver = 50 # Change this for more precision
        dT = 1.0 / tf.cast(nStepSolver, tf.float32)
        
        # Get cpab basis
        B = tf.cast(file['B'], tf.float32)

        # Repeat basis for batch multiplication
        B = tf_repeat_matrix(B, n_theta)
        
        # Calculate the row-flatted affine transformations Avees 
        Avees = tf.matmul(B, tf.expand_dims(theta, 2))
		
        # Reshape into (number of cells, 2, 3) tensor
        As = tf.reshape(Avees, shape = (n_theta * nC, 2, 3)) # format [n_theta * nC, 2, 3]
        
        # Multiply by the step size and do matrix exponential on each matrix
        Trels = tf_expm3x3_analytic(dT*As)
        Trels = tf.concat([Trels, tf.cast(tf.reshape(tf.tile([0,0,1], 
                [n_theta*nC]), (n_theta*nC, 1, 3)), tf.float32)], axis=1)
        
        # Batch index to add to correct for the batch effect
        batch_idx = (4*ncx*ncy) * tf.reshape(tf.transpose(tf.ones((n_points, n_theta), 
                    dtype=tf.int32)*tf.cast(tf.range(n_theta), tf.int32)),(-1,))
        
        # Body function for while loop (executes the computation)
        def body(i, points):
            # Find cell index of each point
            idx = tf_findcellidx(points, ncx, ncy, inc_x, inc_y)
            
            # Correct for batch
            corrected_idx = tf.cast(idx, tf.int32) + batch_idx
            
            # Gether relevant matrices
            Tidx = tf.gather(Trels, corrected_idx)
            
            # Transform points
            newpoints = tf.matmul(Tidx, points)
            
            # Shape information is lost, but tf.while_loop requires shape 
            # invariance so we need to manually set it (easy in this case)
            newpoints.set_shape((None, 3, 1)) 
            return i+1, newpoints
        
        # Condition function for while loop (indicates when to stop)
        def cond(i, points):
            # Return iteration bound
            return tf.less(i, nStepSolver)
        
        # Run loop
        trans_points = tf.while_loop(cond, body, [tf.constant(0), newpoints],
                                     parallel_iterations=10, back_prop=True)[1]
        # Reshape to batch format
        trans_points = tf.reshape(trans_points[:,:2], (n_theta, 2, n_points))
        return trans_points
        
#%%
if __name__ == '__main__':
    pass        