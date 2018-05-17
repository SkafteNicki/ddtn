# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:27:39 2018

@author: nsde
"""

#%%
from ddtn.helper.tf_funcs import tf_interpolate
from ddtn.transformers.setup_CPAB_transformer import setup_CPAB_transformer
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#%%
def tf_repeat(x, n_repeats):
    """
    Tensorflow implementation of np.repeat(x, n_repeats)
    """
    with tf.name_scope('repeat'):
        ones = tf.ones(shape=(n_repeats, ))
        rep = tf.transpose(tf.expand_dims(ones, 1), [1, 0])
        rep = tf.cast(rep, x.dtype)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

#%%
if __name__ == '__main__':
    s = setup_CPAB_transformer()
    
    N = 5
    im = np.random.normal(size=(N, 20, 20, 3))
    out_size = im.shape[1:3]
    points = s.sample_grid_image(out_size)
    theta = s.sample_theta_without_prior(N)
    newpoints = np.array([s.calcTrans(theta[i].flatten(), points) for i in range(N)])
    
    x = newpoints[:,0].flatten()
    y = newpoints[:,1].flatten()
    #im = tf.cast(im, tf.float32)
    im = tf.placeholder(tf.float32, (None, 20, 20, 3))
    
    with tf.name_scope('interpolate'):
        # Constants
        n_batch = tf.shape(im)[0] # (often) unknown size
        _, height, width, n_channels = im.shape.as_list() # known sizes

        # Cast value to float dtype
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # Scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # Do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        # Find index of each corner point
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = tf_repeat(tf.range(n_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, (-1, n_channels))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # And finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        newim = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        
        # Reshape into image format
        newim = tf.reshape(newim, (n_batch, out_height, out_width, n_channels))
