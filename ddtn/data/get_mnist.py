# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:57:24 2018

@author: nsde
"""

#%%
import tensorflow as tf
import numpy as np
import urllib
import os

#%%
def get_mnist():
    """ Download mnist by the keras interface """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test

#%%
def get_mnist_distorted():
    """ Downloads distorted mnist from internet """
    url = "https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz"
    file_name = url.split('/')[-1]
    if not os.path.isfile(file_name):
        urllib.request.urlretrieve(url, file_name)
    
    data = np.load(file_name)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    return X_train, y_train, X_test, y_test

#%%
if __name__ == '__main__':
    mnist = get_mnist()
    distorted_mnist = get_mnist_distorted()