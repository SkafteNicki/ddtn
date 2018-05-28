# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:57:24 2018

@author: nsde
"""

#%%
import numpy as np
import urllib
import os
from ddtn.helper.utility import get_dir

#%%
def get_mnist():
    """ Downloads mnist from internet """
    url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
    direc = get_dir(__file__)
    file_name = url.split('/')[-1]
    if not os.path.isfile(direc+'/'+file_name):
        print('Downloading the mnist dataset (11.5MB)')
        urllib.request.urlretrieve(url, direc+'/'+file_name)
    
    data = np.load(direc+'/'+file_name)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    return X_train, y_train, X_test, y_test

#%%
def get_mnist_distorted():
    """ Downloads distorted mnist from internet """
    url = "https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz"
    direc = get_dir(__file__)
    file_name = url.split('/')[-1]
    if not os.path.isfile(direc+'/'+file_name):
        print('Downloading the distorted mnist dataset (43MB)')
        urllib.request.urlretrieve(url, direc+'/'+file_name)
    
    data = np.load(direc+'/'+file_name)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    X_train = np.reshape(X_train, (X_train.shape[0], 60, 60, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 60, 60, 1))
    return X_train, y_train, X_test, y_test

#%%
if __name__ == '__main__':
    mnist = get_mnist()
    distorted_mnist = get_mnist_distorted()