# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:13:26 2018

@author: nsde
"""
#%%
import numpy as np
from ddtn.helper.utility import load_basis
from ddtn.helper.math import create_grid
from ddtn.transformers.transformer_layers import ST_Affine_transformer
from ddtn.transformers.transformer_layers import ST_Affine_diffio_transformer
from ddtn.transformers.transformer_layers import ST_Homografy_transformer
from ddtn.transformers.transformer_layers import ST_CPAB_transformer
from ddtn.transformers.transformer_layers import ST_TPS_transformer

#%%
def get_transformer(transformer_name='affine'):
    """ Returns the transformer for a given name """
    lookup = {'affine': ST_Affine_transformer,
              'affine_diffeo': ST_Affine_diffio_transformer,
              'homografy': ST_Homografy_transformer,
              'CPAB': ST_CPAB_transformer,
              'TPS': ST_TPS_transformer
             }
    assert (transformer_name in lookup), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in lookup.keys()])
    return lookup[transformer_name]

#%%
def get_transformer_dim(transformer_name='affine'):
    """ Returns the size of parametrization for a given transformer """
    lookup = {'affine': 6,
              'affine_diffeo': 6,
              'homografy': 9,
              'CPAB': load_basis()['d'],
              'TPS': 32
             }
    assert (transformer_name in lookup), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in lookup.keys()])
    return lookup[transformer_name]

#%%
def get_transformer_init_weights(n_units, transformer_name='affine'):
    """ Get weights that initialize a given transformer to the identity transformation """
    dim = get_transformer_dim(transformer_name)
    kernel = {'affine': np.zeros((n_units, dim), dtype=np.float32),
              'affine_diffeo': np.zeros((n_units, dim), dtype=np.float32),
              'homografy': np.zeros((n_units, dim), dtype=np.float32),
              'CPAB': np.zeros((n_units, dim), dtype=np.float32),
              'TPS': np.zeros((n_units, dim), dtype=np.float32)}
    
    bias = {'affine': np.array([1,0,0,0,1,0], dtype=np.float32),
            'affine_diffeo': np.zeros((dim,), dtype=np.float32),
            'homografy': np.array([1,0,0,0,1,0,0,0,0], dtype=np.float32),
            'CPAB': np.zeros((dim,), dtype=np.float32),
            'TPS': create_grid([-1,-1],[1,1],[4,4]).T.flatten()}
    
    return [kernel, bias]

#%%
def get_random_theta(N, transformer_name='affine'):
    """ Samples N random samples from a given transformation family """
    dim = get_transformer_dim(transformer_name)
    
    if transformer_name == 'affine':
        theta = np.zeros((N, dim))
        theta[:,0] = np.abs(np.random.normal(loc=1, scale=0.2, size=N))
        theta[:,4] = np.abs(np.random.normal(loc=1, scale=0.2, size=N))
        theta[:,1] = theta[:,3] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,2] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,5] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        
    elif transformer_name == 'affine_diffeo':
        theta = np.zeros((N, dim))
        theta[:,0] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,4] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,1] = theta[:,3] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,2] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,5] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
    
    elif transformer_name == 'homografy':
        theta = np.zeros((N, dim))
        theta[:,0] = np.random.normal(loc=1, scale=0.2, size=N)
        theta[:,4] = np.random.normal(loc=1, scale=0.2, size=N)
        theta[:,1] = theta[:,3] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,2] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,5] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,6] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,7] = np.abs(np.random.normal(loc=0, scale=0.2, size=N))
        theta[:,8] = np.ones((N,))
        
    elif transformer_name == 'CPAB':
        theta = 0.5*np.random.normal(size=(N, dim))
    
    elif transformer_name == 'TPS':
        x,y=np.meshgrid(np.linspace(-1,1,4), np.linspace(-1,1,4))
        points = np.concatenate((x.reshape((1,-1)),y.reshape((1,-1))), axis=0)
        theta = np.tile(points.T.reshape(1,-1), (N, 1)) + 0.1*np.random.normal(size=(N, dim))
    
    return theta    
