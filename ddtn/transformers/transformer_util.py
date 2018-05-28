# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:13:26 2018

@author: nsde
"""
#%%
import numpy as np
from ddtn.helper.utility import load_basis
from ddtn.helper.math import create_grid

from ddtn.transformers.transformers import tf_Affine_transformer
from ddtn.transformers.transformers import tf_Affinediffeo_transformer
from ddtn.transformers.transformers import tf_Homografy_transformer
from ddtn.transformers.transformers import tf_CPAB_transformer
from ddtn.transformers.transformers import tf_TPS_transformer

from ddtn.transformers.transformer_layers import ST_Affine_transformer
from ddtn.transformers.transformer_layers import ST_Affinediffeo_transformer
from ddtn.transformers.transformer_layers import ST_Homografy_transformer
from ddtn.transformers.transformer_layers import ST_CPAB_transformer
from ddtn.transformers.transformer_layers import ST_TPS_transformer

from ddtn.transformers.keras_layers import SpatialAffineLayer
from ddtn.transformers.keras_layers import SpatialAffineDiffeoLayer
from ddtn.transformers.keras_layers import SpatialHomografyLayer
from ddtn.transformers.keras_layers import SpatialCPABLayer
from ddtn.transformers.keras_layers import SpatialTPSLayer

#%% 
def get_transformer(transformer_name='affine'):
    """ Returns the transformer layer for a given name """
    lookup = {'affine': tf_Affine_transformer,
              'affinediffeo': tf_Affinediffeo_transformer,
              'homografy': tf_Homografy_transformer,
              'CPAB': tf_CPAB_transformer,
              'TPS': tf_TPS_transformer
             }
    assert (transformer_name in lookup), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in lookup.keys()])
    return lookup[transformer_name]

#%%
def get_transformer_layer(transformer_name='affine'):
    """ Returns the transformer layer for a given name """
    lookup = {'affine': ST_Affine_transformer,
              'affinediffeo': ST_Affinediffeo_transformer,
              'homografy': ST_Homografy_transformer,
              'CPAB': ST_CPAB_transformer,
              'TPS': ST_TPS_transformer
             }
    assert (transformer_name in lookup), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in lookup.keys()])
    return lookup[transformer_name]

#%%
def get_keras_layer(transformer_name='affine'):
    """ Returns the keras layer for a given name """
    lookup = {'affine': SpatialAffineLayer,
              'affinediffeo': SpatialAffineDiffeoLayer,
              'homografy': SpatialHomografyLayer,
              'CPAB': SpatialCPABLayer,
              'TPS': SpatialTPSLayer
             }
    assert (transformer_name in lookup), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in lookup.keys()])
    return lookup[transformer_name]

#%%
def get_transformer_dim(transformer_name='affine'):
    """ Returns the size of parametrization for a given transformer """
    lookup = {'affine': 6,
              'affinediffeo': 6,
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
              'affinediffeo': np.zeros((n_units, dim), dtype=np.float32),
              'homografy': np.zeros((n_units, dim), dtype=np.float32),
              'CPAB': np.zeros((n_units, dim), dtype=np.float32),
              'TPS': np.zeros((n_units, dim), dtype=np.float32)}
    
    bias = {'affine': np.array([1,0,0,0,1,0], dtype=np.float32),
            'affinediffeo': np.zeros((dim,), dtype=np.float32),
            'homografy': np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32),
            'CPAB': np.zeros((dim,), dtype=np.float32),
            'TPS': create_grid([-1,-1],[1,1],[4,4]).T.flatten()}
    
    return (kernel[transformer_name], bias[transformer_name])

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
        
    elif transformer_name == 'affinediffeo':
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

#%%
def format_theta(theta, transformer_name):
    if transformer_name == 'affine' or transformer_name == 'affinediffeo':
        theta = np.reshape(theta, (-1, 2, 3))
    elif transformer_name == 'homografy':
        theta = np.reshape(theta, (-1, 3, 3))
    elif transformer_name == 'TPS':
        theta = np.reshape(theta, (-1, 16, 2))
    return theta

#%%
if __name__ == '__main__':
    pass
