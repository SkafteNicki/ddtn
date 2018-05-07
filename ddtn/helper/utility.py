#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:39:44 2017

@author: nsde
"""
#%%
try:
    import cPickle as pkl
except:
    import pickle as pkl
import os

#%%
def make_hashable(arr):
    """ Make an array hasable. In this way we can use built-in functions like
        set(...) and intersection(...) on the array
    """
    return tuple([tuple(r.tolist()) for r in arr])

#%%
def load_obj(name):
    """ Function for saving a variable as a pickle file """
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

#%%
def save_obj(obj, name):
    """ Function for loading a pickle file """
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

#%%
def get_path(file):
    """ Get the path of the input file """
    return os.path.realpath(file)

#%%
def get_dir(file):
    """ Get directory of the input file """
    return os.path.dirname(os.path.realpath(file))

#%%
def load_basis():
    """ Load a pre-calculated CPAB basis. See transformer/setup_CPAB_transformer
        for more information on this.
    """
    basis_loc = get_dir(__file__) + '/../cpab_basis'
    try:
        basis = load_obj(basis_loc)
    except:
        raise ValueError('call setup_CPAB.py first')
    return basis
