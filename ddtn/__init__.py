# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:34:54 2018

@author: nsde
"""

# Import directories
from . import cuda
from . import data
from . import helper
from . import sampling
from . import transformers

#%%
from sys import platform as _platform
from ddtn.helper.utility import gpu_support
# Tells the user what kind of implementation will be used for the
# CPAB transformations

print(70*'-')
print('Operating system:', _platform)

if (_platform == "linux" or _platform == "linux2" \
    or _platform == "darwin") and gpu_support(): 
   print('Using the fast cuda implementation for CPAB')
else:
   # Windows 32 or 64-bit or no GPU
   print('Using the slow pure tensorflow implementation for CPAB')
print(70*'-')