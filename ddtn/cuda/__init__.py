# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:34:54 2018

@author: nsde
"""

#%%
from sys import platform as _platform
# This will load the fast cuda version of the CPAB transformer and gradient for
# linux and MAC OS X and load the slower pure tensorflow implemented CPAB 
# transformer for windows
print('Operating system:', _platform)

if _platform == "linux" or _platform == "linux2" or _platform == "darwin": 
   # linux or MAC OS X
   from ddtn.cuda.CPAB_transformer import tf_cuda_CPAB_transformer as tf_CPAB_transformer
   print('Using the fast cuda implementation for CPAB')
elif _platform == "win32" or _platform == "win64":
   # Windows 32 or 64-bit
   from ddtn.cuda.CPAB_transformer import tf_pure_CPAB_transformer as tf_CPAB_transformer
   print('Using the slow pure tf implementation for CPAB')