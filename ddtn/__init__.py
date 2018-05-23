# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:34:54 2018

@author: nsde
"""

from sys import platform as _platform
from ddtn.helper.utility import check_for_gpu, check_cuda_support
# This will load the fast cuda version of the CPAB transformer and gradient for
# linux and MAC OS X and load the slower pure tensorflow implemented CPAB 
# transformer for windows

print(50*'-')
print('Operating system:', _platform)

gpu = check_for_gpu() and check_cuda_support()
if (_platform == "linux" or _platform == "linux2" \
    or _platform == "darwin") and gpu: 
   # linux or MAC OS X
   from ddtn.cuda.CPAB_transformer import tf_cuda_CPAB_transformer as tf_CPAB_transformer
   print('Using the fast cuda implementation for CPAB')
else:
   # Windows 32 or 64-bit or no GPU
   from ddtn.cuda.CPAB_transformer import tf_pure_CPAB_transformer as tf_CPAB_transformer
   print('Using the slow pure tensorflow implementation for CPAB')
print(50*'-')

# Import directories
from . import helper
from . import transformers
from . import cuda
from . import data