#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:18:21 2018

@author: nsde
"""

#%%
import os

#%%
ne = '-ne 50 '
bs = '-bs 100 '
lr = '-lr 1e-5 '

#%%
os.system("PYTHONPATH='/home/nsde/Documents/ddtn' python mnist_classifier.py -tt no "
          + ne + bs + lr)
os.system("PYTHONPATH='/home/nsde/Documents/ddtn' python mnist_classifier.py -tt affine "
          + ne + bs + lr)
os.system("PYTHONPATH='/home/nsde/Documents/ddtn' python mnist_classifier.py -tt affinediffeo "
          + ne + bs + lr)
os.system("PYTHONPATH='/home/nsde/Documents/ddtn' python mnist_classifier.py -tt homografy "
          + ne + bs + lr)
os.system("PYTHONPATH='/home/nsde/Documents/ddtn' python mnist_classifier.py -tt TPS "
          + ne + bs + lr)
os.system("PYTHONPATH='/home/nsde/Documents/ddtn' python mnist_classifier.py -tt CPAB "
          + ne + bs + lr)