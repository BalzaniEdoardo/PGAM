#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:23:46 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt

dat = np.load('coupling_info.npy')

dtype_dict = {
    'names':('session','sender area','receiver area','sender id','receiver id',
             'cond type','value A','value B','value C','sign A','sign B','sign C'),
    'formats':('U30','U30','U30',int,int,'U30',float,float,float,bool,bool,bool)
    }