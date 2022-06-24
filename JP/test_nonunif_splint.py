#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:27:05 2022

@author: edoardo
"""
import numpy as np
import os,sys
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library/')
from gam_data_handlers import splineDesign
import matplotlib.pylab as plt
from copy import deepcopy
import seaborn
plt.close('all')

def create_diff_pen(knots,order):
    assert(all(knots[:order] == knots[0]))
    assert(all(knots[-order:] == knots[-1]))
    if order == 1:
        int_knots = knots
    else:
        int_knots = knots[order-1: -(order-1)]

    Ak =  splineDesign(knots, int_knots, ord=order, der=0, outer_ok=False)
    Amid = splineDesign(knots, int_knots[:-2] + int_knots[2:] - int_knots[1:-1], ord=order, der=0, outer_ok=False)
    
    
    B = np.zeros((Ak.shape[0]-2,Ak.shape[1]))
    
    for i in range(Ak.shape[1]):
        B[:,i] = (Ak[:-2, i] - Ak[1:-1, i] - Amid[:, i] + Ak[2:, i])
    
    S = np.dot(B.T,B)
    return S,B

knots = np.hstack(([0], np.sort(np.random.uniform(0,10,40)),[10]))
# knots = np.linspace(0,10,40)
order = 1
knots = np.hstack(([knots[0]]*(order-1), knots,[knots[-1]]*(order-1)))

S,B = create_diff_pen(knots,order)
seaborn.heatmap(S)

