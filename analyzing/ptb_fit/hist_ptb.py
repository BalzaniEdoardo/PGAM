#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:35:58 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
plt.close('all')
dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s51.npz',
        allow_pickle=True)

concat = dat['data_concat']
var_names = dat['var_names']

unit_info = dat['unit_info'].all()
brain_area = unit_info['brain_area']
pert_idx = concat.all()['Xt'][:,var_names=='t_ptb']

yt = concat.all()['Yt']

idx = np.where(pert_idx)[0]

matrix_unit = np.zeros((yt.shape[1],idx.shape[0],201))

cnt = 0
for ii in idx:
    matrix_unit[:,cnt, :] = yt[ii:ii+201,:].T
    cnt+=1

mean_counts = matrix_unit.mean(axis=1)/0.006

plt.figure(figsize=(10,8))

time = (np.arange(201) )*6

k = 0
for unt in range(mean_counts.shape[0]):
    if k==25:
        plt.tight_layout() 
        plt.figure(figsize=(10,8))
        k=0
        
    plt.subplot(5,5,k+1)
    plt.title('%s - unit %d'%(brain_area[unt],unt+1))
    plt.plot(time,mean_counts[unt])
    # plt.vlines(0,0,mean_counts[unt].max())
    
    k+=1
    
plt.tight_layout() 