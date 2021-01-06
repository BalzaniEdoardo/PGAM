#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:30:36 2020

@author: edoardo
"""

import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(thisPath)),'GAM_Library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import matplotlib.pylab as plt

dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET/PPC+PFC/m53s47.npz',allow_pickle=True)
concat = dat['data_concat'].all()

Y = concat['Yt']
X = concat['Xt']
all_index = concat['trial_idx']


info_trial = dat['info_trial'].all()
active = info_trial.trial_type['replay'] == 1
passive = info_trial.trial_type['replay'] == 0
all_tr = info_trial.trial_type['all']
var_names = dat['var_names']

# passive extract
idx_subselect = np.where(passive)[0]
keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(all_index == ii)[0]))
    
    
keep = np.array(keep,dtype=int)

Y_passive = Y[keep]
X_passive = X[keep]

# active extract
idx_subselect = np.where(all_tr*active)[0]
keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(all_index == ii)[0]))
    
    
keep = np.array(keep,dtype=int)

Y_active = Y[keep]
X_active = X[keep]

# compute histogram passive
flyOff = np.where(X_passive[:,var_names=='t_flyOFF'])[0]

time = np.arange(0,151)*6 - 50*6 
hist_passive = np.zeros((151,Y_passive.shape[1]))
for idx in flyOff:
    add = Y_passive[idx-100:idx+51,:]
    if add.shape == hist_passive.shape:
        hist_passive = hist_passive + add
    
    
flyOff = np.where(X_active[:,var_names=='t_flyOFF'])[0]

# time = np.arange(0,151)*6 - 50*6 
hist_active = np.zeros((151,Y_active.shape[1]))
for idx in flyOff:
    add = Y_active[idx-100:idx+51,:]
    if add.shape == hist_active.shape:
        hist_active = hist_active + add


plt.figure(figsize=(10,8))
plt.suptitle('test PSTH')
for k in range(36):
    plt.subplot(6,6,k+1)
    plt.plot(time, hist_active[:,k],label='active')
    plt.plot(time, hist_passive[:,k],label='passive')
    
    mn = np.min([hist_active[:,k].min(),hist_passive[:,k].min()])
    mx = np.max([hist_active[:,k].max(),hist_passive[:,k].max()])
    
    
    plt.plot([0,0],[mn,mx],'--k')
    plt.plot([300,300],[mn,mx],'--r')

    plt.xticks([])
    plt.yticks([])
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure(figsize=(10,8))
plt.suptitle('test PSTH 2')
for k in range(36):
    plt.subplot(6,6,k+1)
    plt.plot(time, hist_active[:,36+k],label='active')
    plt.plot(time, hist_passive[:,36+k],label='passive')
    
    mn = np.min([hist_active[:,36+k].min(),hist_passive[:,36+k].min()])
    mx = np.max([hist_active[:,36+k].max(),hist_passive[:,36+k].max()])
    
    
    plt.plot([0,0],[mn,mx],'--k')
    plt.plot([300,300],[mn,mx],'--r')

    plt.xticks([])
    plt.yticks([])
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# compute histogram passive
reward = np.where(X_passive[:,var_names=='t_stop'])[0]

time = np.arange(0,101)*6 - 50*6
hist_passive = np.zeros((101,Y_passive.shape[1]))
for idx in reward:
    add = Y_passive[idx-50:idx+51,:]
    if add.shape == hist_passive.shape:
        hist_passive = hist_passive + add
    
    
reward = np.where(X_active[:,var_names=='t_stop'])[0]

# time = np.arange(0,151)*6 - 50*6 
hist_active = np.zeros((101,Y_active.shape[1]))
for idx in reward:
    add = Y_active[idx-50:idx+51,:]
    if add.shape == hist_active.shape:
        hist_active = hist_active + add


plt.figure(figsize=(10,8))
plt.suptitle('test PSTH')
for k in range(36):
    plt.subplot(6,6,k+1)
    plt.plot(time, hist_active[:,k],label='active')
    plt.plot(time, hist_passive[:,k],label='passive')
    
    mn = np.min([hist_active[:,k].min(),hist_passive[:,k].min()])
    mx = np.max([hist_active[:,k].max(),hist_passive[:,k].max()])
    
    
    plt.plot([0,0],[mn,mx],'--k')
    plt.plot([300,300],[mn,mx],'--r')

    plt.xticks([])
    plt.yticks([])
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.figure(figsize=(10,8))
plt.suptitle('test PSTH 2')
for k in range(36):
    plt.subplot(6,6,k+1)
    plt.plot(time, hist_active[:,36+k],label='active')
    plt.plot(time, hist_passive[:,36+k],label='passive')
    
    mn = np.min([hist_active[:,36+k].min(),hist_passive[:,36+k].min()])
    mx = np.max([hist_active[:,36+k].max(),hist_passive[:,36+k].max()])
    
    
    plt.plot([0,0],[mn,mx],'--k')
    plt.plot([300,300],[mn,mx],'--r')

    plt.xticks([])
    plt.yticks([])
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

