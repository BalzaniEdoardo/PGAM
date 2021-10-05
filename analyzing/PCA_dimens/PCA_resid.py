#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:42:17 2021

@author: edoardo
"""
import numpy as np
import sys,os,dill
#sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
#sys.path.append('/scratch/eb162/GAM_library/')
main_dir = '/Users/edoardo/Work/Code/GAM_code/'
# sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
# sys.path.append('/scratch/eb162/GAM_library/')
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(main_dir,'GAM_library'))
from GAM_library import *
from data_handler import *
from spike_times_class import spike_counts
from behav_class import behavior_experiment,load_trial_types
from lfp_class import lfp_class
from copy import deepcopy
from scipy.io import loadmat,savemat
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PIL import ImageColor
from sklearn.decomposition import PCA

def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        sm_x[sel] = np.convolve(x[sel],filter,mode='same')
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk


filtwidth = 15
t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)


session = 'm53s113'
file_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'
dat = np.load(os.path.join(file_fld,session+'.npz'),allow_pickle=True)

concat = dat['data_concat'].all()
trial_idx = concat['trial_idx']


yt = concat['Yt']
X = concat['Xt']
var_names = dat['var_names']

dat_rate = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/PCA_dimens/meanRate/meanFR_%s.npz'%session,
                   allow_pickle=True)

pred_fr = dat_rate['meanFr']
pseudo_r2 = dat_rate['pseudo_r2']
neuron_fit = dat_rate['neuron_fit'][pseudo_r2>0.01]
pred_fr = pred_fr[:,pseudo_r2>0.01]
pseudo_r2 = pseudo_r2[pseudo_r2>0.01]

yt = yt[:,neuron_fit-1]

dt = 0.006
firing_rate_est = pop_spike_convolve(yt, trial_idx, h)/dt
pred_rate_sm = pop_spike_convolve(pred_fr, trial_idx, h)/dt

model_total = PCA()
fit_total = model_total.fit(firing_rate_est)

model_resid = PCA()
fit_resid = model_resid.fit(pred_rate_sm-firing_rate_est)

model_gam = PCA()
fit_gam = model_gam.fit(pred_rate_sm)

plt.figure()
plt.title('dimensionality reduction')
plt.plot(np.arange(1,yt.shape[1]+1),np.cumsum(fit_total.explained_variance_ratio_),label='total')
plt.plot(np.arange(1,yt.shape[1]+1),np.cumsum(fit_resid.explained_variance_ratio_),label='residual')
plt.plot(np.arange(1,yt.shape[1]+1),np.cumsum(fit_gam.explained_variance_ratio_),label='gam model')

plt.plot([1,62],[0.9,0.9],'k')

i_tot = (np.cumsum(fit_total.explained_variance_ratio_) <= 0.9).sum()
i_resid = (np.cumsum(fit_resid.explained_variance_ratio_) <= 0.9).sum()

plt.plot([i_tot,i_tot],[0,np.cumsum(fit_total.explained_variance_ratio_)[i_tot-1]],'k')
plt.plot([i_resid,i_resid],[0,np.cumsum(fit_resid.explained_variance_ratio_)[i_resid-1]],'k')

plt.legend(loc=4)
plt.title('PCA GAM residuals')
plt.xlabel('PCs')
plt.ylabel('expl. variance')
plt.ylim(0,1.05)
plt.savefig('residual_pca.png')


plt.figure()

unq_trials = np.unique(trial_idx)
sel = pseudo_r2 > 0.1
for k in range(5):
    plt.subplot(2,3,k+1)
    ii = np.where(sel)[0][k]
    fr_tr = firing_rate_est[trial_idx==unq_trials[120], :]
    fr_tr = fr_tr[:,sel]
    
    
    pred_tr = pred_rate_sm[trial_idx==unq_trials[120], :]
    pred_tr = pred_tr[:,sel]
    plt.plot(fr_tr[:,ii])
    plt.plot(pred_tr[:,ii])
plt.tight_layout()
    

plt.figure()
unq_trials = np.unique(trial_idx)
sel = pseudo_r2 > 0.05
for k in range(5):
    plt.subplot(2,3,k+1)
    ii = np.where(sel)[0][k]
    pr2 = pseudo_r2[sel]
    
    fr_tr = firing_rate_est[trial_idx==unq_trials[1001], :]
    fr_tr = fr_tr[:,sel]
    
    y_tr = yt[trial_idx==unq_trials[1001], :]
    y_tr = y_tr[:, sel]
    pred_tr = pred_rate_sm[trial_idx==unq_trials[1001], :]
    pred_tr = pred_tr[:,sel]
    plt.plot(y_tr[:,ii],'k')
    plt.plot(fr_tr[:,ii])
    plt.plot(pred_tr[:,ii])
    plt.title('%.3f'%pr2[ii])
plt.tight_layout()

plt.figure()

unq_trials = np.unique(trial_idx)
sel = pseudo_r2 > 0.05
for k in range(5):
    plt.subplot(2,3,k+1)
    ii = np.where(sel)[0][k]
    pr2 = pseudo_r2[sel]
    
    fr_tr = firing_rate_est[trial_idx==unq_trials[1001], :]
    fr_tr = fr_tr[:,sel]
    
    y_tr = yt[trial_idx==unq_trials[1001], :]
    y_tr = y_tr[:, sel]
    pred_tr = pred_fr[trial_idx==unq_trials[1001], :]
    pred_tr = pred_tr[:,sel]
    plt.plot(y_tr[:,ii],'k')
    # plt.plot(fr_tr[:,ii])
    plt.plot(pred_tr[:,ii]/dt)
    plt.title('%.3f'%pr2[ii])
plt.tight_layout()