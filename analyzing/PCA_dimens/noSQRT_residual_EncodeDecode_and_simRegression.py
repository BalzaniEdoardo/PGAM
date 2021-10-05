#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:20:22 2021

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

dat_rate = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/PCA_dimens/meanFR_%s.npz'%session,
                   allow_pickle=True)

pred_fr = dat_rate['meanFr']
pred_fr_noHist = dat_rate['meanFr_noHist']
pred_fr_noInt = dat_rate['meanFr_noInt']
pseudo_r2 = dat_rate['pseudo_r2']
neuron_fit = dat_rate['neuron_fit'][pseudo_r2>0.01]
pred_fr_noHist = pred_fr_noHist[:,pseudo_r2>0.01]
pred_fr_noInt  = pred_fr_noInt[:,pseudo_r2>0.01]
pred_fr = pred_fr[:,pseudo_r2>0.01]
pseudo_r2 = pseudo_r2[pseudo_r2>0.01]


yt = yt[:,neuron_fit-1]




# residual computation, attempt A:
# 1) square root of spikes and predicted rates (GAM)
# 2) filter smooth
#   3.1) compute the subtract rates and check with PCA
#   3.2) compute projection weights using GAMs, keep 90% variance,
#   3.3) encode + decode smoothed spikes and subtract from the activity

dt = 0.006
firing_rate_est = pop_spike_convolve(yt, trial_idx, h)/dt
pred_rate_sm = pop_spike_convolve(pred_fr, trial_idx, h)/dt
pred_rate_noHist_sm = pop_spike_convolve(np.sqrt(pred_fr_noHist), trial_idx, h)/dt
pred_rate_noInt_sm = pop_spike_convolve(np.sqrt(pred_fr_noInt), trial_idx, h)/dt



## use the FULL GAM prediction to remove variance
model_gam = PCA()
fit_gam = model_gam.fit(pred_rate_sm)

num_dim = np.where(np.cumsum(fit_gam.explained_variance_ratio_) > 0.9)[0][0]


model_gam_red = PCA(num_dim)
fit_gam_red = model_gam_red.fit(pred_rate_sm)


# subtract the GAM PC encoded from true activity
resid_activity = firing_rate_est - fit_gam_red.inverse_transform(fit_gam_red.transform(firing_rate_est))

# pca resid
model_resid = PCA()
fit_resid = model_resid.fit(resid_activity)

# pca total
model_total = PCA()
fit_total = model_total.fit(firing_rate_est)

# residual from enc/dec
plt.figure(figsize=((10,5)))
plt.subplot(121)
plt.plot(np.arange(1,yt.shape[1]+1), np.cumsum(fit_total.explained_variance_ratio_),'-k',label='total')
plt.plot(np.arange(1,yt.shape[1]+1), np.cumsum(fit_resid.explained_variance_ratio_),'-r',label='resid')


plt.legend()
plt.title('Residual by Encode->Decode')
plt.xlabel('PCs')
plt.ylabel('explained variance')


## RESID on simple differences
model_diff = PCA()
fit_diff = model_diff.fit(firing_rate_est-pred_rate_sm)

plt.subplot(122)
plt.plot(np.arange(1,yt.shape[1]+1), np.cumsum(fit_total.explained_variance_ratio_),'-k',label='total')
plt.plot(np.arange(1,yt.shape[1]+1), np.cumsum(fit_diff.explained_variance_ratio_),'-r',label='resid')


plt.legend()
plt.title('Residual based on delta rates')
plt.xlabel('PCs')
plt.ylabel('explained variance')

plt.tight_layout()

plt.savefig('noSQRT_resid_compute_twoMethods.png')


plt.figure(figsize=(10,6))
unq_trial = np.unique(trial_idx)
sel = np.where(pseudo_r2 > 0.1)[0]
tr = unq_trial[1100]

for un in range(6):  
    un_id = sel[un]
    plt.subplot(2,3,un+1)
    time = 6*np.arange((trial_idx==tr).sum())
    plt.plot(time,firing_rate_est[trial_idx==tr,un_id],label='transf rate')
    plt.plot(time,pred_rate_sm[trial_idx==tr,un_id],label='gam pred')
    plt.xlabel('time [ms]')
    plt.legend()
plt.tight_layout()
plt.savefig('noSQRT_ratePred.png')

