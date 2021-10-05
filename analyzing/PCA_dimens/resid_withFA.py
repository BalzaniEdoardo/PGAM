#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:19:50 2021

@author: edoardo
"""
# try dim reductioin using PPCA
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
import scipy.linalg as linalg

sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')

import dill
from utils_loading import unpack_preproc_data
from knots_constructor import *
from PPCA_withStim import em_FA_withStim,whiten,inv_whiten

def get_factor_variance(loadings):
    """
    A helper method to get the factor variances,
    because sometimes we need them even before the
    model is fitted.

    Parameters
    ----------
    loadings : array-like
        The factor loading matrix,
        in whatever state.

    Returns
    -------
    variance : numpy array
        The factor variances.
    proportional_variance : numpy array
        The proportional factor variances.
    cumulative_variances : numpy array
        The cumulative factor variances.
    """
    n_rows = loadings.shape[0]

    # calculate variance
    loadings = loadings ** 2
    variance = np.sum(loadings, axis=0)

    # calculate proportional variance
    proportional_variance = variance / n_rows

    # calculate cumulative variance
    cumulative_variance = np.cumsum(proportional_variance, axis=0)

    return (variance,
            proportional_variance,
            cumulative_variance)
# from numpy.linalg import svd

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d/d_old < tol: break
    return np.dot(Phi, R)

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
cond_value = True
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
cond_type = 'all'

par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']

(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
  trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
  cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
  channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)



# get the unit to include as input covariates
cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate_filter > 0.9
isi_v_filter = isi_v_filter < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

# unit number according to matlab indexing
neuron_keep = np.arange(1, yt.shape[1] + 1)[combine_filter]


cond_knots = None
neuron_fit = neuron_keep
# neuron_fit = neuron_keep[brain_area[combine_filter]==area]


train_trials = np.where(trial_type[cond_type] == cond_value)[0]


# take the train trials
keep = []
for ii in train_trials:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))
    
print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
trial_idx_train = trial_idx[keep]


# fit with coupling
hand_vel_temp = True
sm_handler = smooths_handler()
dict_xlims = {}

var_list = ['rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target',
       'ang_target', 't_move', 't_flyOFF', 't_stop', 't_reward',
       'eye_vert', 'eye_hori' ]


for var in var_list:
    cc = np.where(var_names == var)[0][0]
    x = Xt[keep, cc]


    knots, x_trans, include_var, is_cyclic, order, \
    kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
        knots_cerate(x, var, session, hand_vel_temp=hand_vel_temp, hist_filt_dur='long',
                     exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                     condition=cond_knots)



    if not var.startswith('t_') and var != 'spike_hist':
        if 'lfp' in var:
            dict_xlims[var] = (-np.pi, np.pi)
        else:
            if not knots is None:
                xx0 = max(np.nanpercentile(x_trans, 0), knots[0])
                xx1 = min(np.nanpercentile(x_trans, 100), knots[-1])
            else:
                xx0 = None
                xx1 = None
            dict_xlims[var] = (xx0, xx1)
    else:
        dict_xlims[var] = None

    # print(np.nanmax(np.abs(x_trans)),np.nanmax(np.abs(x_test)))
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                              knots_num=None, perc_out_range=None,
                              is_cyclic=[is_cyclic], lam=50,
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx_train, time_bin=time_bin,
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len,
                              kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                              repeat_extreme_knots=False)
    
niter = 1000
s,idx_dict = sm_handler.get_exog_mat_fast(var_list)
dt=0.006
xdim=60
firing_rate_est = pop_spike_convolve(np.sqrt(yt/dt), trial_idx, h)

# frW = whiten(firing_rate_est)

Cnew, Bnew, sigma2_new, mu, cov, ll_result = em_FA_withStim(firing_rate_est,s,xdim, niter=niter,
              C0=None,B0=None,diagR0=None,
              add_intercept=False, tol=10**-8)




dat_rate = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/PCA_dimens/meanFR_%s.npz'%session,
                    allow_pickle=True)

U,d,Vh = linalg.svd(Cnew)

d.sort()
d = d[::-1]
resid_var = (Cnew.shape[0]-Cnew.shape[1]) * sigma2_new

# tot_var = d.sum() + resid_var
# var_expl_ratio = np.cumsum(d/tot_var)
# plt.plot(var_expl_ratio)

np.savez('FA_dimEstimation_%d.npz'%xdim,C=Cnew,B=Bnew,sigma2=sigma2_new, mu_post=mu,cov_post=cov,
        ll_result=ll_result,modelS=s,fr=firing_rate_est,idx_dict=idx_dict,trial_idx=trial_idx,whitening=False)


pred = np.dot(Cnew,mu.T).T + np.einsum('ij,tj->ti',Bnew,s)

ESS = np.sum((pred - firing_rate_est.mean(axis=0))**2,axis=0)
TSS = np.sum((firing_rate_est - firing_rate_est.mean(axis=0))**2,axis=0)
r2 = ESS/TSS

CVariMax = varimax(Cnew, gamma = 1., q = 20, tol = 1e-6)

model = PCA()
fir_resid = model.fit(np.dot(Cnew,mu.T).T)

model = PCA()
fit_total = model.fit(firing_rate_est)

plt.figure()
plt.title('Variance Explained - Residual regressed using FA')
plt.plot(np.arange(1,fit_total.explained_variance_ratio_.shape[0]+1),
         np.cumsum(fit_total.explained_variance_ratio_),label='tot')
plt.plot(np.arange(1,fir_resid.explained_variance_ratio_.shape[0]+1),
         np.cumsum(fir_resid.explained_variance_ratio_),label='resid')
plt.legend()

plt.savefig('FA_resid_computation.png')


tr = np.unique(trial_idx)[1001]
Bs = np.einsum('ij,tj->ti',Bnew,s)
Cmu = np.dot(Cnew,mu.T).T
idx_srt = np.argsort(r2)[::-1]

plt.figure(figsize=(10,4))
for k in range(4,5):
    plt.subplot(1,3,1)
    bl = trial_idx == tr
    plt.plot(firing_rate_est[bl, 40+k],label='filt rate')
    plt.plot(Cmu[bl, idx_srt[k]],label='latent',color='g')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(firing_rate_est[bl, 40+k],label='filt rate')
    plt.plot(Bs[bl, idx_srt[k]],label='stimulus',color='r')
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(firing_rate_est[bl, 40+k],label='filt rate')
    plt.plot(pred[bl, idx_srt[k]],label='latent+stimulus')
    plt.legend()
plt.tight_layout()
plt.legend()
plt.savefig('example_rate_recon.png')