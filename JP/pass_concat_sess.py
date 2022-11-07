#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:09:32 2022

@author: edoardo
"""

import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))


if os.path.exists('/scratch/eb162/GAM_Repo/GAM_library/'):
    sys.path.append('/scratch/eb162/GAM_Repo/GAM_library/')
    #sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')
    sys.path.append('/scratch/eb162/GAM_Repo/firefly_utils/')
    sys.path.append('/scratch/eb162/pathtopostproc')

else:
    gam_lib='/Users/edoardo/Work/Code/GAM_code/'
    sys.path.append(os.path.join(os.path.dirname(gam_lib),'GAM_library'))
    sys.path.append(os.path.join(os.path.dirname(gam_lib),'preprocessing_pipeline/util_preproc'))
    sys.path.append(os.path.join(os.path.dirname(gam_lib),'firefly_utils'))
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/JP/MOUSE_FF/code/')

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import scipy.stats as sts
from copy import deepcopy
from knots_constructor import *
from path_class import get_paths_class
from scipy.io import loadmat
from utils_loading import unpack_preproc_data
import dill
from postprocess_utils import *

import matplotlib.pylab as plt
from scipy.io import savemat


sess_copy = ['m53s51','m53s48','m53s43','m44s218','m44s183']

session = 'm53s51'
neuron = 118 # matlab index

gam_file = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_all_1.0000.dill'
gam_file = gam_file%(session, session, neuron)

fhName = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']
(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
  trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
  cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
  channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)



for session in sess_copy:

    gam_file = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_all_1.0000.dill'
    gam_file = gam_file%(session, session, neuron)
    
    fhName = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
    par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
            'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
            'unit_type','channel_id','electrode_id','cluster_id']
    (Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
      trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
      cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
      channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)
    break
    # savemat('/Volumes/server/Projects/Firefly-PGAM/concat_data/concat_%s.mat'%session,mdict={'X':Xt,'spk':yt,'trial_idx':trial_idx,'var_names':var_names})
plt.close('all')
for neuron in range(1,101):
    if neuron!=68:
        continue
    plt.figure()
    
    plt.title('%s - neu: %d'%(session,neuron))
    targ_vals = np.linspace(-np.pi,np.pi,16)
    xx = lfp_beta[:, neuron-1]
    fr = np.zeros(targ_vals.shape[0]-1)
    yt_neu = yt[:, neuron-1]
    cc = 0
    for ii in range(targ_vals.shape[0]-1):
        r0,r1 = targ_vals[ii], targ_vals[ii+1]
        sel = (xx >= r0) & (xx<r1)
        fr[cc] = yt_neu[sel.flatten()].mean()/0.006
        cc+=1
        
    plt.plot(0.5*(targ_vals[1:]+targ_vals[:-1]), fr)
    
neuron = 14
plt.close('all')
kk=1
for trial in np.unique(trial_idx)[:10]:
    plt.subplot(2,5,kk)
    sel = trial_idx==trial
    xx = lfp_beta[:, neuron-1]
    
    y_neu = yt[sel]
    y_neu = y_neu[:,neuron-1]
    
    time = np.arange(sel.sum())
    plt.plot(time, xx[sel])
    idx_spk = np.where(y_neu>0)[0]
    
    plt.plot(time[idx_spk],xx[sel][idx_spk],'ok')
    kk+=1
    
    


