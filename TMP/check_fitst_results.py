#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:04:02 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))

if os.path.exists('/scratch/jpn5/GAM_Repo/GAM_library/'):
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils/')
else:
    sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'preprocessing_pipeline/util_preproc'))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))

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

import matplotlib.pylab as plt


neuron = 13
cond = 'controlgain'
val = 1.5

with open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_m53s41/fit_results_m53s41_c%d_%s_%.4f.dill'%(neuron,cond,val),'rb') as fh:
    res1 = dill.load(fh)
cond = 'controlgain'    
val = 2.
with open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_m53s41/fit_results_m53s41_c%d_%s_%.4f.dill'%(neuron,cond,val),'rb') as fh:
    res2 = dill.load(fh)


monkeyID = 'm53s41'
gam_res = res1['full']
FLOAT_EPS = np.finfo(float).eps

var_names = gam_res.var_list

pvals = np.clip(gam_res.covariate_significance['p-val'], FLOAT_EPS, np.inf)
dropvar = np.log(pvals) > np.mean(np.log(pvals)) + 1.5 * np.std(np.log(pvals))
dropvar = pvals > 0.001
drop_names = gam_res.covariate_significance['covariate'][dropvar]
fig = plt.figure(figsize=(14,8))
plt.suptitle('%s - neuron %d'%(monkeyID,neuron))
cc = 0
cc_plot =1
ax_dict = {}
for var in np.hstack((var_names)):
    if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
        cc += 1
        continue
    print('plotting var',var)
    
    if var.startswith('neu_'):
        continue

    ax = plt.subplot(5,4,cc_plot)
    ax_dict[var] = ax
    ax.set_title(var)
    sele = gam_res.covariate_significance['p-val'] < 0.001
    if not var in gam_res.covariate_significance['covariate'][sele]:
        cc+=1
        cc_plot+=1
        continue
    if var == 'spike_hist' or var.startswith('t_'):
        pass
    else:        # max_x, min_x = X[var].max(), X[var].min()
        min_x,max_x=res1['xlim'][var]


    if gam_res.smooth_info[var]['is_temporal_kernel']:

        dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
        ord_ =gam_res.smooth_info[var]['ord']
        idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

        impulse = np.zeros(dim_kern)
        impulse[(dim_kern - 1) // 2] = 1
        xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
        fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99,trial_idx=None)
        if var != 'spike_hist':
            xx = xx[idx_select][1:-1]
            fX = fX[idx_select][1:-1]
            fX_p_ci = fX_p_ci[idx_select][1:-1]
            fX_m_ci = fX_m_ci[idx_select][1:-1]
        else:
            xx = xx[:(-ord_ - 1)]
            fX = fX[:(-ord_ - 1)]
            fX_p_ci = fX_p_ci[:(-ord_ - 1)]
            fX_m_ci = fX_m_ci[:(-ord_ - 1)]

    else:
        knots = gam_res.smooth_info[var]['knots']
        knots_sort = np.unique(knots[0])
        knots_sort.sort()
        xx = np.linspace(min_x,max_x,100)

        fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)
    if np.sum(drop_names==var):
        label = var
    else:
        label = var

    if var == 'spike_hist':
        iend = xx.shape[0]//2

        print('set spike_hist')
        fX = fX[iend + 1:][::-1]
        fX_p_ci = fX_p_ci[iend + 1:][::-1]
        fX_m_ci = fX_m_ci[iend + 1:][::-1]
        plt.plot(xx[:iend], fX, ls='-', color='k')
        plt.fill_between(xx[:iend], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
    else:
        plt.plot(xx, fX , ls='-', color='k', label=label)
        plt.fill_between(xx,fX_m_ci,fX_p_ci,color='k',alpha=0.4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.legend()



    cc+=1
    cc_plot+=1
    


gam_res = res2['full']
FLOAT_EPS = np.finfo(float).eps

var_names = gam_res.var_list

pvals = np.clip(gam_res.covariate_significance['p-val'], FLOAT_EPS, np.inf)
dropvar = np.log(pvals) > np.mean(np.log(pvals)) + 1.5 * np.std(np.log(pvals))
dropvar = pvals > 0.001
drop_names = gam_res.covariate_significance['covariate'][dropvar]

cc = 0
cc_plot =1
for var in np.hstack((var_names)):
    if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
        cc += 1
        continue
    print('plotting var',var)
    
    if var.startswith('neu_'):
        
        continue
    if var == 't_reward' and cond == 'reward' and val ==0:
        continue

    sele = res1['full'].covariate_significance['p-val'] < 0.001
    if not var in gam_res.covariate_significance['covariate'][sele]:
        cc+=1
        cc_plot+=1
        continue
   
    if var == 'spike_hist' or var.startswith('t_'):
        pass
    else:        # max_x, min_x = X[var].max(), X[var].min()
        min_x,max_x=res1['xlim'][var]


    if gam_res.smooth_info[var]['is_temporal_kernel']:

        dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
        ord_ =gam_res.smooth_info[var]['ord']
        idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

        impulse = np.zeros(dim_kern)
        impulse[(dim_kern - 1) // 2] = 1
        xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
        fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99,trial_idx=None)
        if var != 'spike_hist':
            xx = xx[idx_select][1:-1]
            fX = fX[idx_select][1:-1]
            fX_p_ci = fX_p_ci[idx_select][1:-1]
            fX_m_ci = fX_m_ci[idx_select][1:-1]
        else:
            xx = xx[:(-ord_ - 1)]
            fX = fX[:(-ord_ - 1)]
            fX_p_ci = fX_p_ci[:(-ord_ - 1)]
            fX_m_ci = fX_m_ci[:(-ord_ - 1)]

    else:
        knots = gam_res.smooth_info[var]['knots']
        knots_sort = np.unique(knots[0])
        knots_sort.sort()
        xx = np.linspace(min_x,max_x,100)

        fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)
    if np.sum(drop_names==var):
        label = var
    else:
        label = var

    if var == 'spike_hist':
        iend = xx.shape[0]//2

        print('set spike_hist')
        fX = fX[iend + 1:][::-1]
        fX_p_ci = fX_p_ci[iend + 1:][::-1]
        fX_m_ci = fX_m_ci[iend + 1:][::-1]
        ax_dict[var].plot(xx[:iend], fX, ls='-', color='r', )
        ax_dict[var].fill_between(xx[:iend], fX_m_ci, fX_p_ci, color='r', alpha=0.4)
    else:
        ax_dict[var].plot(xx, fX , ls='-', color='r', label=label)
        ax_dict[var].fill_between(xx,fX_m_ci,fX_p_ci,color='r',alpha=0.4)

    ax_dict[var].spines['top'].set_visible(False)
    ax_dict[var].spines['right'].set_visible(False)
    # plt.legend()



    cc+=1
    cc_plot+=1
    
plt.tight_layout()
            