#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:56:20 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import dill
import pandas as pd
import scipy.stats as sts
import scipy.linalg as linalg
from time import perf_counter
from seaborn import heatmap
from path_class import get_paths_class
from knots_constructor import knots_cerate
from copy import deepcopy
from time import sleep
from spline_basis_toolbox import *
from bisect import bisect_left
from statsmodels.distributions import ECDF
import venn
plt.close('all')
condition = 'ptb'
session = 'm53s36'
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_change/significance_tuning_function_change/newTest_%s_%s_tuningChange.npz'%(session,condition),
        allow_pickle=True)

tensor_A = dat['tensor_A']
tensor_B = dat['tensor_B']
index_dict_A = dat['index_dict_A'].all()
index_dict_B = dat['index_dict_B'].all()


var_sign = dat['var_sign']
unit_list = dat['unit_list']
variable = 't_stop'
pv_th = 0.01



sbcnt = 1
for unit in range(tensor_A.shape[0]):
    if unit % 20 == 0:
        if unit != 0:
           plt.tight_layout() 
        sbcnt = 1
        plt.figure(figsize=(10,8))
    plt.subplot(5,4,sbcnt)
    
    neuron_id = unit_list[unit]
    filt = (var_sign['variable'] == variable) * (var_sign['unit'] == neuron_id)
    pval = var_sign[filt]['p-val']
    
    plt.title('Unit %d - %f'%(neuron_id, pval[0]))
    plt.plot(tensor_A[unit,0,index_dict_A[variable]])
    plt.fill_between(range(index_dict_A[variable].shape[0]), tensor_A[unit,1,index_dict_A[variable]],
                     tensor_A[unit,2,index_dict_A[variable]],alpha=0.4)
    
    plt.plot(tensor_B[unit,0,index_dict_B[variable]])
    plt.fill_between(range(index_dict_B[variable].shape[0]), tensor_B[unit,1,index_dict_B[variable]],
                     tensor_B[unit,2,index_dict_B[variable]],alpha=0.4)
    sbcnt+=1
plt.tight_layout()



# Filter significant
volume_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'

dict_xlim = {'rad_vel':(0.,200),
             'ang_vel':(-60,60),
             'rad_path':(0,300),
             'ang_path':(-60,60),
             'rad_target':(5,330),
             'ang_target':(-35,35),
             'hand_vel1':(-100., 100),
             'hand_vel2':(-100,100),
             'phase':(-np.pi,np.pi),
             'rad_acc':(-800,800),
             'ang_acc':(-100,100),
             
            'lfp_alpha':(-np.pi,np.pi),
            'lfp_beta':(-np.pi,np.pi),
            'lfp_theta':(-np.pi,np.pi),
             't_move':(-0.36,0.36),
             't_flyOFF':(-0.36,0.36),
             't_stop':(-0.36,0.36),
             't_reward':(-0.36,0.36),'eye_vert':(-2,2),
             'eye_hori':(-2,2)}

first = True

# filt = (var_sign['variable'] == variable) * (var_sign['p-val'] < pv_th)

sbcnt = 1
oldsub =-1
for unit in range(tensor_A.shape[0]):

    if sbcnt%20 == 1:
        if unit != 0:
            plt.tight_layout()
        sbcnt = 1
        if oldsub != sbcnt:
            oldsub = sbcnt
            plt.figure(figsize=(10, 8))
            sb_open = []
    if not sbcnt in sb_open:
        plt.subplot(5, 4, sbcnt)
        oldsub = sbcnt
        
        sb_open += [ sbcnt]
    
    print(sb_open)

    neuron_id = unit_list[unit]
    

    filt = (var_sign['variable'] == variable) * (var_sign['unit'] == neuron_id)
    pval = var_sign[filt]['p-val']
    # check_tuned = True
    cond_list = []
    cnt_tun=0
    for key in var_sign.dtype.names:
        if 'p-val ' in key:
            cond_list += [float(key.split(' ')[2])]
            if var_sign[filt][key] < 0.001:
                cnt_tun += 1
                # check_tuned = False
    check_tuned = cnt_tun != 0
    if check_tuned:
        
        val = cond_list[0]
        with open(volume_fld+'/gam_%s/fit_results_%s_c%d_%s_%.4f.dill'% (session,session,neuron_id,condition,val), "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)
            gam_res_A = deepcopy(gam_res_dict['full'])
            
        val = cond_list[1]
        with open(volume_fld+'/gam_%s/fit_results_%s_c%d_%s_%.4f.dill'% (session,session,neuron_id,condition,val), "rb") as dill_file:
            gam_res_dict = dill.load(dill_file)
            gam_res_B = gam_res_dict['full']
            
        if gam_res_A.smooth_info[variable]['is_temporal_kernel']:
            dim_kern = gam_res_A.smooth_info[variable]['basis_kernel'].shape[0]
            knots_num = gam_res_A.smooth_info[variable]['knots'][0].shape[0]

            idx_select = np.arange(0,dim_kern,(dim_kern+1)//knots_num)

            impulse = np.zeros(dim_kern)
            impulse[(dim_kern-1)//2] = 1
            x = 0.006*np.linspace(-(dim_kern+1)/2,(dim_kern-1)/2,dim_kern)
            fX_A, fX_p_ci_A, fX_m_ci_A = gam_res_A.smooth_compute([impulse], variable, perc=0.99,)
            fX_B, fX_p_ci_B, fX_m_ci_B = gam_res_B.smooth_compute([impulse], variable, perc=0.99,)
            
        else:
            x = np.linspace(dict_xlim[variable][0], dict_xlim[variable][1], 100)
            fX_A, fX_p_ci_A, fX_m_ci_A = gam_res_A.smooth_compute([x], variable, perc=0.99,)
            fX_B, fX_p_ci_B, fX_m_ci_B = gam_res_B.smooth_compute([x], variable, perc=0.99,)
        ba = gam_res_dict['brain_area']
            
        fX_B = fX_B - np.nanmean(fX_B-fX_A)
        fX_m_ci_B = fX_m_ci_B - np.nanmean(fX_B-fX_A)
        fX_B = fX_B - np.nanmean(fX_B-fX_A)
                # knots, x_trans, include_var, is_cyclic, order, \
            # kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
            #     knots_cerate(x, variable, session, hand_vel_temp=True, hist_filt_dur='short',
            #                   exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
            #                   condition='controlgain')
        
        # basis = spline_basis(knots, order=order, is_cyclic=is_cyclic)


        plt.title('%s c%d - %f' % (ba,neuron_id, pval[0]))



        # beta = np.hstack((tensor_A[unit, 0, index_dict_A[variable]], [0]))
        # betaB =  np.hstack((tensor_B[unit, 0, index_dict_B[variable]], [0]))
        # tuning = tuning_function(basis, beta)
        # tuningB = tuning_function(basis, betaB)
        # a = knots[0]
        # b = knots[-1]
        # integr = tuning.integrate(a, b)
        # integrB = tuningB.integrate(a, b)


        # tun_centred = lambda x: tuning(x) - integr / (b - a)
        # tun_centredB = lambda x: tuningB(x) - integrB / (b - a)
        
        
        # plt.figure()
        pA, = plt.plot(x, fX_A)
        pB, = plt.plot(x, fX_B)
        plt.fill_between(x,fX_m_ci_A,fX_p_ci_A,color=pA.get_color(),alpha=0.4)
        plt.fill_between(x,fX_m_ci_B,fX_p_ci_B,color=pB.get_color(),alpha=0.4)



    # plt.plot(tensor_A[unit, 0, index_dict_A[variable]])
    # plt.fill_between(range(index_dict_A[variable].shape[0]), tensor_A[unit, 1, index_dict_A[variable]],
    #                  tensor_A[unit, 2, index_dict_A[variable]], alpha=0.4)
    #
    # plt.plot(tensor_B[unit, 0, index_dict_B[variable]])
    # plt.fill_between(range(index_dict_B[variable].shape[0]), tensor_B[unit, 1, index_dict_B[variable]],
    #                  tensor_B[unit, 2, index_dict_B[variable]], alpha=0.4)
        
        sbcnt += 1
        
plt.tight_layout()


