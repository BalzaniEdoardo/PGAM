#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:51:15 2021

@author: edoardo
"""
import numpy as np
import sys, os, dill
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
# sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline/util_preproc'))
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils/'))
from utils_loading import unpack_preproc_data, add_smooth
from GAM_library import *
from time import perf_counter
import statsmodels.api as sm
from basis_set_param_per_session import *
from knots_util import *
from path_class import get_paths_class
import statsmodels.api as sm
import matplotlib.pylab as plt
from copy import deepcopy
from time import perf_counter
from scipy.io import savemat
from knots_constructor import *

from utils_loading import unpack_preproc_data, add_smooth
# dat = np.load('input_hist.npz')
# hist_matrix = dat['hist']
# edge_matrix = dat['edge']
# info = dat['info']
use_var = 'rad_vel'
reload = True
perform_PQL = False
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s44.npz'
session = os.path.basename(fhName).split('.')[0]
# dat = np.load(fhName,allow_pickle=True)
neuron = 95
# neuron = 2
cond_type = 'odd'
cond_value = 1




par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type']
if reload:
    (Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
     trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
     cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type) = unpack_preproc_data(fhName, par_list)



# =============================================================================
#  Use 1/3 half of the data and knots betweem (0,400)
# =============================================================================
par_list = [ 'info_trial','trial_idx']

(trial_type,
 trial_idx) = unpack_preproc_data(fhName, par_list)


if cond_type != 'odd':
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
else:
    idx_subselect = np.where(trial_type['all'])[0]
    if cond_value == 1:
        idx_subselect = idx_subselect[1::3]
    else:
        idx_subselect = idx_subselect[::3]
keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
trial_idx = trial_idx[keep]

sm_handler = smooths_handler()

for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
    # for now skip
    # if var != use_var:
        # continue
    if 'hand' in var:
        continue
    if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
        is_cyclic = True
    else:
        is_cyclic = False

    if var == 'lfp_theta':
        x = lfp_theta[keep, neuron - 1]

    elif var == 'lfp_beta':
        x = lfp_beta[keep, neuron - 1]

    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, neuron - 1]

    elif var == 'spike_hist':
        tmpy = yt[keep, neuron - 1]
        x = tmpy
        # x = np.hstack(([0], tmpy[:-1]))

    else:
        cc = np.where(var_names == var)[0][0]
        x = Xt[keep, cc]
    
   

    knots, x_trans, include_var, is_cyclic, order,\
        kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
            knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                          exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
    
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots], 
                              knots_num=None, perc_out_range=None,
                      is_cyclic=[is_cyclic], lam=50., 
                      penalty_type=penalty_type,
                      der=der,
                      trial_idx=trial_idx, time_bin=time_bin, 
                      is_temporal_kernel=is_temporal_kernel,
                      kernel_length=kernel_len, kernel_direction=kernel_direction,
                      ord_AD=3,ad_knots=8,repeat_extreme_knots=False)



link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                    fisher_scoring=True)

# t0 = perf_counter()

fit_third_stdKnots = gam_model.optim_gam(sm_handler.smooths_var,max_iter=10**3,tol=1e-8,conv_criteria='gcv',
                  perform_PQL=perform_PQL,use_dgcv=True,initial_smooths_guess=False,method='L-BFGS-B',
                  compute_AIC=False,random_init=False,bounds_rho=None,gcv_sel_tol=1e-10,fit_initial_beta=True,
                  filter_trials=None,compute_MI=False)



# =============================================================================
#  Use 1/2 half of the data and knots betweem (0,400)
# =============================================================================
par_list = [ 'info_trial','trial_idx']

(trial_type,
 trial_idx) = unpack_preproc_data(fhName, par_list)


if cond_type != 'odd':
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
else:
    idx_subselect = np.where(trial_type['all'])[0]
    if cond_value == 1:
        idx_subselect = idx_subselect[1::2]
    else:
        idx_subselect = idx_subselect[::2]
keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
trial_idx = trial_idx[keep]

sm_handler = smooths_handler()

for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
    # for now skip
    # if var != use_var:
        # continue
    if 'hand' in var:
        continue
    if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
        is_cyclic = True
    else:
        is_cyclic = False

    if var == 'lfp_theta':
        x = lfp_theta[keep, neuron - 1]

    elif var == 'lfp_beta':
        x = lfp_beta[keep, neuron - 1]

    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, neuron - 1]

    elif var == 'spike_hist':
        tmpy = yt[keep, neuron - 1]
        x = tmpy
        # x = np.hstack(([0], tmpy[:-1]))

    else:
        cc = np.where(var_names == var)[0][0]
        x = Xt[keep, cc]
    
   

    knots, x_trans, include_var, is_cyclic, order,\
        kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
            knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                          exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
    
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots], 
                              knots_num=None, perc_out_range=None,
                      is_cyclic=[is_cyclic], lam=50., 
                      penalty_type=penalty_type,
                      der=der,
                      trial_idx=trial_idx, time_bin=time_bin, 
                      is_temporal_kernel=is_temporal_kernel,
                      kernel_length=kernel_len, kernel_direction=kernel_direction,
                      ord_AD=3,ad_knots=8,repeat_extreme_knots=False)



link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                    fisher_scoring=True)

# t0 = perf_counter()

fit_half_stdKnots = gam_model.optim_gam(sm_handler.smooths_var,max_iter=10**3,tol=1e-8,conv_criteria='gcv',
                  perform_PQL=perform_PQL,use_dgcv=True,initial_smooths_guess=False,method='L-BFGS-B',
                  compute_AIC=False,random_init=False,bounds_rho=None,gcv_sel_tol=1e-10,fit_initial_beta=True,
                  filter_trials=None,compute_MI=False)



# =============================================================================
#  Use 1/2 half of the data and knots betweem (0,200)
# =============================================================================
par_list = [ 'info_trial','trial_idx']

(trial_type,
 trial_idx) = unpack_preproc_data(fhName, par_list)


if cond_type != 'odd':
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
else:
    idx_subselect = np.where(trial_type['all'])[0]
    idx_tmp =  np.where(trial_type['controlgain']==1.)[0]
    if cond_value == 1:
        idx_subselect = idx_subselect[1::2]
    else:
        idx_subselect = idx_subselect[::2]
keep = []

for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

keep_knots = []
for ii in idx_tmp:
    keep_knots = np.hstack((keep_knots, np.where(trial_idx == ii)[0]))
print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
keep_knots = np.array(keep_knots, dtype=int)

trial_idx_knots = trial_idx[keep_knots]
trial_idx = trial_idx[keep]

sm_handler = smooths_handler()

for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
    # for now skip
    # if var != use_var:
        # continue
    if 'hand' in var:
        continue
    if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
        is_cyclic = True
    else:
        is_cyclic = False

    if var == 'lfp_theta':
        x = lfp_theta[keep, neuron - 1]

    elif var == 'lfp_beta':
        x = lfp_beta[keep, neuron - 1]

    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, neuron - 1]

    elif var == 'spike_hist':
        tmpy = yt[keep, neuron - 1]
        x = tmpy
        # x = np.hstack(([0], tmpy[:-1]))

    else:
        cc = np.where(var_names == var)[0][0]
        x = Xt[keep, cc]
        xknots = Xt[keep_knots,cc]
    
    
    if var != 'rad_vel' and var != 'ang_vel':
        knots, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
    else:
        knots =\
                knots_cerate(xknots,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])[0]
        
        # if var == 'rad_vel':
        #     print('HERE',knots,np.nanpercentile(xknots,1),np.nanpercentile(xknots,99))
            
        _, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
        # if var == 'rad_vel':
        #     print(knots)
        #     break
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots], 
                              knots_num=None, perc_out_range=None,
                      is_cyclic=[is_cyclic], lam=50., 
                      penalty_type=penalty_type,
                      der=der,
                      trial_idx=trial_idx, time_bin=time_bin, 
                      is_temporal_kernel=is_temporal_kernel,
                      kernel_length=kernel_len, kernel_direction=kernel_direction,
                      ord_AD=3,ad_knots=8,repeat_extreme_knots=False)



link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                    fisher_scoring=True)

# t0 = perf_counter()

fit_hslf_shortKnots = gam_model.optim_gam(sm_handler.smooths_var,max_iter=10**3,tol=1e-8,conv_criteria='gcv',
                  perform_PQL=perform_PQL,use_dgcv=True,initial_smooths_guess=False,method='L-BFGS-B',
                  compute_AIC=False,random_init=False,bounds_rho=None,gcv_sel_tol=1e-10,fit_initial_beta=True,
                  filter_trials=None,compute_MI=False)





# =============================================================================
#  Use 1/3 half of the data and knots betweem (0,200)
# =============================================================================
par_list = [ 'info_trial','trial_idx']

(trial_type,
 trial_idx) = unpack_preproc_data(fhName, par_list)


if cond_type != 'odd':
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
else:
    idx_subselect = np.where(trial_type['all'])[0]
    idx_tmp =  np.where(trial_type['controlgain']==1.)[0]
    if cond_value == 1:
        idx_subselect = idx_subselect[1::3]
    else:
        idx_subselect = idx_subselect[::3]
keep = []

for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

keep_knots = []
for ii in idx_tmp:
    keep_knots = np.hstack((keep_knots, np.where(trial_idx == ii)[0]))
print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
keep_knots = np.array(keep_knots, dtype=int)

trial_idx_knots = trial_idx[keep_knots]
trial_idx = trial_idx[keep]

sm_handler = smooths_handler()

for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
    # for now skip
    # if var != use_var:
        # continue
    if 'hand' in var:
        continue
    if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
        is_cyclic = True
    else:
        is_cyclic = False

    if var == 'lfp_theta':
        x = lfp_theta[keep, neuron - 1]

    elif var == 'lfp_beta':
        x = lfp_beta[keep, neuron - 1]

    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, neuron - 1]

    elif var == 'spike_hist':
        tmpy = yt[keep, neuron - 1]
        x = tmpy
        # x = np.hstack(([0], tmpy[:-1]))

    else:
        cc = np.where(var_names == var)[0][0]
        x = Xt[keep, cc]
        xknots = Xt[keep_knots,cc]
    
    
    if var != 'rad_vel' and var != 'ang_vel':
        knots, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
    else:
        knots =\
                knots_cerate(xknots,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])[0]
        
        # if var == 'rad_vel':
        #     print('HERE',knots,np.nanpercentile(xknots,1),np.nanpercentile(xknots,99))
            
        _, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
        # if var == 'rad_vel':
        #     print(knots)
        #     break
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots], 
                              knots_num=None, perc_out_range=None,
                      is_cyclic=[is_cyclic], lam=50., 
                      penalty_type=penalty_type,
                      der=der,
                      trial_idx=trial_idx, time_bin=time_bin, 
                      is_temporal_kernel=is_temporal_kernel,
                      kernel_length=kernel_len, kernel_direction=kernel_direction,
                      ord_AD=3,ad_knots=8,repeat_extreme_knots=False)



link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                    fisher_scoring=True)

# t0 = perf_counter()

fit_third_shortKnots = gam_model.optim_gam(sm_handler.smooths_var,max_iter=10**3,tol=1e-8,conv_criteria='gcv',
                  perform_PQL=perform_PQL,use_dgcv=True,initial_smooths_guess=False,method='L-BFGS-B',
                  compute_AIC=False,random_init=False,bounds_rho=None,gcv_sel_tol=1e-10,fit_initial_beta=True,
                  filter_trials=None,compute_MI=False)






# =============================================================================
# Plot compare radial velocity
# =============================================================================
xx_400 = np.linspace(0,390,100)
xx_200 = np.linspace(0,200,100)
fX_third_400, fX_p_ci_third_400, fX_m_ci_third_400 = fit_third_stdKnots.smooth_compute([xx_400], 'rad_vel', perc=0.99)
fX_half_400, fX_p_ci_half_400, fX_m_ci_half_400 = fit_half_stdKnots.smooth_compute([xx_400], 'rad_vel', perc=0.99)

fX_third_200, fX_p_ci_third_200, fX_m_ci_third_200 = fit_third_shortKnots.smooth_compute([xx_200], 'rad_vel', perc=0.99)
fX_half_200, fX_p_ci_half_200, fX_m_ci_half_200 = fit_hslf_shortKnots.smooth_compute([xx_200], 'rad_vel', perc=0.99)


plt.figure()
p, = plt.plot(xx_200,fX_third_200,label='1/3 data (0,200)')
plt.fill_between(xx_200, fX_m_ci_third_200, fX_p_ci_third_200, color=p.get_color(), alpha=0.4)


p,=plt.plot(xx_200,fX_half_200,label='1/2 data (0,200)')
plt.fill_between(xx_200, fX_m_ci_half_200, fX_p_ci_half_200, color=p.get_color(), alpha=0.4)


p,=plt.plot(xx_400,fX_third_400,label='1/3 data (0,400)')
plt.fill_between(xx_400, fX_m_ci_third_400, fX_p_ci_third_400, color=p.get_color(), alpha=0.4)


p,=plt.plot(xx_400,fX_half_400,label='1/2 data (0,400)')
plt.fill_between(xx_400, fX_m_ci_half_400, fX_p_ci_half_400, color=p.get_color(), alpha=0.4)

plt.legend()

plt.tight_layout()
plt.savefig('check_cond_odd_%s_%d.png'%(session,neuron))

# gam_res = fit_third_shortKnots
# FLOAT_EPS = np.finfo(float).eps
# import matplotlib.pylab as plt

# var_list = gam_res.var_list

# pvals = np.clip(gam_res.covariate_significance['p-val'], FLOAT_EPS, np.inf)
# dropvar = np.log(pvals) > np.mean(np.log(pvals)) + 1.5 * np.std(np.log(pvals))
# dropvar = pvals > 0.001
# drop_names = gam_res.covariate_significance['covariate'][dropvar]
# fig = plt.figure(figsize=(14, 8)) 
# plt.suptitle('%s - neuron %d  - PQL %d - %s %f' % (session, neuron,perform_PQL,cond_type,cond_value))
# cc = 0
# cc_plot = 1
# for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
#     # if var != use_var:
#     #     continue
    
#     if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
#         cc += 1
#         continue
#     print('plotting var', var)

#     ax = plt.subplot(5, 4, cc_plot)
#     # ax = plt.subplot(1, 1, cc_plot)

#     if var == 'spike_hist':
#         continue
#     else:
#         cc = np.where(var_names == var)[0][0]
#         x = Xt[keep, cc]
#         # max_x, min_x = X[var].max(), X[var].min()
#         min_x = gam_res.smooth_info[var]['knots'][0][0]
#         max_x = gam_res.smooth_info[var]['knots'][0][-1]
#         min_x = np.max([min_x,np.nanpercentile(x, 1)])
#         max_x = np.min([max_x,np.nanpercentile(x, 99)])



#     if gam_res.smooth_info[var]['is_temporal_kernel']:

#         dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
#         knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
#         ord_ = gam_res.smooth_info[var]['ord']
#         idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

#         impulse = np.zeros(dim_kern)
#         impulse[(dim_kern - 1) // 2] = 1
#         xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
#         fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99, trial_idx=None)
#         if var != 'spike_hist':
            
#             xx = xx[idx_select][1:-1]
#             fX = fX[idx_select][1:-1]
#             fX_p_ci = fX_p_ci[idx_select][1:-1]
#             fX_m_ci = fX_m_ci[idx_select][1:-1]
#         else:
#             if xx.shape[0] > 20:
#                 xx = xx[:(-ord_ - 1)]
#                 fX = fX[:(-ord_ - 1)]
#                 fX_p_ci = fX_p_ci[:(-ord_ - 1)]
#                 fX_m_ci = fX_m_ci[:(-ord_ - 1)]
#             else:
#                 xx = xx[:(-ord_ )]
#                 fX = fX[:(-ord_ )]
#                 fX_p_ci = fX_p_ci[:(-ord_ )]
#                 fX_m_ci = fX_m_ci[:(-ord_ )]


#     else:
#         knots = gam_res.smooth_info[var]['knots']
#         knots_sort = np.unique(knots[0])
#         knots_sort.sort()
        
#         xx = np.linspace(min_x,max_x,100)#(knots_sort[1:] + knots_sort[:-1]) * 0.5

#         fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)
#     # if np.sum(drop_names == var):
#     #     label = var
#     # else:
#     #     label = var
#     label = var
#     if var == 'spike_hist':
#         iend = xx.shape[0] // 2

#         print('set spike_hist')
        
#         iidx = np.where(impulse==1)[0][0]
#         if impulse.shape[0] < 20:
#             iidx = iidx 
#         fX = fX[iidx + 1:][::-1]
#         fX_p_ci = fX_p_ci[iidx+1:][::-1]
#         fX_m_ci = fX_m_ci[iidx+1:][::-1]
#         plt.plot(xx[:fX.shape[0]], fX, ls='-',marker='o', color='k', label=label)
#         plt.fill_between(xx[:fX_m_ci.shape[0]], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
#     else:
#         plt.plot(xx, fX, ls='-', color='k', label=label)
#         plt.fill_between(xx, fX_m_ci, fX_p_ci, color='k', alpha=0.4)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.legend()

#     cc += 1
#     cc_plot += 1
        
# full, _ = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001, method='L-BFGS-B', tol=1e-8,
#                                                 conv_criteria='gcv',
#                                                 max_iter=100, gcv_sel_tol=10 ** -13, random_init=False,
#                                                 use_dgcv=True, initial_smooths_guess=False,
#                                                 fit_initial_beta=True, pseudoR2_per_variable=True,
#                                                 trial_num_vec=trial_idx, k_fold=False, fold_num=5,
#                                                 reducedAdaptive=False,compute_MI=False,
#                                                 perform_PQL=perform_PQL)

 