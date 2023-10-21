#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:09:13 2020

@author: edoardo
"""
import os,inspect, sys, re
print(inspect.getfile(inspect.currentframe()))
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

def pseudo_r2_compute(spk, family, modelX, params):

    lin_pred = np.dot(modelX, params)
    mu = family.fitted(lin_pred)
    res_dev_t = family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t ** 2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t ** 2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2



user_paths = get_paths_class()
folder_name = deepcopy(thisPath)

tot_fits = 1
plot_res = False
fit_fully_coupled = False
use_k_fold = True
reducedcoupling = False
num_folds = 5

print('folder name')
print(folder_name)
print(' ')
main_dir = os.path.dirname(folder_name)

use_fisher_scoring = False

# load the data Kaushik passed me
try:
    folder_name = '/scratch/jpn5/dataset_firefly' #user_paths.get_path('data_hpc')
    sv_folder_base = ''#user_paths.get_path('code_hpc')
    fhName = os.path.join(folder_name, sys.argv[2])
    dat = np.load(os.path.join(folder_name, fhName), allow_pickle=True)
except:
    print('EXCEPTION RAISED')
    folder_name = ''
    sv_folder_base = ''
    fhName = os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/','m53s39.npz')
    # fhName = '/Users/edoardo/Downloads/PPC+PFC+MST/m53s109.npz'
    if fhName.endswith('.mat'):
        dat = loadmat(fhName)
    elif fhName.endswith('.npy'):
        dat = np.load(fhName, allow_pickle=True).all()
    elif fhName.endswith('.npz'):
        dat = np.load(fhName, allow_pickle=True)





par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']
(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
  trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
  cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
  channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)


dict_xlims = {}

# get the unit to include as input covariates
cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate_filter > 0.9
isi_v_filter = isi_v_filter < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

# unit number according to matlab indexing
neuron_keep = np.arange(1, yt.shape[1] + 1)[combine_filter]

# extract the condition to be filtered and the triL
session = os.path.basename(fhName).split('.')[0]

try:  # IF CLUSTER JOB IS RUNNING
    JOB = int(sys.argv[1]) - 1
    list_condition = np.load(os.path.join('','condition_list_%s.npy' % session))
    neuron_list = list_condition[JOB:JOB + tot_fits]['neuron']
    cond_type_list = list_condition[JOB:JOB + tot_fits]['condition']
    cond_value_list = list_condition[JOB:JOB + tot_fits]['value']
    pop_size_max = yt.shape[1]
except Exception as ex:

    JOB = 0
    list_condition = np.load(os.path.join(os.path.join(main_dir,'preprocessing_pipeline', 'util_preproc'),

        'condition_list_%s.npy' % session))
    # tot_fits = list_condition.shape[0]
    neuron_list = list_condition[JOB:JOB + tot_fits]['neuron']
    cond_type_list = list_condition[JOB:JOB + tot_fits]['condition']
    cond_value_list = list_condition[JOB:JOB + tot_fits]['value']
    pop_size_max = yt.shape[1]

if 'ptb' in list_condition['condition']:
     cond_knots = 'ptb'
elif 'controlgain' in list_condition['condition']:
     cond_knots = 'controlgain'
elif 'density' in list_condition['condition']:
     cond_knots = 'density'
else:
    cond_knots = None
    # neuron = neuron_list[0]
    
numfit = 0
for neuron in neuron_list:
    cond_type = cond_type_list[numfit]
    cond_value = cond_value_list[numfit]
    numfit += 1
    
    if cond_type == 'odd':
        all_trs = np.arange(trial_type.shape[0])
        all_trs = all_trs[trial_type['all'] == 1]
        if cond_value == 1:
            idx_subselect = all_trs[1::2]
        else:
            idx_subselect = all_trs[::2]
    else:
        idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
    
    test_trials = idx_subselect[::10]
    train_trials = np.sort(list(set(idx_subselect).difference(set(idx_subselect[::10]))))
    
    
    # take the train trials
    keep = []
    for ii in train_trials:
        keep = np.hstack((keep, np.where(trial_idx == ii)[0]))
        
    keep_test = []
    for ii in test_trials:
        keep_test = np.hstack((keep_test, np.where(trial_idx == ii)[0]))
    
    
    print(' condition', cond_type, cond_value)
    
    keep = np.array(keep, dtype=int)
    trial_idx_train = trial_idx[keep]
    
    keep_test = np.array(keep_test, dtype=int)
    trial_idx_test = trial_idx[keep_test]
     
    # fit with coupling 
     
    hand_vel_temp = True
    sm_handler = smooths_handler()
    sm_handler_test = smooths_handler()
    for var in np.hstack((var_names, ['lfp_beta','lfp_alpha','lfp_theta','spike_hist'])):
        # for now skip
        # if var !='spike_hist':
        #     continue
        if var == 'hand_vel1' or var == 'hand_vel2':
            continue
        
        if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
            is_cyclic = True
        else:
            is_cyclic = False
    
        if var == 'lfp_theta':
            x = lfp_theta[keep, neuron - 1]
            x_test = lfp_theta[keep_test, neuron - 1]
    
        elif var == 'lfp_beta':
            x = lfp_beta[keep, neuron - 1]
            x_test = lfp_beta[keep_test, neuron - 1]
    
    
        elif var == 'lfp_alpha':
            x = lfp_alpha[keep, neuron - 1]
            x_test = lfp_alpha[keep_test, neuron - 1]
    
    
        elif var == 'spike_hist':
            tmpy = yt[keep, neuron - 1]
            x = tmpy
            
            x_test = yt[keep_test, neuron - 1]
    
            # x = np.hstack(([0], tmpy[:-1]))
    
        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep, cc]
            
            x_test = Xt[keep_test, cc]
    
       
        
        knots, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=hand_vel_temp,hist_filt_dur='short',
                exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'],
                condition=cond_knots)
                
        _, x_test, _, _, _,\
            _,_,_,_,_ =\
                knots_cerate(x_test,var,session,hand_vel_temp=hand_vel_temp,
                exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
        
        if not var.startswith('t_') and var != 'spike_hist':
            if 'lfp' in var:
                dict_xlims[var] = (-np.pi,np.pi)
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
                          kernel_direction=kernel_direction,ord_AD=3,ad_knots=4,
                          repeat_extreme_knots=False)
    
    
            sm_handler_test.add_smooth(var, [x_test], ord=order, knots=[knots], 
                                  knots_num=None, perc_out_range=None,
                          is_cyclic=[is_cyclic], lam=50, 
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx_test, time_bin=time_bin, 
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len, 
                          kernel_direction=kernel_direction,ord_AD=3,ad_knots=4,
                          repeat_extreme_knots=False)
            
    
    for other in neuron_keep:
        # break
        if other == neuron:
            continue
        print('adding unit: %d'%other)
        if brain_area[neuron - 1] == brain_area[other - 1]:
            filt_len = 'short'
        else:
            filt_len = 'long'
        
        tmpy = yt[keep, other - 1]
        x = tmpy
        x_test = yt[keep_test, other - 1]
        
        knots, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,'spike_hist',session,hand_vel_temp=hand_vel_temp,hist_filt_dur=filt_len,
                exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
                
        _, x_test, _, _, _,\
            _,_,_,_,_ =\
                knots_cerate(x_test,'spike_hist',session,hand_vel_temp=hand_vel_temp,hist_filt_dur=filt_len,
                exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
        var = 'neu_%d'%other
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
                          kernel_length=kernel_len, kernel_direction=kernel_direction,ord_AD=3,ad_knots=4)
    
    
            sm_handler_test.add_smooth(var, [x_test], ord=order, knots=[knots], 
                                  knots_num=None, perc_out_range=None,
                          is_cyclic=[is_cyclic], lam=50, 
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx_test, time_bin=time_bin, 
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len, kernel_direction=kernel_direction,ord_AD=3,ad_knots=4)
            
            
            
                
                
        
    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)
    
    
    gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                            fisher_scoring=use_fisher_scoring)
    
    
    
    full_coupling, reduced_coupling = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001, method='L-BFGS-B', tol=1e-8,
                                                        conv_criteria='gcv',
                                                        max_iter=1000, gcv_sel_tol=10 ** -13, random_init=False,
                                                        use_dgcv=True, initial_smooths_guess=False,
                                                        fit_initial_beta=True, pseudoR2_per_variable=True,
                                                        trial_num_vec=trial_idx_train, k_fold=use_k_fold, fold_num=num_folds,
                                                        reducedAdaptive=False,compute_MI=False,perform_PQL=True)
    
    modelX_test, idx_dict_test = sm_handler_test.get_exog_mat(full_coupling.var_list)
    spk = yt[keep_test,neuron-1]
    p_r2_coupling = pseudo_r2_compute(spk,family, modelX_test,full_coupling.beta)
    beta = np.zeros(full_coupling.beta.shape[0])
    if not reduced_coupling is None:
        beta[0] = reduced_coupling.beta[0]
    
        for var in reduced_coupling.var_list:
            beta[full_coupling.index_dict[var]] = reduced_coupling.beta[reduced_coupling.index_dict[var]]
            
        p_r2_coupling_reduced = pseudo_r2_compute(spk,family, modelX_test,beta)
    else:
        p_r2_coupling_reduced = np.nan
    
    if not os.path.exists('gam_%s'%session):
        os.makedirs('gam_%s'%session)
        
    
        
    with open('gam_%s/fit_results_%s_c%d_%s_%.4f.dill'%(session,session,neuron,cond_type,cond_value),'wb') as fh:
       
            data_dict = {
              'full':full_coupling,
              'reduced':reduced_coupling,
              'p_r2_coupling_full':p_r2_coupling,
              'p_r2_coupling_reduced':p_r2_coupling_reduced,
              'brain_area':brain_area[neuron-1],
              'unit_type':unit_type[neuron-1],
              'cluster_id':cluster_id[neuron-1], 
              'electrode_id':electrode_id[neuron-1],
              'channel_id':channel_id[neuron-1],
              'all_areas':brain_area,
              'xlim':dict_xlims
              }
            fh.write(dill.dumps(data_dict))
    
    
