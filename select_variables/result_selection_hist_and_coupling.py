#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:06:40 2020

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



plt.close('all')
fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/coupling_vs_spike_history/'
file_list = os.listdir(fh_folder)

counts_hand_vel = {'input':0,'spike_hist':0}
counts_all = 0
pr2_input = []
pr2_spike_hist = []
pr2_coupling = []


dtype_dict = {'names':('session','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2'),
              'formats':('U30','U4','U30',int,int,int,'U40',float)}


dtype_dict_info = {'names':('session','variable','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2 input','pr2 spike_hist','pr2 w/o spike_hist',
                       'pr2 w/o coupling',
                       'pr2 coupling','is significant input','is significant spike_hist',
                       'is significant coupling',
                       'p-val input','p-val spike_hist','p-val coupling'),
              'formats':('U30','U30','U4','U30',int,int,int,'U40',float,float,float,float,float,bool,bool,bool,float,float,float)}



df_input = np.zeros(0,dtype=dtype_dict)
df_spike_hist = np.zeros(0,dtype=dtype_dict)
df_coupling = np.zeros(0,dtype=dtype_dict)

df_info = np.zeros(0,dtype=dtype_dict_info)
df_all = {}
first = True

current_session = ''
# load input variables
for fhname in file_list:
    session = fhname.split('coupling_')[1].split('_')[0]
    
   
        
    with open(os.path.join(fh_folder, fhname), 'rb') as fh:
        results = dill.load(fh)
    if results['reduced_spike_hist'] is None:
        pass
    elif ('hand_vel1' in results['reduced_spike_hist'].var_list) or\
        ('hand_vel2' in results['reduced_spike_hist'].var_list):
        counts_hand_vel['spike_hist'] = counts_hand_vel['spike_hist'] + 1
        
        tmp_spike_hist = np.zeros(1,dtype=dtype_dict)
        tmp_spike_hist['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_spike_hist['brain area'] = results['brain_area']
        tmp_spike_hist['unit type'] = results['unit_typ']
        tmp_spike_hist['cluster id'] = results['cluster_id']
        tmp_spike_hist['electrode id'] = results['electrode_id']
        tmp_spike_hist['channel id'] = results['channel_id']
        tmp_spike_hist['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_spike_hist['pr2'] = results['p_r2_spike_hist']
        df_spike_hist = np.hstack((df_spike_hist,tmp_spike_hist))

    
    if results['reduced_input'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_input'].var_list) or\
        ('hand_vel2' in results['reduced_input'].var_list):
        counts_hand_vel['input'] = counts_hand_vel['input'] + 1
        
        tmp_input = np.zeros(1,dtype=dtype_dict)
        tmp_input['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_input['brain area'] = results['brain_area']
        tmp_input['unit type'] = results['unit_typ']
        tmp_input['cluster id'] = results['cluster_id']
        tmp_input['electrode id'] = results['electrode_id']
        tmp_input['channel id'] = results['channel_id']
        tmp_input['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_input['pr2'] = results['p_r2_input']
        df_input = np.hstack((df_input,tmp_input))
        
    if results['reduced_coupling'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_coupling'].var_list) or\
        ('hand_vel2' in results['reduced_coupling'].var_list):
        counts_hand_vel['coupling'] = counts_hand_vel['coupling'] + 1
        
        tmp_coupling = np.zeros(1,dtype=dtype_dict)
        tmp_coupling['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_coupling['brain area'] = results['brain_area']
        tmp_coupling['unit type'] = results['unit_typ']
        tmp_coupling['cluster id'] = results['cluster_id']
        tmp_coupling['electrode id'] = results['electrode_id']
        tmp_coupling['channel id'] = results['channel_id']
        tmp_coupling['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_coupling['pr2'] = results['p_r2_coupling']
        df_coupling = np.hstack((df_coupling,tmp_coupling))

    # if not results['reduced_input'] is None:  
        
        # print(results['reduced_input'].var_list)
    
    full_input = results['full_input']
    full_spike_hist =  results['full_spike_hist']
    full_coupling =  results['full_coupling']
    
    tmp_info = np.zeros(len(full_input.var_list),dtype=dtype_dict_info)
    cc = 0
    for var in full_input.var_list:
        tmp_info['session'][cc] = fhname.split('filt_')[1].split('_')[0]
        tmp_info['variable'][cc] = var
        tmp_info['brain area'][cc] = results['brain_area']
        tmp_info['unit type'][cc] = results['unit_typ']
        tmp_info['cluster id'][cc] = results['cluster_id']
        tmp_info['electrode id'][cc] = results['electrode_id']
        tmp_info['channel id'][cc] = results['channel_id']
        tmp_info['ID'][cc] = (fhname.split('filt_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_info['pr2 input'][cc] = results['p_r2_input']
        tmp_info['pr2 spike_hist'][cc] = results['p_r2_spike_hist']
        tmp_info['pr2 coupling'][cc] = results['p_r2_coupling']
        
        if results['reduced_input'] is None:
            tmp_info['is significant input'][cc] = False
            idx = results['full_input'].covariate_significance['covariate'] == var
            tmp_info['p-val input'][cc] = results['full_input'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_input'].var_list:
            tmp_info['is significant input'][cc] = False
            idx = results['full_input'].covariate_significance['covariate'] == var
            tmp_info['p-val input'][cc] = results['full_input'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_input'].covariate_significance['covariate'] == var
            if results['reduced_input'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant input'][cc] = bl
            tmp_info['p-val input'][cc] = results['reduced_input'].covariate_significance['p-val'][idx]

        if results['reduced_spike_hist'] is None:
            tmp_info['is significant spike_hist'][cc] = False
            idx = results['full_spike_hist'].covariate_significance['covariate'] == var
            tmp_info['p-val spike_hist'][cc] = results['full_spike_hist'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_spike_hist'].var_list:
            tmp_info['is significant spike_hist'][cc] = False
            idx = results['full_spike_hist'].covariate_significance['covariate'] == var
            tmp_info['p-val spike_hist'][cc] = results['full_spike_hist'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_spike_hist'].covariate_significance['covariate'] == var
            if results['reduced_spike_hist'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant spike_hist'][cc] = bl
            tmp_info['p-val spike_hist'][cc] = results['reduced_spike_hist'].covariate_significance['p-val'][idx]
            
            
        if results['reduced_coupling'] is None:
            tmp_info['is significant reduced_coupling'][cc] = False
            idx = results['full_coupling'].covariate_significance['covariate'] == var
            tmp_info['p-val reduced_coupling'][cc] = results['full_reduced_coupling'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_coupling'].var_list:
            tmp_info['is significant coupling'][cc] = False
            idx = results['full_coupling'].covariate_significance['covariate'] == var
            tmp_info['p-val coupling'][cc] = results['full_coupling'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_coupling'].covariate_significance['covariate'] == var
            if results['reduced_coupling'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant coupling'][cc] = bl
            tmp_info['p-val coupling'][cc] = results['reduced_coupling'].covariate_significance['p-val'][idx]
            
        
        cc += 1
       
    pr2_input += [results['p_r2_input']]
    pr2_spike_hist += [results['p_r2_spike_hist']]
    pr2_coupling += [results['p_r2_coupling']]

    
    
    kern_dim = full_coupling.smooth_info['spike_hist']['basis_kernel'].shape[0]
    if first:
        fX_mat_spike_hist = np.zeros((0,kern_dim))
        fX_mat_input = np.zeros((0,kern_dim))
        fX_mat_coupling = np.zeros((0,kern_dim))
        first = False

    impulse = np.zeros(kern_dim)
    impulse[(kern_dim - 1) // 2] = 1
    # fX_input, _, _ = full_input.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    fX_spike_hist, _, _ = full_spike_hist.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    fX_coupling, _, _ = full_coupling.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)


    # fX_mat_input = np.vstack((fX_mat_input, fX_input.reshape((1,fX_input.shape[0]))))
    fX_mat_spike_hist = np.vstack((fX_mat_spike_hist, fX_spike_hist.reshape((1,fX_spike_hist.shape[0]))))
    fX_mat_coupling = np.vstack((fX_mat_coupling, fX_coupling.reshape((1,fX_coupling.shape[0]))))

    
    # create the test model matrix
    neuron = int(fhname.split('_c')[-1].split('.')[0])
    print('NEURONE',neuron)
    if session != current_session:
        print('extracting sessiion',session)
        current_session = session
        # load inputs
        par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']
        fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
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
        cond_type = 'all'
        cond_value =1
    
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
        
        
        # fit with the spatial hand velocity
        hand_vel_temp = False
        sm_handler = smooths_handler()
        sm_handler_test = smooths_handler()
        for var in np.hstack((var_names, ['lfp_beta','lfp_alpha','lfp_theta','spike_hist'])):
            # for now skip
            
            if var=='hand_vel1' or var == 'hand_vel2':
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
                    knots_cerate(x,var,session,hand_vel_temp=hand_vel_temp,hist_filt_dur='long',
                    exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
                    
            _, x_test, _, _, _,\
                _,_,_,_,_ =\
                    knots_cerate(x_test,var,session,hand_vel_temp=hand_vel_temp,
                    exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
            
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
                              kernel_length=kernel_len, kernel_direction=kernel_direction,ord_AD=3,ad_knots=4)
        
        
                sm_handler_test.add_smooth(var, [x_test], ord=order, knots=[knots], 
                                      knots_num=None, perc_out_range=None,
                              is_cyclic=[is_cyclic], lam=50, 
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx_test, time_bin=time_bin, 
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len, kernel_direction=kernel_direction,ord_AD=3,ad_knots=4)
                
        
        for other in neuron_keep:
            
            # if other == neuron:
            #     continue
            
            tmpy = yt[keep, other - 1]
            x = tmpy
            x_test = yt[keep_test, other - 1]
            
            knots, x_trans, include_var, is_cyclic, order,\
                kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                    knots_cerate(x,'spike_hist',session,hand_vel_temp=hand_vel_temp,hist_filt_dur='long',
                    exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
                    
            _, x_test, _, _, _,\
                _,_,_,_,_ =\
                    knots_cerate(x_test,'spike_hist',session,hand_vel_temp=hand_vel_temp,
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
            
    
        modelX_all, idx_dict_all = sm_handler.get_exog_mat(sm_handler.smooths_var)
        modelX_test_all, idx_dict_test_all = sm_handler_test.get_exog_mat(sm_handler_test.smooths_var)
        
        link = deriv3_link(sm.genmod.families.links.log())
        poissFam = sm.genmod.families.family.Poisson(link=link)
        family = d2variance_family(poissFam)
        # modelX_neu = np.ones((modelX_all.shape[0], modelX_all.shape[1] - len(idx_dict_all['spike_hist'])))
        # modelX_test_neu = np.ones((modelX_test_all.shape[0], modelX_test_all.shape[1] - len(idx_dict_test_all['spike_hist'])))

    
    # substite hisory and pop coupling
    curren_hist = sm_handler.smooths_dict.pop('spike_hist')
    curren_hist_test = sm_handler_test.smooths_dict.pop('spike_hist')
    
    curren_coupl = deepcopy(sm_handler.smooths_dict['neu_%d'%neuron])
    curren_coupl_test = deepcopy(sm_handler_test.smooths_dict['neu_%d'%neuron])
    
    for var in ['lfp_beta','lfp_alpha','lfp_theta']:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)
            sm_handler_test.smooths_dict.pop(var)
            sm_handler_test.smooths_var.remove(var)
        if 'beta' in var:
            x = lfp_beta[keep,neuron-1]
            x_test = lfp_beta[keep_test,neuron-1]
        if 'alpha' in var:
            x = lfp_alpha[keep,neuron-1]
            x_test = lfp_alpha[keep_test,neuron-1]
        if 'theta' in var:
            x = lfp_theta[keep,neuron-1]
            x_test = lfp_theta[keep_test,neuron-1]
        
        knots, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=hand_vel_temp,hist_filt_dur='long',
                exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
                
        _, x_test, _, _, _,\
            _,_,_,_,_ =\
                knots_cerate(x_test,var,session,hand_vel_temp=hand_vel_temp,
                exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'])
                
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
        
        
        LFP = sm_handler[var].X[:,:-1].todense() - np.nanmean(sm_handler[var].X[:,:-1].todense(),axis=0)
        LFP_test = np.array(sm_handler_test[var].X[:,:-1].todense()) - np.nanmean(sm_handler_test[var].X[:,:-1].todense(),axis=0)
        modelX_all[:, idx_dict_all[var]] = LFP
        modelX_test_all[:, idx_dict_test_all[var]] = LFP_test


    sm_handler.smooths_dict['spike_hist'] = curren_coupl
    sm_handler_test.smooths_dict['spike_hist'] = curren_coupl_test
    
    modelX_all[:, idx_dict_all['spike_hist']] = modelX_all[:, idx_dict_all['neu_%d'%neuron]]
    modelX_test_all[:, idx_dict_all['spike_hist']] = modelX_test_all[:, idx_dict_all['neu_%d'%neuron]]

    
    # this_beta = full_coupling.beta
    beta_no_hist = np.zeros(modelX_all.shape[1])
    beta_no_coupl = np.zeros(modelX_all.shape[1])
    beta_all = np.zeros(modelX_all.shape[1])
    
    beta_no_hist[0] = full_coupling.beta[0]
    beta_no_coupl[0] = full_coupling.beta[0]
    beta_all[0] = full_coupling.beta[0]
    

    for var in sm_handler.smooths_var:
        if var == 'neu_%d'%neuron:
            continue
        beta_all[idx_dict_all[var]] = full_coupling.beta[full_coupling.index_dict[var]]
        if var!='spike_hist':
            beta_no_hist[idx_dict_all[var]] = full_coupling.beta[full_coupling.index_dict[var]]
        
        if var.startswith('neu_'):
            continue
        beta_no_coupl[idx_dict_all[var]] = full_coupling.beta[full_coupling.index_dict[var]]
    
    
    tmp_info['pr2 w/o coupling'] = pseudo_r2_compute(yt[keep_test,neuron-1], family, modelX_test_all, beta_no_coupl)
    tmp_info['pr2 w/o spike_hist'] = pseudo_r2_compute(yt[keep_test,neuron-1], family, modelX_test_all, beta_no_hist)

    df_info = np.hstack((df_info,tmp_info))
    
    

    counts_all += 1

pr2_nocoupl = []
pr2_nohist = []
pr2_full=[]
pr2_input = []
for unit in np.unique(df_info['ID']):
    idx = np.where(df_info['ID'] == unit)[0][0]
    pr2_nocoupl += [df_info[idx]['pr2 w/o coupling']]
    pr2_nohist += [df_info[idx]['pr2 w/o spike_hist']]
    pr2_full += [df_info[idx]['pr2 coupling']]
    pr2_input += [df_info[idx]['pr2 input']]
    
    
from sklearn.linear_model import HuberRegressor     

plt.figure()
plt.subplot(221)
plt.scatter(pr2_nocoupl,pr2_nohist)
X = np.ones((len(pr2_nocoupl),1))
X[:,0] = pr2_nocoupl
linreg = HuberRegressor()
linreg.fit(X,pr2_nohist)

x = np.linspace(min(pr2_nocoupl),max(pr2_nocoupl),10)
plt.plot(x, linreg.predict(x.reshape(x.shape[0],1)),'g')
plt.plot([min(pr2_nocoupl),max(pr2_nocoupl)],[min(pr2_nocoupl),max(pr2_nocoupl)],'r')
plt.xlabel('no coupl')
plt.ylabel('no hist')

plt.subplot(222)

X = np.ones((len(pr2_nocoupl),1))
X[:,0] = pr2_nocoupl
linreg = HuberRegressor()
linreg.fit(X,pr2_full)

x = np.linspace(min(pr2_nocoupl),max(pr2_nocoupl),10)
plt.plot(x, linreg.predict(x.reshape(x.shape[0],1)),'g')

plt.scatter(pr2_nocoupl,pr2_full)
plt.plot([min(pr2_nocoupl),max(pr2_full)],[min(pr2_nocoupl),max(pr2_full)],'r')
plt.xlabel('no coupl')
plt.ylabel('full')

plt.subplot(223)


X = np.ones((len(pr2_nocoupl),1))
X[:,0] = pr2_nohist
linreg = HuberRegressor()
linreg.fit(X,pr2_full)

x = np.linspace(min(pr2_nohist),max(pr2_nohist),10)
plt.plot(x, linreg.predict(x.reshape(x.shape[0],1)),'g')

plt.scatter(pr2_nohist,pr2_full)
plt.plot([min(pr2_nocoupl),max(pr2_full)],[min(pr2_nocoupl),max(pr2_full)],'r')
plt.xlabel('no hist')
plt.ylabel('full')

plt.subplot(224)


X = np.ones((len(pr2_nocoupl),1))
X[:,0] = pr2_input
linreg = HuberRegressor()
linreg.fit(X,pr2_full)

x = np.linspace(min(pr2_input),max(pr2_input),10)
plt.plot(x, linreg.predict(x.reshape(x.shape[0],1)),'g')



plt.scatter(pr2_input,pr2_full)
plt.plot([min(pr2_input),max(pr2_full)],[min(pr2_input),max(pr2_full)],'r')
plt.xlabel('only input')
plt.ylabel('full')
plt.suptitle('cv R^2')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('hist_vs_coupling_compare.png')

# check basis sets
var = 'lfp_beta'
plt.close('all')
full_tmp = deepcopy(full_coupling)
numbasis = len(full_tmp.beta[full_tmp.index_dict[var]])
plt.figure()
x = np.linspace(np.pi,np.pi,100)
for k in range(numbasis):
    beta = np.zeros(numbasis)
    beta[k] = 1
    # full_tmp.beta[full_tmp.index_dict[var]] = 
    full_tmp.beta[full_tmp.index_dict[var]] = beta
    
    fX,_,_ = full_tmp.smooth_compute([x],var)
    plt.plot(x,fX)
# pr2_spike_hist_arr = np.array(pr2_spike_hist)
# pr2_input_arr = np.array(pr2_input)
# pr2_coupling_arr = np.array(pr2_coupling)



# keep = pr2_spike_hist_arr > np.nanpercentile(pr2_spike_hist,1)
# pr2_spike_hist_arr = pr2_spike_hist_arr[keep]
# pr2_input_arr = pr2_input_arr[keep]
# pr2_coupling_arr = pr2_coupling_arr[keep]

# plt.figure(figsize=(8,6))
# plt.subplot(111,aspect='equal')
# plt.scatter(pr2_input_arr,pr2_spike_hist_arr,s=15,label='hist only')
# plt.scatter(pr2_input_arr,pr2_coupling_arr,s=15,label='coupling')


# xlim = plt.xlim()
# plt.plot(xlim,xlim,'k')
# plt.xlim(xlim)
# plt.ylim(xlim)
# plt.title('Cross validatted pseudo-r^2',fontsize=15)
# plt.ylabel('input + coupling/history',fontsize=12)
# plt.xlabel('input only',fontsize=12)
# plt.tight_layout()

# plt.figure(figsize=(14,10))
# for k in range(fX_mat_spike_hist.shape[0]):
#     if k + 1 == 101:
#         break
#     # row = k//5 + 1
#     # col = k%5 + 1
#     plt.subplot(10,10,k+1)
#     time = (np.arange(fX_mat_spike_hist.shape[1])-fX_mat_spike_hist.shape[1]//2)*0.006
#     sele = np.abs(fX_mat_spike_hist[k,:] - fX_mat_spike_hist[k,0])>10**-6

#     plt.plot(time[sele],fX_mat_spike_hist[k,sele],label='hist only')
#     # plt.title('pr2: %.3f'%pr2_spike_hist[k])
# # plt.tight_layout()

# # plt.figure(figsize=(14,10))


# # for k in range(fX_mat_input.shape[0]):
# #     # row = k//5 + 1
# #     # col = k%5 + 1
# #     plt.subplot(5,10,k+1)
#     sele = np.abs(fX_mat_coupling[k,:] - fX_mat_coupling[k,0])>10**-6
#     fs = deepcopy(fX_mat_coupling[k,:].flatten())
#     time = (np.arange(fs.shape[0])-fs.shape[0]//2)*0.006
#     fs[~sele] = np.nan
#     # plt.plot(time,fX_mat_input[k,:])
    
#     # plt.plot(time,fX_mat_spike_hist[k,:]-fX_mat_spike_hist[k,0] + fX_mat_input[k,0])
#     plt.plot(time[sele],fX_mat_coupling[k,sele]-fX_mat_coupling[k,0] + fX_mat_coupling[k,0],label='coupling')

    
#     # plt.plot(time[sele],fs[sele]-fX_mat_spike_hist[k,0] + fX_mat_input[k,0])
# plt.tight_layout()
