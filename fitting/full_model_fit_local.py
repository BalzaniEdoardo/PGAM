#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:09:13 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
# thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
from scipy.io import savemat
if os.path.exists('/scratch/ges6/GAM_Repo/GAM_library/'):
    sys.path.append('/scratch/ges6/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/ges6/GAM_Repo/preprocessing_pipeline/util_preproc')
    sys.path.append('/scratch/ges6/GAM_Repo/firefly_utils/')
else:
    thisPath = '/Users/edoardo/Work/Code/GAM_code/'
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
from scipy.integrate import simps

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


# structured array that will contain the reuslts
dtype_dict = {'names':('monkey','session','condition_type','condition_value','neuron','cluster_id','channel_id',
                       'electrode_id','unit_type','brain_area','pseudo_r2','variable','pval','mutual_info', 'x',
                       'model_rate_Hz','raw_rate_Hz','kernel_strength','signed_kernel_strength'),
              'formats':('U30','U30','U30',float,int,int,int,int,'U30','U30',float,'U30',float,float,object,object,object,float,float)}

monkey_dict = {'m53':'Schro','m73':'Jimmy'}

tot_fits = 1
plot_res = False
fit_fully_coupled = False
use_k_fold = True
reducedcoupling = False
num_folds = 5

# print('folder name')
# print(folder_name)
# print(' ')
# main_dir = os.path.dirname(folder_name)

use_fisher_scoring = False

# load the data Kaushik passed me

minSess = 27
folder_name = ''
sv_folder_base = ''
file_loc = '/Users/edoardo/Downloads/troubleshooting files/'
for root, dirs, files in os.walk(file_loc, topdown=False):
    
    for name in files:
        if not '.npz' in name:
            continue
        sess_num = int(name.split('s')[1].split('.')[0])
        if sess_num < minSess:
            continue
        
        fhName = os.path.join(file_loc,name)
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
        cont_rate_filter = (cont_rate_filter < 0.5) & (unit_type == 'singleunit')
        presence_rate_filter = presence_rate_filter > 0.5
        isi_v_filter = isi_v_filter < 0.2
        combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
        
        # unit number according to matlab indexing
        neuron_keep = np.arange(1, yt.shape[1] + 1)[combine_filter]
        
        # extract the condition to be filtered and the trial
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
            list_condition = np.load(os.path.join(file_loc,'condition_list_%s.npy' % session))
            # list_condition = np.load(os.path.join(os.path.join(main_dir,'preprocessing_pipeline', 'util_preproc'),
            #     'condition_list_%s.npy' % session))
            list_condition = list_condition[list_condition['condition']=='all'] # choose conditions
            tot_fits = list_condition.shape[0]
            neuron_list = list_condition[JOB:JOB + tot_fits]['neuron']
            cond_type_list = list_condition[JOB:JOB + tot_fits]['condition']
            cond_value_list = list_condition[JOB:JOB + tot_fits]['value']
            pop_size_max = yt.shape[1]
        
        if  'ptb' in list_condition['condition']:
             cond_knots = 'ptb'
        elif  'controlgain' in list_condition['condition']:
             cond_knots = 'controlgain'
        elif  'density' in list_condition['condition']:
             cond_knots = 'density'
        else:
            cond_knots = None
            # neuron = neuron_list[0]
        regr_res = np.zeros(0, dtype=dtype_dict)

        numfit=0
        for neuron in neuron_list:  # fit units loop
            cond_type = cond_type_list[numfit]
            cond_value = cond_value_list[numfit]
            numfit+=1
            
            print('.............fitting session:%s, unit:c%d, condition:%s, value:%.4f'%(session,neuron,cond_type,cond_value))
            
            if cond_type == 'odd':
                all_trs = np.arange(trial_type.shape[0])
                all_trs = all_trs[trial_type['all']==1]
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
            # trial_idx_train = trial_idx_train[:500]
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
                if all(np.isnan(x_trans)):
                    print('var %s is all NANs!'%var)
                    continue
                if is_temporal_kernel and x_trans.sum() == 0:
                    print('No events present in %s'%var)
                    continue
                # print(np.nanmax(np.abs(x_trans)),np.nanmax(np.abs(x_test)))
                if include_var:
                    if var in sm_handler.smooths_dict.keys():
                        sm_handler.smooths_dict.pop(var)
                        sm_handler.smooths_var.remove(var)
                    print(var,np.isnan(x_trans).mean())
            
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
                                                                reducedAdaptive=False,compute_MI=True,perform_PQL=True)
            
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

            tmp_res = np.zeros(len((full_coupling.var_list)),dtype=dtype_dict)
            cs_table = full_coupling.covariate_significance
            for cc in range(len(full_coupling.var_list)):

                var = full_coupling.var_list[cc]
                cs_var = cs_table[cs_table['covariate'] == var]
                tmp_res['brain_area'][cc] = brain_area[neuron-1]
                tmp_res['monkey'][cc] = monkey_dict[session.split('s')[0]]
                tmp_res['session'][cc] = session
                tmp_res['neuron'][cc] = neuron
                tmp_res['variable'][cc] = var
                tmp_res['condition_type'][cc] = cond_type
                tmp_res['condition_value'][cc] = cond_value
                tmp_res['pseudo_r2'][cc] = p_r2_coupling
                tmp_res['pval'][cc] = cs_var['p-val']
                tmp_res['channel_id'][cc] = channel_id[neuron-1]
                tmp_res['electrode_id'][cc] = channel_id[neuron - 1]
                tmp_res['cluster_id'][cc] = channel_id[neuron - 1]
                tmp_res['unit_type'][cc] = unit_type[neuron-1]
                if var in full_coupling.mutual_info.keys():
                    tmp_res['mutual_info'][cc] = full_coupling.mutual_info[var]
                else:
                    tmp_res['mutual_info'][cc] = np.nan
                if var in full_coupling.tuning_Hz.__dict__.keys():
                    tmp_res['x'][cc] = full_coupling.tuning_Hz.__dict__[var].x
                    tmp_res['model_rate_Hz'][cc] = full_coupling.tuning_Hz.__dict__[var].y_model
                    tmp_res['raw_rate_Hz'][cc] = full_coupling.tuning_Hz.__dict__[var].y_raw

                # compute kernel strength
                if full_coupling.smooth_info[var]['is_temporal_kernel']:
                    dim_kern = full_coupling.smooth_info[var]['basis_kernel'].shape[0]
                    knots_num = full_coupling.smooth_info[var]['knots'][0].shape[0]
                    x = np.zeros(dim_kern)
                    x[(dim_kern - 1) // 2] = 1
                    fX = full_coupling.smooth_compute([x], var, 0.95)[0]
                    if (var == 'spike_hist') or ('neu_') in var:
                        fX = fX[(dim_kern - 1) // 2:] - fX[0]
                    else:
                        fX = fX - fX[-1]
                    tmp_res['kernel_strength'][cc] = simps(fX**2, dx=0.006)/(0.006*fX.shape[0])
                    tmp_res['signed_kernel_strength'][cc] = simps(fX,dx=0.006)/(0.006*fX.shape[0])

                else:
                    knots = full_coupling.smooth_info[var]['knots']
                    xmin = knots[0].min()
                    xmax = knots[0].max()
                    func = lambda x: (full_coupling.smooth_compute([x],var,0.95)[0] - full_coupling.smooth_compute([x],var,0.95)[0].mean() )**2
                    xx = np.linspace(xmin,xmax,500)
                    dx = xx[1] - xx[0]
                    tmp_res['kernel_strength'][cc] = simps(func(xx),dx=dx)/(xmax-xmin)

            regr_res = np.hstack((regr_res,tmp_res))

            savemat('%s_fit_info_results.mat'%session,mdict={'pgam_results':regr_res})