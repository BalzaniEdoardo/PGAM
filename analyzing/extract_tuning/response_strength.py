#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:23:39 2020

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

path_gen = get_paths_class()


def compute_integral_mean(gam_res,var,discr=1000):
     
    knots = gam_res.smooth_info[var]['knots'][0]
    order = gam_res.smooth_info[var]['ord']
    
    if 'lfp' in var:
        is_cyclic = True
    else:
        is_cyclic = False


    # construct the basis function
    if gam_res.smooth_info[var]['is_temporal_kernel'] and var == 'spike_hist':
        exp_bspline = spline_basis(knots, order, is_cyclic=is_cyclic)
    elif gam_res.smooth_info[var]['is_temporal_kernel']:
        exp_bspline = spline_basis(knots/2., order, is_cyclic=is_cyclic)
    else:
        exp_bspline = spline_basis(knots, order, is_cyclic=is_cyclic)
    
    select = gam_res.index_dict[var]
    beta = np.hstack((gam_res.beta[select],[0]))
    tuning = tuning_function(exp_bspline, beta, subtract_integral_mean=True)
    x = np.linspace(knots[0], knots[-1]-0.0001, discr)
    y = tuning(x) ** 2
    integr = simps(y, dx=x[1] - x[0]) / (x[-1] - x[0])
    return integr


def unpack_name(name):
    session = re.findall('m\d+s\d+',name)[0]
    unitID = int(re.findall('_c\d+_',name)[0].split('_')[1].split('c')[1])
    man_type = re.findall('_c\d+_[a-z]+_',name)[0].split('_')[2]
    man_val = float(re.findall('\d+.\d\d\d\d',name)[0])
    return session,unitID,man_type,man_val

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno','m72':'Marco'}


    


# use an example session for creating the knots
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m44s174.npz'
session = os.path.basename(fhName).split('.')[0]
# dat = np.load(fhName,allow_pickle=True)
neuron = 11
# neuron = 2
cond_type = 'all'
cond_value = 1

par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type']
(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
      trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
      cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type) = unpack_preproc_data(fhName, par_list)
        
pattern = '^fit_results_m\d+s\d+_c\d+_[a-z]+_\d+.\d\d\d\d.dill$'
session_prev = ''
# info_all = np.load('knot_num_and_range.npy',allow_pickle=True)


eval_continuous = 15

dict_range  = {}
dict_eval_num = {}
     
        
for var in np.hstack((var_names,['lfp_beta', 'lfp_alpha', 'lfp_theta'])):
    dict_range[var] = -np.inf,np.inf


idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]

keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
trial_idx = trial_idx[keep]
    
index_list = []
for var in np.hstack((var_names,['lfp_beta', 'lfp_alpha', 'lfp_theta'])):
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
        
    dict_range[var] = max(dict_range[var][0],knots[0]), \
                      min(dict_range[var][1],knots[-1])
                      
    if var.startswith('t_'):
        dict_eval_num[var] = len(knots)
    else:
        dict_eval_num[var] = 20
    
    index_list += [var] * dict_eval_num[var]

npz_path = None
first=True
path_to_gam = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'

eval_matrix = np.zeros((0,len(index_list)))
index_list = np.array(index_list)


variables = np.hstack((var_names,['lfp_beta', 'lfp_alpha', 'lfp_theta','spike_hist']))
keep_var = np.ones(variables.shape,dtype=bool)
cc = 0
for var in variables:
    if 'hand' in var:
        keep_var[cc] = False
    cc+=1
variables = variables[keep_var]


resp_magnitude_full = []
resp_magnitude_reduced = []
for var in variables:
    resp_magnitude_full += ['%s_resp_magn_full'%var]

for var in variables:
    resp_magnitude_reduced += ['%s_resp_magn_reduced'%var]
dtype_dict = {
                'names':('monkey','session','unit','cluster_id','electrode_id','channel_id','brain_area','manipulation type',
                         'manipulation value')+tuple(variables)
                +tuple(resp_magnitude_full)+tuple(resp_magnitude_reduced),
                'formats':('U30','U30') + (int,)*4 + ('U3','U20',float) + (bool,)*variables.shape[0] + (float,)*variables.shape[0]
                + (float,)*variables.shape[0]
                    }
info_matrix = np.zeros(0,dtype=dtype_dict)


# # check done already
if os.path.exists('response_strength_info.npy'):
    info_matrix = np.load('response_strength_info.npy',allow_pickle=True)
orig_done = deepcopy(info_matrix)



done_id = []
for k in range(orig_done.shape[0]):
    done_id += ['%s_%d_%s_%s'%(orig_done['session'][k],
                                orig_done['unit'][k],
                                orig_done['manipulation type'][k],
                                orig_done['manipulation value'][k]
                                )]
done_id = np.sort(np.array(done_id))


session_list_todo = ['gam_m53s50','gam_m53s49','gam_m53s42',
                     'gam_m53s48','gam_m53s51','gam_m53s43','gam_m53s44','gam_m53s39',
                     'gam_m53s40','gam_m53s41','gam_m53s36']
up_until = False
for (root,dirs,files) in os.walk(path_to_gam):
    
    # if not root.split('/')[-1] in session_list_todo:
    #     continue
    # if not 'm53s41' in root:
    #     continue
    for name in files:
        if not re.match(pattern, name):
            continue
        session,unitID,man_type,man_val = unpack_name(name)
        
        
        item = '%s_%d_%s_%s'%(session,unitID,man_type,man_val)
        
        idxFirst = bisect_left(done_id,item)
        # assert(check_done <= 1)
        # if check_done >= 1:
        #     print('skip')
            # continue
        try:
            if done_id[idxFirst] == item:
                # rmv_index = np.ones(done_id.shape,dtype=bool)
                # rmv_index[idxFirst:idxLast] = False
                # done_id = done_id[rmv_index]
                # print('skip')
                continue
        except:
            
            
            check_done = ((orig_done['session'] == session) * 
                          (orig_done['unit'] == unitID) *
                          (orig_done['manipulation type'] == man_type) *
                          (orig_done['manipulation value'] == man_val)
                          ).sum()
        # assert(check_done <= 1)
            if check_done == 1:
                # print('skip %s %d %s %f'%(session,unitID,man_type,man_val))
                continue
        
        
        npz_path = path_gen.search_npz(session)
        print(item)
        if npz_path is None:
            continue
        if session != session_prev:
            print('session',session)
            session_prev = session
            np.save('response_strength_info.npy',info_matrix)

           
            dat = np.load(os.path.join(npz_path, session+'.npz'), allow_pickle=True)
            unit_info = dat['unit_info'].all()
        
            # check filters for unit quality not needed because gam fit are only 
            # for qualty uints, but just in case
            cont_rate_filter = (unit_info['cR'] < 0.2) | (unit_info['unit_type']=='multiunit')
            presence_rate_filter = unit_info['presence_rate'] > 0.9
            isi_v_filter = unit_info['isiV'] < 0.2
            pp_size = isi_v_filter.shape[0]
            combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
        
        if not combine_filter[unitID - 1]:
            print('BAD unit: ',session, unitID)
            continue
        
        try:
        # open fits
            with open(os.path.join(root, name), 'rb') as fh:
                gam_res = dill.load(fh)
                full = gam_res['full']
                reduced =  gam_res['reduced']
                del gam_res
        except:
            print('BAD dill',session,unitID,man_type,man_val)
            continue
        
        info_neu = np.zeros(1,dtype=dtype_dict)

        for var in variables:
            try:
                info_neu['%s_resp_magn_full'%var] = compute_integral_mean(full,var)
                if var in reduced.var_list:
                    info_neu['%s_resp_magn_reduced' % var] = compute_integral_mean(reduced, var)
                else:
                    info_neu['%s_resp_magn_reduced' % var] = 0
            except:
                info_neu['%s_resp_magn_full'%var] = np.nan
                info_neu['%s_resp_magn_reduced' % var] = np.nan
                
        info_neu['session'] = session
        info_neu['monkey'] = monkey_dict[session.split('s')[0]]
        info_neu['unit'] = unitID
        info_neu['electrode_id'] = unit_info['electrode_id'][unitID-1]
        info_neu['cluster_id'] = unit_info['cluster_id'][unitID-1]
        info_neu['channel_id'] = unit_info['channel_id'][unitID-1]
        info_neu['brain_area'] = unit_info['brain_area'][unitID-1]
        
        info_neu['manipulation type'] = man_type
        info_neu['manipulation value'] = man_val
        if not reduced is None:
            for var in reduced.var_list:
                    if var.startswith('neu') or var.startswith('t_ptb') :
                        continue
                    idx = np.where(reduced.covariate_significance['covariate'] == var)[0]
                    if reduced.covariate_significance['p-val'][idx]<0.001:
                        info_neu[var] = True
                        
        info_matrix = np.hstack((info_matrix,info_neu))

        

np.save('response_strength_info.npy',info_matrix)


            
# save for JP

        
            
            
            
        
        