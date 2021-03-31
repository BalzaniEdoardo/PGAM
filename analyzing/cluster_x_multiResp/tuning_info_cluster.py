#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:23:39 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))

if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
    cluster = False
    JOB = 1
else:
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc/')
    cluster = True
    JOB = int(sys.argv[1])
    
    
from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
# from scipy.io import loadmat
import numpy as np
# import matplotlib.pylab as plt
# import statsmodels.api as sm
import dill
# import pandas as pd
# import scipy.stats as sts
# import scipy.linalg as linalg
# from time import perf_counter
# from seaborn import heatmap
from path_class import get_paths_class
from knots_constructor import knots_cerate
from copy import deepcopy
# from time import sleep
path_gen = get_paths_class()
from numba import njit

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None


def unpack_name(name):
    session = re.findall('m\d+s\d+',name)[0]
    unitID = int(re.findall('_c\d+_',name)[0].split('_')[1].split('c')[1])
    man_type = re.findall('_c\d+_[a-z]+_',name)[0].split('_')[2]
    man_val = float(re.findall('\d+.\d\d\d\d',name)[0])
    return session,unitID,man_type,man_val

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno','m72':'Marco'}


    
if not cluster:
    path_to_gam = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'
    npz_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

else:
    path_to_gam = '/scratch/jpn5/mutual_info/'
    npz_path = '/scratch/jpn5/dataset_firefly/'
    
# build session list
sess_list = []
for root, dirs, files in os.walk(path_to_gam):
    print(root)
    if not re.match('^.+/gam_m\d+s\d+$', root):
        continue
    session = os.path.basename(root).split('_')[1]
    if session != 'm53s130':
        sess_list += [session]

print(len(sess_list))
  




# use an example session for creating the knots
fhName = os.path.join(npz_path,'m44s174.npz')
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

    if 'hand' in var:
        continue
    dict_range[var] = max(dict_range[var][0],knots[0]), \
                      min(dict_range[var][1],knots[-1])
                      
    if 'lfp' in var:
        dict_range[var] = (-np.pi,np.pi-0.0001)
        
    if var == 'rad_acc':
        dict_range[var] = (-700,700)
    
    if var == 'ang_vel':
        dict_range[var] = (-50,50)
        
    if var == 'ang_path':
        dict_range[var] = (-50,50)
        
    if var == 'rad_target':
        dict_range[var] = (0,350)
        
    
    if var.startswith('t_'):
        dict_eval_num[var] = len(knots)
    else:
        dict_eval_num[var] = 15
    
    index_list += [var] * dict_eval_num[var]

# npz_path = None
first=True

eval_matrix = np.zeros((0,len(index_list)))
index_list = np.array(index_list)

dtype_dict = {
    'names': [
        'monkey','session', 'brain area','unit id', 'electrode id', 'cluster id','channel id', 'manipulation type','manipulation value',
        'pseudo_r2'
        ],
    'formats':
        [
        'U20','U20', 'U20', int, int, int,int,'U20',float,float
            ]
        }
info_matrix = np.zeros(0,dtype=dtype_dict)


# check done already

orig_done = deepcopy(info_matrix)

session = sess_list[JOB]
up_until = False
for name in os.listdir(os.path.join(path_to_gam,'gam_%s'%session)):

    root = os.path.join(path_to_gam,'gam_%s'%session)
    
    # if not 'controlgain' in name:
    #     continue
    if not re.match(pattern, name):
        continue
    session_2,unitID,man_type,man_val = unpack_name(name)
    assert(session==session_2)
    
    check_done = ((orig_done['session'] == session) * 
                  (orig_done['unit id'] == unitID) *
                  (orig_done['manipulation type'] == man_type) *
                  (orig_done['manipulation value'] == man_val)
                  ).sum()
    assert(check_done <= 1)
    if check_done == 1:
        # print('skip %s %d %s %f'%(session,unitID,man_type,man_val))
        continue
    # tst = (session == 'm53s115') & (unitID ==76) &(man_type=='reward') & (man_val==1)
    
    # if (not up_until):
    #     if (not tst):
    #         print(not up_until)
    #         continue
    # if tst:
    #     print('FOUND')
    #     up_until = True
    # print(not up_until)
    # sleep(0.1)
    # find = (info_all['session'] == session) * (info_all['unit id'] == unitID) *\
    #     (info_all['manipulation value'] == man_val) * (info_all['manipulation type'] == man_type)*\
    #         (info_all['variable'] == 'rad_vel')
    # assert(find.sum()==1 or find.sum()==0)
    
   
    if session != session_prev:
        np.savez('%s_eval_matrix_and_info.npz'%session,eval_matrix=eval_matrix,info=info_matrix,index_list=index_list)

        print('session',session)
        session_prev = session
        

       
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
            pr2 = gam_res['p_r2_coupling_full'] 
            full = gam_res['full']
            del gam_res
    except:
        print('BAD dill',session,unitID,man_type,man_val)
        continue
        
    eval_tmp = np.zeros((1,index_list.shape[0]))*np.nan
    cnt_vars = 0
    for var in full.var_list:
        if var == 'spike_hist' or var.startswith('neu') or (var == 't_ptb') or ('hand' in var):
            continue
        
        
        beta = np.hstack((full.beta[full.index_dict[var]],[0]))
        
        
        x_knots = full.smooth_info[var]['knots'][0]
        order = full.smooth_info[var]['ord']
        is_cyclic = full.smooth_info[var]['is_cyclic']
        basis = spline_basis(x_knots,order=order,is_cyclic=is_cyclic)
        tuning = tuning_function(basis, beta)
        
        a = x_knots[0]
        b = x_knots[-1]
        integr = tuning.integrate(a, b)

        tun_centred = lambda x:tuning(x) - integr/(b-a)
        xx=np.linspace(dict_range[var][0],dict_range[var][1],dict_eval_num[var])
        xx[-1] = xx[-1] - 0.0001
        eval_tmp[0,index_list==var] = tun_centred(xx)
        
        
        cnt_vars += 1
    
    info_neu = np.zeros(1,dtype=dtype_dict)
    info_neu['session'] = session
    info_neu['monkey'] = monkey_dict[session.split('s')[0]]
    info_neu['unit id'] = unitID
    info_neu['electrode id'] = unit_info['electrode_id'][unitID-1]
    info_neu['cluster id'] = unit_info['cluster_id'][unitID-1]
    info_neu['channel id'] = unit_info['channel_id'][unitID-1]
    info_neu['brain area'] = unit_info['brain_area'][unitID-1]
    
    info_neu['manipulation type'] = man_type
    info_neu['manipulation value'] = man_val
    info_neu['pseudo_r2'] = pr2
    info_matrix = np.hstack((info_matrix,info_neu))
    eval_matrix = np.vstack((eval_matrix,eval_tmp))

        

        

        

         
            
#         info_all = np.hstack((info_all,info_neu))

np.savez('%s_eval_matrix_and_info.npz'%session,eval_matrix=eval_matrix,info=info_matrix,index_list=index_list)


            
# save for JP

        
            
            
            

        