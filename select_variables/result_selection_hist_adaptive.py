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

plt.close('all')
fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_history_filter_adaptive/'
file_list = os.listdir(fh_folder)

counts_hand_vel = {'derivative':0,'adaptive':0}
counts_all = 0
pr2_derivative = []
pr2_adaptive = []


dtype_dict = {'names':('session','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2'),
              'formats':('U30','U4','U30',int,int,int,'U40',float)}


dtype_dict_info = {'names':('session','variable','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2 derivative','pr2 adaptive','is significant derivative','is significant adaptive',
                       'p-val derivative','p-val adaptive'),
              'formats':('U30','U30','U4','U30',int,int,int,'U40',float,float,bool,bool,float,float)}



df_derivative = np.zeros(0,dtype=dtype_dict)
df_adaptive = np.zeros(0,dtype=dtype_dict)
df_info = np.zeros(0,dtype=dtype_dict_info)
df_all = {}
first = True
for fhname in file_list:
    with open(os.path.join(fh_folder, fhname), 'rb') as fh:
        results = dill.load(fh)
    if results['reduced_short'] is None:
        pass
    elif ('hand_vel1' in results['reduced_short'].var_list) or\
        ('hand_vel2' in results['reduced_short'].var_list):
        counts_hand_vel['adaptive'] = counts_hand_vel['adaptive'] + 1
        
        tmp_adaptive = np.zeros(1,dtype=dtype_dict)
        tmp_adaptive['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_adaptive['brain area'] = results['brain_area']
        tmp_adaptive['unit type'] = results['unit_typ']
        tmp_adaptive['cluster id'] = results['cluster_id']
        tmp_adaptive['electrode id'] = results['electrode_id']
        tmp_adaptive['channel id'] = results['channel_id']
        tmp_adaptive['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_adaptive['pr2'] = results['p_r2_adaptive']
        df_adaptive = np.hstack((df_adaptive,tmp_adaptive))

    
    if results['reduced_derivative'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_derivative'].var_list) or\
        ('hand_vel2' in results['reduced_derivative'].var_list):
        counts_hand_vel['derivative'] = counts_hand_vel['derivative'] + 1
        
        tmp_derivative = np.zeros(1,dtype=dtype_dict)
        tmp_derivative['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_derivative['brain area'] = results['brain_area']
        tmp_derivative['unit type'] = results['unit_typ']
        tmp_derivative['cluster id'] = results['cluster_id']
        tmp_derivative['electrode id'] = results['electrode_id']
        tmp_derivative['channel id'] = results['channel_id']
        tmp_derivative['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_derivative['pr2'] = results['p_r2_derivative']
        df_derivative = np.hstack((df_derivative,tmp_derivative))

    # if not results['reduced_derivative'] is None:  
        
        # print(results['reduced_derivative'].var_list)
    
    full_derivative = results['full_derivative']
    full_adaptive =  results['full_adaptive']
    
    tmp_info = np.zeros(len(full_derivative.var_list),dtype=dtype_dict_info)
    cc = 0
    for var in full_derivative.var_list:
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
        tmp_info['pr2 derivative'][cc] = results['p_r2_derivative']
        tmp_info['pr2 adaptive'][cc] = results['p_r2_adaptive']
        
        if results['reduced_derivative'] is None:
            tmp_info['is significant derivative'][cc] = False
            idx = results['full_derivative'].covariate_significance['covariate'] == var
            tmp_info['p-val derivative'][cc] = results['full_derivative'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_derivative'].var_list:
            tmp_info['is significant derivative'][cc] = False
            idx = results['full_derivative'].covariate_significance['covariate'] == var
            tmp_info['p-val derivative'][cc] = results['full_derivative'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_derivative'].covariate_significance['covariate'] == var
            if results['reduced_derivative'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant derivative'][cc] = bl
            tmp_info['p-val derivative'][cc] = results['reduced_derivative'].covariate_significance['p-val'][idx]

        if results['reduced_short'] is None:
            tmp_info['is significant adaptive'][cc] = False
            idx = results['full_derivative'].covariate_significance['covariate'] == var
            tmp_info['p-val adaptive'][cc] = results['full_adaptive'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_short'].var_list:
            tmp_info['is significant adaptive'][cc] = False
            idx = results['full_derivative'].covariate_significance['covariate'] == var
            tmp_info['p-val adaptive'][cc] = results['full_adaptive'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_short'].covariate_significance['covariate'] == var
            if results['reduced_short'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant adaptive'][cc] = bl
            tmp_info['p-val adaptive'][cc] = results['reduced_short'].covariate_significance['p-val'][idx]

            
        
        cc += 1
       
    df_info = np.hstack((df_info,tmp_info))
    pr2_derivative += [results['p_r2_derivative']]
    pr2_adaptive += [results['p_r2_adaptive']]
    
    
    kern_dim = full_derivative.smooth_info['spike_hist']['basis_kernel'].shape[0]
    if first:
        fX_mat_adaptive = np.zeros((0,kern_dim))
        fX_mat_derivative = np.zeros((0,kern_dim))
        first = False

    impulse = np.zeros(kern_dim)
    impulse[(kern_dim - 1) // 2] = 1
    fX_derivative, _, _ = full_derivative.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    fX_adaptive, _, _ = full_adaptive.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    
    fX_mat_derivative = np.vstack((fX_mat_derivative, fX_derivative.reshape((1,fX_derivative.shape[0]))))
    fX_mat_adaptive = np.vstack((fX_mat_adaptive, fX_adaptive.reshape((1,fX_adaptive.shape[0]))))


    counts_all += 1
    
pr2_adaptive_arr = np.array(pr2_adaptive)
pr2_derivative_arr = np.array(pr2_derivative)

keep = pr2_adaptive_arr > np.nanpercentile(pr2_adaptive,1)
pr2_adaptive_arr = pr2_adaptive_arr[keep]
pr2_derivative_arr = pr2_derivative_arr[keep]

plt.figure(figsize=(8,6))
plt.subplot(111,aspect='equal')
plt.scatter(pr2_derivative_arr,pr2_adaptive_arr,s=15)

xlim = plt.xlim()
plt.plot(xlim,xlim,'k')
plt.xlim(xlim)
plt.ylim(xlim)
plt.title('Cross validatted pseudo-r^2',fontsize=15)
plt.ylabel('adaptive filter pseudo-r^2',fontsize=12)
plt.xlabel('long filter pseudo-r^2',fontsize=12)
plt.tight_layout()

plt.figure(figsize=(14,10))


for k in range(fX_mat_derivative.shape[0]):
    # row = k//5 + 1
    # col = k%5 + 1
    plt.subplot(5,10,k+1)
    time = (np.arange(fX_mat_derivative.shape[1])-fX_mat_derivative.shape[1]//2)*0.006
    sele = np.abs(fX_mat_derivative[k,:] - fX_mat_derivative[k,0])>10**-6

    plt.plot(time[sele],fX_mat_derivative[k,sele])
    plt.title('pr2: %.3f'%pr2_derivative[k])
# plt.tight_layout()

# plt.figure(figsize=(14,10))


# for k in range(fX_mat_derivative.shape[0]):
#     # row = k//5 + 1
#     # col = k%5 + 1
#     plt.subplot(5,10,k+1)
    sele = np.abs(fX_mat_adaptive[k,:] - fX_mat_adaptive[k,0])>10**-6
    fs = deepcopy(fX_mat_adaptive[k,:].flatten())
    time = (np.arange(fs.shape[0])-fs.shape[0]//2)*0.006
    fs[~sele] = np.nan
    # plt.plot(time,fX_mat_derivative[k,:])
    
    # plt.plot(time,fX_mat_adaptive[k,:]-fX_mat_adaptive[k,0] + fX_mat_derivative[k,0])
    plt.plot(time[sele],fX_mat_adaptive[k,sele]-fX_mat_adaptive[k,0] + fX_mat_derivative[k,0])

    
    # plt.plot(time[sele],fs[sele]-fX_mat_adaptive[k,0] + fX_mat_derivative[k,0])
plt.tight_layout()
    
