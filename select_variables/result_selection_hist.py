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

fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_history_filter/'
file_list = os.listdir(fh_folder)

counts_hand_vel = {'long':0,'short':0}
counts_all = 0
pr2_long = []
pr2_short = []


dtype_dict = {'names':('session','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2'),
              'formats':('U30','U4','U30',int,int,int,'U40',float)}


dtype_dict_info = {'names':('session','variable','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2 long','pr2 short','is significant long','is significant short',
                       'p-val long','p-val short'),
              'formats':('U30','U30','U4','U30',int,int,int,'U40',float,float,bool,bool,float,float)}



df_long = np.zeros(0,dtype=dtype_dict)
df_short = np.zeros(0,dtype=dtype_dict)
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
        counts_hand_vel['short'] = counts_hand_vel['short'] + 1
        
        tmp_short = np.zeros(1,dtype=dtype_dict)
        tmp_short['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_short['brain area'] = results['brain_area']
        tmp_short['unit type'] = results['unit_typ']
        tmp_short['cluster id'] = results['cluster_id']
        tmp_short['electrode id'] = results['electrode_id']
        tmp_short['channel id'] = results['channel_id']
        tmp_short['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_short['pr2'] = results['p_r2_short']
        df_short = np.hstack((df_short,tmp_short))

    
    if results['reduced_long'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_long'].var_list) or\
        ('hand_vel2' in results['reduced_long'].var_list):
        counts_hand_vel['long'] = counts_hand_vel['long'] + 1
        
        tmp_long = np.zeros(1,dtype=dtype_dict)
        tmp_long['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_long['brain area'] = results['brain_area']
        tmp_long['unit type'] = results['unit_typ']
        tmp_long['cluster id'] = results['cluster_id']
        tmp_long['electrode id'] = results['electrode_id']
        tmp_long['channel id'] = results['channel_id']
        tmp_long['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_long['pr2'] = results['p_r2_long']
        df_long = np.hstack((df_long,tmp_long))

    # if not results['reduced_long'] is None:  
        
        # print(results['reduced_long'].var_list)
    
    full_long = results['full_long']
    full_short =  results['full_short']
    
    tmp_info = np.zeros(len(full_long.var_list),dtype=dtype_dict_info)
    cc = 0
    for var in full_long.var_list:
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
        tmp_info['pr2 long'][cc] = results['p_r2_long']
        tmp_info['pr2 short'][cc] = results['p_r2_short']
        
        if results['reduced_long'] is None:
            tmp_info['is significant long'][cc] = False
            idx = results['full_long'].covariate_significance['covariate'] == var
            tmp_info['p-val long'][cc] = results['full_long'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_long'].var_list:
            tmp_info['is significant long'][cc] = False
            idx = results['full_long'].covariate_significance['covariate'] == var
            tmp_info['p-val long'][cc] = results['full_long'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_long'].covariate_significance['covariate'] == var
            if results['reduced_long'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant long'][cc] = bl
            tmp_info['p-val long'][cc] = results['reduced_long'].covariate_significance['p-val'][idx]

        if results['reduced_short'] is None:
            tmp_info['is significant short'][cc] = False
            idx = results['full_long'].covariate_significance['covariate'] == var
            tmp_info['p-val short'][cc] = results['full_short'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_short'].var_list:
            tmp_info['is significant short'][cc] = False
            idx = results['full_long'].covariate_significance['covariate'] == var
            tmp_info['p-val short'][cc] = results['full_short'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_short'].covariate_significance['covariate'] == var
            if results['reduced_short'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant short'][cc] = bl
            tmp_info['p-val short'][cc] = results['reduced_short'].covariate_significance['p-val'][idx]

            
        
        cc += 1
       
    df_info = np.hstack((df_info,tmp_info))
    pr2_long += [results['p_r2_long']]
    pr2_short += [results['p_r2_short']]
    
    
    kern_dim = full_long.smooth_info['spike_hist']['basis_kernel'].shape[0]
    if first:
        fX_mat_short = np.zeros((0,kern_dim))
        fX_mat_long = np.zeros((0,kern_dim))
        first = False

    impulse = np.zeros(kern_dim)
    impulse[(kern_dim - 1) // 2] = 1
    fX_long, _, _ = full_long.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    fX_short, _, _ = full_short.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    
    fX_mat_long = np.vstack((fX_mat_long, fX_long.reshape((1,fX_long.shape[0]))))
    fX_mat_short = np.vstack((fX_mat_short, fX_short.reshape((1,fX_short.shape[0]))))


    counts_all += 1
    
pr2_short_arr = np.array(pr2_short)
pr2_long_arr = np.array(pr2_long)

keep = pr2_short_arr > np.nanpercentile(pr2_short,1)
pr2_short_arr = pr2_short_arr[keep]
pr2_long_arr = pr2_long_arr[keep]

plt.figure(figsize=(8,6))
plt.subplot(111,aspect='equal')
plt.scatter(pr2_long_arr,pr2_short_arr,s=15)

xlim = plt.xlim()
plt.plot(xlim,xlim,'k')
plt.xlim(xlim)
plt.ylim(xlim)
plt.title('Cross validatted pseudo-r^2',fontsize=15)
plt.ylabel('short filter pseudo-r^2',fontsize=12)
plt.xlabel('long filter pseudo-r^2',fontsize=12)
plt.tight_layout()

plt.figure(figsize=(14,10))


for k in range(fX_mat_long.shape[0]):
    # row = k//5 + 1
    # col = k%5 + 1
    plt.subplot(5,10,k+1)
    time = time = (np.arange(fs.shape[0])-fs.shape[0]//2)*0.006
    sele = time>=0

    plt.plot(time[sele],fX_mat_long[k,sele])
    plt.title('pr2: %.3f'%pr2_long[k])
plt.tight_layout()

plt.figure(figsize=(14,10))


for k in range(fX_mat_long.shape[0]):
    # row = k//5 + 1
    # col = k%5 + 1
    plt.subplot(5,10,k+1)
    sele = np.abs(fX_mat_short[k,:] - fX_mat_short[k,0])>10**-6
    fs = deepcopy(fX_mat_short[k,:].flatten())
    time = time = (np.arange(fs.shape[0])-fs.shape[0]//2)*0.006
    fs[~sele] = np.nan
    # plt.plot(time,fX_mat_long[k,:])
    
    # plt.plot(time,fX_mat_short[k,:]-fX_mat_short[k,0] + fX_mat_long[k,0])
    plt.plot(time[sele],fX_mat_short[k,sele]-fX_mat_short[k,0] + fX_mat_long[k,0])

    
    # plt.plot(time[sele],fs[sele]-fX_mat_short[k,0] + fX_mat_long[k,0])
plt.tight_layout()
    
