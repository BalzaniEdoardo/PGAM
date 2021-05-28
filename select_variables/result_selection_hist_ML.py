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
fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_history_filter_ML/'
file_list = os.listdir(fh_folder)

counts_hand_vel = {'PL':0,'ML':0}
counts_all = 0
pr2_PL = []
pr2_ML = []


dtype_dict = {'names':('session','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2'),
              'formats':('U30','U4','U30',int,int,int,'U40',float)}


dtype_dict_info = {'names':('session','variable','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2 PL','pr2 ML','is significant PL','is significant ML',
                       'p-val PL','p-val ML'),
              'formats':('U30','U30','U4','U30',int,int,int,'U40',float,float,bool,bool,float,float)}



df_PL = np.zeros(0,dtype=dtype_dict)
df_ML = np.zeros(0,dtype=dtype_dict)
df_info = np.zeros(0,dtype=dtype_dict_info)
df_all = {}
first = True
for fhname in file_list:
    with open(os.path.join(fh_folder, fhname), 'rb') as fh:
        results = dill.load(fh)
    if results['reduced_ML'] is None:
        pass
    elif ('hand_vel1' in results['reduced_ML'].var_list) or\
        ('hand_vel2' in results['reduced_ML'].var_list):
        counts_hand_vel['ML'] = counts_hand_vel['ML'] + 1
        
        tmp_ML = np.zeros(1,dtype=dtype_dict)
        tmp_ML['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_ML['brain area'] = results['brain_area']
        tmp_ML['unit type'] = results['unit_typ']
        tmp_ML['cluster id'] = results['cluster_id']
        tmp_ML['electrode id'] = results['electrode_id']
        tmp_ML['channel id'] = results['channel_id']
        tmp_ML['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_ML['pr2'] = results['p_r2_ML']
        df_ML = np.hstack((df_ML,tmp_ML))

    
    if results['reduced_PL'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_PL'].var_list) or\
        ('hand_vel2' in results['reduced_PL'].var_list):
        counts_hand_vel['PL'] = counts_hand_vel['PL'] + 1
        
        tmp_PL = np.zeros(1,dtype=dtype_dict)
        tmp_PL['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_PL['brain area'] = results['brain_area']
        tmp_PL['unit type'] = results['unit_typ']
        tmp_PL['cluster id'] = results['cluster_id']
        tmp_PL['electrode id'] = results['electrode_id']
        tmp_PL['channel id'] = results['channel_id']
        tmp_PL['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_PL['pr2'] = results['p_r2_PL']
        df_PL = np.hstack((df_PL,tmp_PL))

    # if not results['reduced_PL'] is None:  
        
        # print(results['reduced_PL'].var_list)
    
    full_PL = results['full_PL']
    full_ML =  results['full_ML']
    
    tmp_info = np.zeros(len(full_PL.var_list),dtype=dtype_dict_info)
    cc = 0
    for var in full_PL.var_list:
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
        tmp_info['pr2 PL'][cc] = results['p_r2_PL']
        tmp_info['pr2 ML'][cc] = results['p_r2_ML']
        
        if results['reduced_PL'] is None:
            tmp_info['is significant PL'][cc] = False
            idx = results['full_PL'].covariate_significance['covariate'] == var
            tmp_info['p-val PL'][cc] = results['full_PL'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_PL'].var_list:
            tmp_info['is significant PL'][cc] = False
            idx = results['full_PL'].covariate_significance['covariate'] == var
            tmp_info['p-val PL'][cc] = results['full_PL'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_PL'].covariate_significance['covariate'] == var
            if results['reduced_PL'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant PL'][cc] = bl
            tmp_info['p-val PL'][cc] = results['reduced_PL'].covariate_significance['p-val'][idx]

        if results['reduced_ML'] is None:
            tmp_info['is significant ML'][cc] = False
            idx = results['full_PL'].covariate_significance['covariate'] == var
            tmp_info['p-val ML'][cc] = results['full_ML'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_ML'].var_list:
            tmp_info['is significant ML'][cc] = False
            idx = results['full_PL'].covariate_significance['covariate'] == var
            tmp_info['p-val ML'][cc] = results['full_ML'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_ML'].covariate_significance['covariate'] == var
            if results['reduced_ML'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant ML'][cc] = bl
            tmp_info['p-val ML'][cc] = results['reduced_ML'].covariate_significance['p-val'][idx]

            
        
        cc += 1
       
    df_info = np.hstack((df_info,tmp_info))
    pr2_PL += [results['p_r2_PL']]
    pr2_ML += [results['p_r2_ML']]
    
    
    kern_dim = full_PL.smooth_info['spike_hist']['basis_kernel'].shape[0]
    if first:
        fX_mat_ML = np.zeros((0,kern_dim))
        fX_mat_PL = np.zeros((0,kern_dim))
        first = False

    impulse = np.zeros(kern_dim)
    impulse[(kern_dim - 1) // 2] = 1
    fX_PL, _, _ = full_PL.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    fX_ML, _, _ = full_ML.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    
    fX_mat_PL = np.vstack((fX_mat_PL, fX_PL.reshape((1,fX_PL.shape[0]))))
    fX_mat_ML = np.vstack((fX_mat_ML, fX_ML.reshape((1,fX_ML.shape[0]))))


    counts_all += 1
    
pr2_ML_arr = np.array(pr2_ML)
pr2_PL_arr = np.array(pr2_PL)

keep = pr2_ML_arr > np.nanpercentile(pr2_ML,1)
pr2_ML_arr = pr2_ML_arr[keep]
pr2_PL_arr = pr2_PL_arr[keep]

plt.figure(figsize=(8,6))
plt.subplot(111,aspect='equal')
plt.scatter(pr2_PL_arr,pr2_ML_arr,s=15)

xlim = plt.xlim()
plt.plot(xlim,xlim,'k')
plt.xlim(xlim)
plt.ylim(xlim)
plt.title('Cross validatted pseudo-r^2',fontsize=15)
plt.ylabel('ML filter pseudo-r^2',fontsize=12)
plt.xlabel('long filter pseudo-r^2',fontsize=12)
plt.tight_layout()

plt.figure(figsize=(14,10))


for k in range(fX_mat_PL.shape[0]):
    # row = k//5 + 1
    # col = k%5 + 1
    plt.subplot(5,10,k+1)
    time = (np.arange(fX_mat_PL.shape[1])-fX_mat_PL.shape[1]//2)*0.006
    sele = np.abs(fX_mat_PL[k,:] - fX_mat_PL[k,0])>10**-6

    plt.plot(time[sele],fX_mat_PL[k,sele])
    plt.title('pr2: %.3f'%pr2_PL[k])
# plt.tight_layout()

# plt.figure(figsize=(14,10))


# for k in range(fX_mat_PL.shape[0]):
#     # row = k//5 + 1
#     # col = k%5 + 1
#     plt.subplot(5,10,k+1)
    sele = np.abs(fX_mat_ML[k,:] - fX_mat_ML[k,0])>10**-6
    fs = deepcopy(fX_mat_ML[k,:].flatten())
    time = (np.arange(fs.shape[0])-fs.shape[0]//2)*0.006
    fs[~sele] = np.nan
    # plt.plot(time,fX_mat_PL[k,:])
    
    # plt.plot(time,fX_mat_ML[k,:]-fX_mat_ML[k,0] + fX_mat_PL[k,0])
    plt.plot(time[sele],fX_mat_ML[k,sele]-fX_mat_ML[k,0] + fX_mat_PL[k,0])

    
    # plt.plot(time[sele],fs[sele]-fX_mat_ML[k,0] + fX_mat_PL[k,0])
plt.tight_layout()
    
