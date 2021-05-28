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
fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_history_filter_knotsnum/'
file_list = os.listdir(fh_folder)

counts_hand_vel = {'fewKnots':0,'manyKnots':0}
counts_all = 0
pr2_fewKnots = []
pr2_manyKnots = []


dtype_dict = {'names':('session','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2'),
              'formats':('U30','U4','U30',int,int,int,'U40',float)}


dtype_dict_info = {'names':('session','variable','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2 fewKnots','pr2 manyKnots','is significant fewKnots','is significant manyKnots',
                       'p-val fewKnots','p-val manyKnots'),
              'formats':('U30','U30','U4','U30',int,int,int,'U40',float,float,bool,bool,float,float)}



df_fewKnots = np.zeros(0,dtype=dtype_dict)
df_manyKnots = np.zeros(0,dtype=dtype_dict)
df_info = np.zeros(0,dtype=dtype_dict_info)
df_all = {}
first = True
for fhname in file_list:
    with open(os.path.join(fh_folder, fhname), 'rb') as fh:
        results = dill.load(fh)
    if results['reduced_manyKnots'] is None:
        pass
    elif ('hand_vel1' in results['reduced_manyKnots'].var_list) or\
        ('hand_vel2' in results['reduced_manyKnots'].var_list):
        counts_hand_vel['manyKnots'] = counts_hand_vel['manyKnots'] + 1
        
        tmp_manyKnots = np.zeros(1,dtype=dtype_dict)
        tmp_manyKnots['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_manyKnots['brain area'] = results['brain_area']
        tmp_manyKnots['unit type'] = results['unit_typ']
        tmp_manyKnots['cluster id'] = results['cluster_id']
        tmp_manyKnots['electrode id'] = results['electrode_id']
        tmp_manyKnots['channel id'] = results['channel_id']
        tmp_manyKnots['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_manyKnots['pr2'] = results['p_r2_manyKnots']
        df_manyKnots = np.hstack((df_manyKnots,tmp_manyKnots))

    
    if results['reduced_fewKnots'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_fewKnots'].var_list) or\
        ('hand_vel2' in results['reduced_fewKnots'].var_list):
        counts_hand_vel['fewKnots'] = counts_hand_vel['fewKnots'] + 1
        
        tmp_fewKnots = np.zeros(1,dtype=dtype_dict)
        tmp_fewKnots['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_fewKnots['brain area'] = results['brain_area']
        tmp_fewKnots['unit type'] = results['unit_typ']
        tmp_fewKnots['cluster id'] = results['cluster_id']
        tmp_fewKnots['electrode id'] = results['electrode_id']
        tmp_fewKnots['channel id'] = results['channel_id']
        tmp_fewKnots['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_fewKnots['pr2'] = results['p_r2_fewKnots']
        df_fewKnots = np.hstack((df_fewKnots,tmp_fewKnots))

    # if not results['reduced_fewKnots'] is None:  
        
        # print(results['reduced_fewKnots'].var_list)
    
    full_fewKnots = results['full_fewKnots']
    full_manyKnots =  results['full_manyKnots']
    
    tmp_info = np.zeros(len(full_fewKnots.var_list),dtype=dtype_dict_info)
    cc = 0
    for var in full_fewKnots.var_list:
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
        tmp_info['pr2 fewKnots'][cc] = results['p_r2_fewKnots']
        tmp_info['pr2 manyKnots'][cc] = results['p_r2_manyKnots']
        
        if results['reduced_fewKnots'] is None:
            tmp_info['is significant fewKnots'][cc] = False
            idx = results['full_fewKnots'].covariate_significance['covariate'] == var
            tmp_info['p-val fewKnots'][cc] = results['full_fewKnots'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_fewKnots'].var_list:
            tmp_info['is significant fewKnots'][cc] = False
            idx = results['full_fewKnots'].covariate_significance['covariate'] == var
            tmp_info['p-val fewKnots'][cc] = results['full_fewKnots'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_fewKnots'].covariate_significance['covariate'] == var
            if results['reduced_fewKnots'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant fewKnots'][cc] = bl
            tmp_info['p-val fewKnots'][cc] = results['reduced_fewKnots'].covariate_significance['p-val'][idx]

        if results['reduced_manyKnots'] is None:
            tmp_info['is significant manyKnots'][cc] = False
            idx = results['full_fewKnots'].covariate_significance['covariate'] == var
            tmp_info['p-val manyKnots'][cc] = results['full_manyKnots'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_manyKnots'].var_list:
            tmp_info['is significant manyKnots'][cc] = False
            idx = results['full_fewKnots'].covariate_significance['covariate'] == var
            tmp_info['p-val manyKnots'][cc] = results['full_manyKnots'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_manyKnots'].covariate_significance['covariate'] == var
            if results['reduced_manyKnots'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant manyKnots'][cc] = bl
            tmp_info['p-val manyKnots'][cc] = results['reduced_manyKnots'].covariate_significance['p-val'][idx]

            
        
        cc += 1
       
    df_info = np.hstack((df_info,tmp_info))
    pr2_fewKnots += [results['p_r2_fewKnots']]
    pr2_manyKnots += [results['p_r2_manyKnots']]
    
    
    kern_dim = full_fewKnots.smooth_info['spike_hist']['basis_kernel'].shape[0]
    if first:
        fX_mat_manyKnots = np.zeros((0,kern_dim))
        fX_mat_fewKnots = np.zeros((0,kern_dim))
        first = False

    impulse = np.zeros(kern_dim)
    impulse[(kern_dim - 1) // 2] = 1
    fX_fewKnots, _, _ = full_fewKnots.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    fX_manyKnots, _, _ = full_manyKnots.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
    
    fX_mat_fewKnots = np.vstack((fX_mat_fewKnots, fX_fewKnots.reshape((1,fX_fewKnots.shape[0]))))
    fX_mat_manyKnots = np.vstack((fX_mat_manyKnots, fX_manyKnots.reshape((1,fX_manyKnots.shape[0]))))


    counts_all += 1
    
pr2_manyKnots_arr = np.array(pr2_manyKnots)
pr2_fewKnots_arr = np.array(pr2_fewKnots)

keep = pr2_manyKnots_arr > np.nanpercentile(pr2_manyKnots,1)
pr2_manyKnots_arr = pr2_manyKnots_arr[keep]
pr2_fewKnots_arr = pr2_fewKnots_arr[keep]

plt.figure(figsize=(8,6))
plt.subplot(111,aspect='equal')
plt.scatter(pr2_fewKnots_arr,pr2_manyKnots_arr,s=15)

xlim = plt.xlim()
plt.plot(xlim,xlim,'k')
plt.xlim(xlim)
plt.ylim(xlim)
plt.title('Cross validatted pseudo-r^2',fontsize=15)
plt.ylabel('manyKnots filter pseudo-r^2',fontsize=12)
plt.xlabel('long filter pseudo-r^2',fontsize=12)
plt.tight_layout()

plt.figure(figsize=(14,10))


for k in range(fX_mat_fewKnots.shape[0]):
    # row = k//5 + 1
    # col = k%5 + 1
    plt.subplot(5,10,k+1)
    time = (np.arange(fX_mat_fewKnots.shape[1])-fX_mat_fewKnots.shape[1]//2)*0.006
    sele = np.abs(fX_mat_fewKnots[k,:] - fX_mat_fewKnots[k,0])>10**-6

    plt.plot(time[sele],fX_mat_fewKnots[k,sele])
    plt.title('pr2: %.3f'%pr2_fewKnots[k])
# plt.tight_layout()

# plt.figure(figsize=(14,10))


# for k in range(fX_mat_fewKnots.shape[0]):
#     # row = k//5 + 1
#     # col = k%5 + 1
#     plt.subplot(5,10,k+1)
    sele = np.abs(fX_mat_manyKnots[k,:] - fX_mat_manyKnots[k,0])>10**-6
    fs = deepcopy(fX_mat_manyKnots[k,:].flatten())
    time = (np.arange(fs.shape[0])-fs.shape[0]//2)*0.006
    fs[~sele] = np.nan
    # plt.plot(time,fX_mat_fewKnots[k,:])
    
    # plt.plot(time,fX_mat_manyKnots[k,:]-fX_mat_manyKnots[k,0] + fX_mat_fewKnots[k,0])
    plt.plot(time[sele],fX_mat_manyKnots[k,sele]-fX_mat_manyKnots[k,0] + fX_mat_fewKnots[k,0])

    
    # plt.plot(time[sele],fs[sele]-fX_mat_manyKnots[k,0] + fX_mat_fewKnots[k,0])
plt.tight_layout()
    
