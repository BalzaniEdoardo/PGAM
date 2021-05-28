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

fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_select_hand_vel/'
file_list = os.listdir(fh_folder)

counts_hand_vel = {'spatial':0,'temporal':0}
counts_all = 0
pr2_spatial = []
pr2_temporal = []


dtype_dict = {'names':('session','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2'),
              'formats':('U30','U4','U30',int,int,int,'U40',float)}


dtype_dict_info = {'names':('session','variable','brain area','unit type','electrode id',
                       'channel id','cluster id', 'ID','pr2 spatial','pr2 temporal','is significant spatial','is significant temporal',
                       'p-val spatial','p-val temporal'),
              'formats':('U30','U30','U4','U30',int,int,int,'U40',float,float,bool,bool,float,float)}



df_spatial = np.zeros(0,dtype=dtype_dict)
df_temporal = np.zeros(0,dtype=dtype_dict)
df_info = np.zeros(0,dtype=dtype_dict_info)
df_all = {}

for fhname in file_list:
    with open(os.path.join(fh_folder, fhname), 'rb') as fh:
        results = dill.load(fh)
    if results['reduced_temporal'] is None:
        pass
    elif ('hand_vel1' in results['reduced_temporal'].var_list) or\
        ('hand_vel2' in results['reduced_temporal'].var_list):
        counts_hand_vel['temporal'] = counts_hand_vel['temporal'] + 1
        
        tmp_temporal = np.zeros(1,dtype=dtype_dict)
        tmp_temporal['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_temporal['brain area'] = results['brain_area']
        tmp_temporal['unit type'] = results['unit_typ']
        tmp_temporal['cluster id'] = results['cluster_id']
        tmp_temporal['electrode id'] = results['electrode_id']
        tmp_temporal['channel id'] = results['channel_id']
        tmp_temporal['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_temporal['pr2'] = results['p_r2_temporal']
        df_temporal = np.hstack((df_temporal,tmp_temporal))

    
    if results['reduced_spatial'] is None:  
        pass
    elif ('hand_vel1' in results['reduced_spatial'].var_list) or\
        ('hand_vel2' in results['reduced_spatial'].var_list):
        counts_hand_vel['spatial'] = counts_hand_vel['spatial'] + 1
        
        tmp_spatial = np.zeros(1,dtype=dtype_dict)
        tmp_spatial['session'] = fhname.split('vel_')[1].split('_')[0]
        tmp_spatial['brain area'] = results['brain_area']
        tmp_spatial['unit type'] = results['unit_typ']
        tmp_spatial['cluster id'] = results['cluster_id']
        tmp_spatial['electrode id'] = results['electrode_id']
        tmp_spatial['channel id'] = results['channel_id']
        tmp_spatial['ID'] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_spatial['pr2'] = results['p_r2_spatial']
        df_spatial = np.hstack((df_spatial,tmp_spatial))

    # if not results['reduced_spatial'] is None:  
        
        # print(results['reduced_spatial'].var_list)
    
    full_spatial = results['full_spatial']
    full_temporal =  results['full_temporal']
    
    tmp_info = np.zeros(len(full_spatial.var_list),dtype=dtype_dict_info)
    cc = 0
    for var in full_spatial.var_list:
        tmp_info['session'][cc] = fhname.split('vel_')[1].split('_')[0]
        tmp_info['variable'][cc] = var
        tmp_info['brain area'][cc] = results['brain_area']
        tmp_info['unit type'][cc] = results['unit_typ']
        tmp_info['cluster id'][cc] = results['cluster_id']
        tmp_info['electrode id'][cc] = results['electrode_id']
        tmp_info['channel id'][cc] = results['channel_id']
        tmp_info['ID'][cc] = (fhname.split('vel_')[1].split('_')[0] + 
            '_ch%d_el%d_cl%d_ba%s'%(results['channel_id'],results['electrode_id'],
                                    results['cluster_id'],results['brain_area']))
        tmp_info['pr2 spatial'][cc] = results['p_r2_spatial']
        tmp_info['pr2 temporal'][cc] = results['p_r2_temporal']
        
        if results['reduced_spatial'] is None:
            tmp_info['is significant spatial'][cc] = False
            idx = results['full_spatial'].covariate_significance['covariate'] == var
            tmp_info['p-val spatial'][cc] = results['full_spatial'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_spatial'].var_list:
            tmp_info['is significant spatial'][cc] = False
            idx = results['full_spatial'].covariate_significance['covariate'] == var
            tmp_info['p-val spatial'][cc] = results['full_spatial'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_spatial'].covariate_significance['covariate'] == var
            if results['reduced_spatial'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant spatial'][cc] = bl
            tmp_info['p-val spatial'][cc] = results['reduced_spatial'].covariate_significance['p-val'][idx]

        if results['reduced_temporal'] is None:
            tmp_info['is significant temporal'][cc] = False
            idx = results['full_spatial'].covariate_significance['covariate'] == var
            tmp_info['p-val temporal'][cc] = results['full_temporal'].covariate_significance['p-val'][idx]
            
        elif not var in results['reduced_temporal'].var_list:
            tmp_info['is significant temporal'][cc] = False
            idx = results['full_spatial'].covariate_significance['covariate'] == var
            tmp_info['p-val temporal'][cc] = results['full_temporal'].covariate_significance['p-val'][idx]
           
        else:
            bl = False
            idx = results['reduced_temporal'].covariate_significance['covariate'] == var
            if results['reduced_temporal'].covariate_significance['p-val'][idx] < 0.001:
                bl = True
            tmp_info['is significant temporal'][cc] = bl
            tmp_info['p-val temporal'][cc] = results['reduced_temporal'].covariate_significance['p-val'][idx]

            
        
        cc += 1
       
    df_info = np.hstack((df_info,tmp_info))
    pr2_spatial += [results['p_r2_spatial']]
    pr2_temporal += [results['p_r2_temporal']]

    counts_all += 1
    
pr2_temporal_arr = np.array(pr2_temporal)
pr2_spatial_arr = np.array(pr2_spatial)

keep = pr2_temporal_arr > np.nanpercentile(pr2_temporal,1)
pr2_temporal_arr = pr2_temporal_arr[keep]
pr2_spatial_arr = pr2_spatial_arr[keep]

plt.figure(figsize=(8,6))
plt.subplot(111,aspect='equal')
plt.scatter(pr2_spatial_arr,pr2_temporal_arr,s=10)

xlim = plt.xlim()
plt.plot(xlim,xlim,'k')
plt.xlim(xlim)
plt.ylim(xlim)

var_list = ['rad_vel','ang_vel','rad_acc','ang_acc','hand_vel1','hand_vel2',
            'rad_target','ang_target','rad_path','ang_path',
            't_stop','t_move','t_reward',
            'lfp_beta','lfp_alpha','lfp_theta','eye_hori','eye_vert']

mst_frac = np.zeros(len(var_list))
ppc_frac = np.zeros(len(var_list))
pfc_frac = np.zeros(len(var_list))

cc = 0
for var in var_list:
    filt = (df_info['brain area'] == 'MST') * (df_info['variable'] == var) 
    mst_frac[cc] = (df_info['is significant spatial'][filt]).sum()/filt.sum()
    
    filt = (df_info['brain area'] == 'PPC') * (df_info['variable'] == var) 
    ppc_frac[cc] = (df_info['is significant spatial'][filt]).sum()/filt.sum()
    
    filt = (df_info['brain area'] == 'PFC') * (df_info['variable'] == var) 
    pfc_frac[cc] = (df_info['is significant spatial'][filt]).sum()/filt.sum()
    cc+=1

x_mst = np.arange(0, 5*len(var_list))[::5]
x_ppc = np.arange(0, 5*len(var_list))[1::5]
x_pfc = np.arange(0, 5*len(var_list))[2::5]

plt.figure(figsize=(14,8))
ax = plt.subplot(211)
plt.title('spatial')

plt.bar(x_mst, mst_frac, width=1, color='g')
plt.bar(x_ppc, ppc_frac, width=1, color='b')
plt.bar(x_pfc, pfc_frac, width=1, color='r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


mst_frac = np.zeros(len(var_list))
ppc_frac = np.zeros(len(var_list))
pfc_frac = np.zeros(len(var_list))
plt.xticks(x_ppc, var_list,rotation=90)

cc = 0
for var in var_list:
    filt = (df_info['brain area'] == 'MST') * (df_info['variable'] == var) 
    mst_frac[cc] = (df_info['is significant temporal'][filt]).sum()/filt.sum()
    
    filt = (df_info['brain area'] == 'PPC') * (df_info['variable'] == var) 
    ppc_frac[cc] = (df_info['is significant temporal'][filt]).sum()/filt.sum()
    
    filt = (df_info['brain area'] == 'PFC') * (df_info['variable'] == var) 
    pfc_frac[cc] = (df_info['is significant temporal'][filt]).sum()/filt.sum()
    cc+=1
    
ax = plt.subplot(212)
plt.title('temporal')
plt.bar(x_mst, mst_frac, width=1, color='g')
plt.bar(x_ppc, ppc_frac, width=1, color='b')
plt.bar(x_pfc, pfc_frac, width=1, color='r')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(x_ppc, var_list,rotation=90)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

  




    