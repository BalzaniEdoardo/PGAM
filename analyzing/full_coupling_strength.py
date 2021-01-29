#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:33:36 2020

@author: edoardo
"""
## script to control that kernel are not forced to zero by the algorithm
import numpy as np
import sys, os, dill, re
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
# sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
sys.path.append(os.path.join(main_dir,os.path.join('preprocessing_pipeline','util_preproc')))
sys.path.append(os.path.join(main_dir,os.path.join('GAM_library')))
sys.path.append(os.path.join(main_dir,os.path.join('firefly_utils')))
from utils_loading import unpack_preproc_data, add_smooth
from GAM_library import *
from time import perf_counter
import statsmodels.api as sm
from basis_set_param_per_session import *
from knots_util import *
from path_class import get_paths_class
import pandas as pd
from scipy.io import savemat

consec_elect_dist_linear = 100
consec_elect_dist_utah = 400

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}


bruno_ppc_map = np.hstack(([np.nan],np.arange(1,9),[np.nan], np.arange(9,89), [np.nan], np.arange(89,97),[np.nan])).reshape((10,10))
electrode_map_dict = {
    'Schro': {'PPC': np.arange(1,49).reshape((8,6)), 'PFC': np.arange(49,97).reshape((8,6)),'MST':np.arange(1,25),'VIP':np.arange(1,25)},
    'Bruno': {'PPC': bruno_ppc_map},
    'Quigley':{'PPC':bruno_ppc_map,'MST':np.arange(1,25),'VIP':np.arange(1,25)}
    }
    


user_paths = get_paths_class()

fit_dir = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'

dtype_dict = {'names':('monkey','session','unit receiver','unit sender','cluster id receiver','cluster id sender','electrode id receiver','electrode id sender','brain area receiver','brain area sender','unit type receiver','unit type sender',
                       'beta','coupling strength','electrode distance','is significant'),
              'formats':('U20','U20',int,int,int,int,int,int,'U10','U10','U20','U20',object,float,float,bool)}


# skipUntil = 'm44s'
found = False
pattern = '^fit_results_m\d+s\d+_c\d+_all_1.0000.dill$'
all_coupling = np.zeros(0,dtype=dtype_dict)
for root, dirs, files in os.walk(fit_dir, topdown=False):
    for fhName in files:
        if re.match(pattern,fhName) is None:
            continue
        
        
        session = fhName.split('_')[2]
        mnk_id = session.split('s')[0]
        monkey = monkey_dict[mnk_id]
        
        # if skipUntil in session:
        #     found = True
            
        # if not found:
        #     continue
        
        # if monkey == 'Ody' or monkey == 'Quigley':
        #     continue
        
        
        path_npz = user_paths.search_npz( session)
        
        
        with open(os.path.join(root,fhName),'rb') as dill_fh:
            res_fit = dill.load(dill_fh)
        
        full = res_fit['full']
        if full is None:
            continue
        
        count_coupling = 0
    
        for var in full.var_list:
            if 'neu' in var:
                count_coupling += 1
            
            
        if count_coupling == 0:
            continue
        
        
        
        
        dat = np.load(os.path.join(path_npz,session+'.npz'),allow_pickle=True)
        unit_info = dat['unit_info'].all()
        unitA = int(fhName.split('_')[3].split('c')[1]) 
        
        print(monkey,session,unitA)
        
        brain_area = unit_info['brain_area']
        electrode_id = unit_info['electrode_id']
        cluster_id = unit_info['cluster_id']
        unit_type = unit_info['unit_type']
        
        
        
        neu_coupl = np.zeros(count_coupling,dtype=dtype_dict)
        neu_coupl['monkey'] = monkey
        neu_coupl['session'] = session
        
        
        cc = 0
        for var in full.var_list:
            if 'neu' in var:
                iisign = full.covariate_significance['covariate'] == var
                is_sign = full.covariate_significance['p-val'][iisign] < 0.001
                
                unitB = int(var.split('_')[1])
                idx_params = full.index_dict[var]
                betas = full.beta[idx_params]
                neu_coupl['unit receiver'][cc] = unitA
                neu_coupl['unit sender'][cc] = unitB
                neu_coupl['cluster id receiver'][cc] = cluster_id[unitA - 1]
                neu_coupl['cluster id sender'][cc] = cluster_id[unitB - 1]
                neu_coupl['electrode id receiver'][cc] = electrode_id[unitA - 1]
                neu_coupl['electrode id sender'][cc] = electrode_id[unitB - 1]
                neu_coupl['unit type receiver'][cc] = unit_type[unitA - 1]
                neu_coupl['unit type sender'][cc] = unit_type[unitB - 1]
                neu_coupl['brain area receiver'][cc] = brain_area[unitA - 1]
                neu_coupl['brain area sender'][cc] = brain_area[unitB - 1]
                neu_coupl['beta'][cc] = betas
                # neu_coupl['beta 1'][cc] = betas[1]
                # neu_coupl['beta 2'][cc] = betas[2]
                # neu_coupl['beta 3'][cc] = betas[3]
                neu_coupl['coupling strength'][cc] = np.linalg.norm(betas)
                neu_coupl['is significant'][cc] = is_sign
                
                
                
                # compute distance
                if brain_area[unitA - 1] != brain_area[unitB - 1]:
                    distance = np.nan
                    
                elif brain_area[unitA - 1] == 'MST' or brain_area[unitA - 1] == 'VIP':
                    distance = np.abs((electrode_id[unitA - 1] - electrode_id[unitB - 1]) * consec_elect_dist_linear)
                  
                else:
                    ba = brain_area[unitA - 1]
                    x_pos_A,y_pos_A = np.where(electrode_map_dict[monkey][ba] ==  electrode_id[unitA - 1])
                    x_pos_B,y_pos_B = np.where(electrode_map_dict[monkey][ba] ==  electrode_id[unitB - 1])
                    
                    distance = np.sqrt(((x_pos_A-x_pos_B)*consec_elect_dist_utah)**2 + ((y_pos_A-y_pos_B)*consec_elect_dist_utah)**2)
                neu_coupl['electrode distance'][cc] = distance
                cc += 1
                
                
                
        all_coupling = np.hstack((all_coupling,neu_coupl))    

np.save('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/fullFit_coupling_results.npy',all_coupling)
savemat('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/fullFit_coupling_results.mat',mdict={'coupling_results':all_coupling})
# dists = np.unique(all_coupling['electrode distance'][])      
# mn_cpl_str = np.zeros(dists.shape[0])   
# std_cpl_str = np.zeros(dists.shape[0])  
# cc =0
# for dd in dists:
#     sele = all_coupling['electrode distance'] == dd
#     mn_cpl_str[cc] = all_coupling['coupling strength'][sele].mean()
#     std_cpl_str[cc] = all_coupling['coupling strength'][sele].std()
    
#     cc+=1

# plt.scatter(all_coupling['electrode distance'],all_coupling['coupling strength'])
             
   
    