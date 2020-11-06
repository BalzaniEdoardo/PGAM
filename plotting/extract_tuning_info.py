#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:36:06 2020

@author: edoardo
"""
import numpy as np
import sys, os, dill
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
# sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline/util_preproc'))
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils/'))
from utils_loading import unpack_preproc_data, add_smooth
from GAM_library import *
from time import perf_counter
import statsmodels.api as sm
from basis_set_param_per_session import *
from knots_util import *
from path_class import get_paths_class
import statsmodels.api as sm
import matplotlib.pylab as plt
from copy import deepcopy
from time import perf_counter
from scipy.io import savemat


def get_npz_filepath(basefld, session):
    fh_name = session + '.npz'
    for root, dirs, files in os.walk(basefld):
        if fh_name in files:
            return os.path.join(root,fh_name)
    return None

session = 'm91s24'
unit = 15
# load file
with open('/Volumes/WD Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/gam_%s/gam_fit_%s_c%d_all_1.0000.dill'%(session,session,unit),'rb') as fh:
    result_dict = dill.load(fh)
    
full = result_dict['full']
reduced = result_dict['reduced']



sm_handler_gam = smooths_handler()

for var in reduced.var_list:
    knots = reduced.smooth_info[var]['knots'][0]
    xmin = reduced.smooth_info[var]['xmin'][0]
    xmax = reduced.smooth_info[var]['xmax'][0]
    if not reduced.smooth_info[var]['is_temporal_kernel']:
        x = np.linspace(xmin,xmax,100)
    else:
        dim_kern = reduced.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = reduced.smooth_info[var]['knots'][0].shape[0]
        x = np.zeros(dim_kern)
        x[(dim_kern-1)//2] = 1
        
    sm_handler_gam = add_smooth(sm_handler_gam, x, var, knots, session, np.ones(len(x)), time_bin=0.006, lam=50)
    
    

sm_handler_gam_full = smooths_handler()

for var in full.var_list:
    knots = full.smooth_info[var]['knots'][0]
    xmin = full.smooth_info[var]['xmin'][0]
    xmax = full.smooth_info[var]['xmax'][0]
    if not full.smooth_info[var]['is_temporal_kernel']:
        x = np.linspace(xmin,xmax,100)
    else:
        dim_kern = full.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = full.smooth_info[var]['knots'][0].shape[0]
        x = np.zeros(dim_kern)
        x[(dim_kern-1)//2] = 1
        
    sm_handler_gam_full = add_smooth(sm_handler_gam_full, x, var, knots, session, np.ones(len(x)), time_bin=0.006, lam=50)
    
    
    
    
plt.figure(figsize=(12,10))

ax_dict = {}

k = 1
for var in sm_handler_gam_full.smooths_var:
    if var.startswith('neu'):
        continue
    ax_dict[var] = plt.subplot(4,4,k)
    ax = ax_dict[var]
    ax.set_title(var)
    
    fX,fX_p,fX_m = full.smooth_compute(sm_handler_gam_full[var]._x,var,0.99)
    # X = sm_handler_gam[var].X.toarray()
    # X = X[:,:-1] - np.mean(X[:,:-1],axis=0)
    # X = np.hstack((np.ones((X.shape[0],1)),X))

    # pred = np.dot(X[:,1:],reduced.beta[reduced.index_dict[var]])
    xx = np.arange(fX.shape[0])
    ax.plot(xx,fX,color='g',label='gam full')
    if not var in reduced.var_list:
        ax.fill_between(xx,fX_m,fX_p,color='g',alpha=0.3)
    if var == 'spike_hist':
        ax.legend()
    ax.set_xticks([])
    
    
    k += 1
    
k = 1
for var in sm_handler_gam.smooths_var:
    if var.startswith('neu'):
        continue
    
    ax = ax_dict[var]
    fX,fX_p,fX_m = reduced.smooth_compute(sm_handler_gam[var]._x,var,0.99)
    
    xx = np.arange(fX.shape[0])
    idx = np.where(reduced.covariate_significance['covariate'] == var)[0]
    if reduced.covariate_significance['p-val'][idx]<0.001:
        color='r'
    else:
        color='k'
    ax.plot(xx,fX,color=color,label='gam reduced')
    
    ax.fill_between(xx,fX_m,fX_p,color=color,alpha=0.3)
    
    k += 1
    

    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# plt.savefig('tuning_example.pdf')

# # GET NEURON INFO
# dat = np.load('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/%s.npz'%session,allow_pickle=True)
# unit_info = dat['unit_info'].all()
# cluster_id = unit_info['cluster_id']
# electrode_id = unit_info['electrode_id']
# brain_area = unit_info['brain_area']
# channel_id = unit_info['channel_id']


variables = np.copy(full.var_list)
# keep = np.ones(variables.shape[0],dtype=bool)
# k = 0
# for var in full.var_list:
#     if var.startswith('neu_'):
#         keep[k] = False
#     k+=1

# variables = variables[keep]
# dict_type = {
#     'names':('session','unit','cluster_id','electrode_id','channel_id','brain_area',)+tuple(variables),
#     'formats':('U30',) + (int,)*4 + ('U3',) + (bool,)*variables.shape[0]
#         }
# table_report = np.zeros(1,dtype=dict_type)

# table_report['session'][0] = session
# table_report['unit'][0] = unit

# # matlab indexing was used for the name
# table_report['cluster_id'][0] = cluster_id[unit-1]
# table_report['channel_id'][0] = channel_id[unit-1]
# table_report['electrode_id'][0] = electrode_id[unit-1]
# table_report['brain_area'][0] = brain_area[unit-1]

# for var in reduced.var_list:
#     if var.startswith('neu'):
#         continue
#     idx = np.where(reduced.covariate_significance['covariate'] == var)[0]
#     if reduced.covariate_significance['p-val'][idx]>0.001:
#         table_report[var][0] = True
        
# # if you want to stack another neuron you can do this
# tmp = np.zeros(1,dtype=dict_type)
# # insert the info and then
# table_report = np.hstack((table_report,tmp))


# # this saves it as a struct
# savemat('table_report.mat',{'table_report':table_report})



dict_type = {
                'names':('session','unit','cluster_id','electrode_id','channel_id','brain_area',)+tuple(variables),
                'formats':('U30',) + (int,)*4 + ('U3',) + (bool,)*variables.shape[0]
                    }
table_report = np.zeros(0,dtype=dict_type)


for gam_sess in os.listdir('/Volumes/WD Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/'):
    if not gam_sess.startswith('gam'):
        continue
    session = gam_sess.split('_')[1]

    for dill_name in os.listdir(os.path.join('/Volumes/WD Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/',gam_sess)):
        print(dill_name)
        if not 'all' in dill_name:
            continue
        try:
            with open(os.path.join('/Volumes/WD Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/',gam_sess,dill_name),'rb') as fh:
                result_dict = dill.load(fh)
                
            tmp = np.zeros(1,dtype=dict_type)
            
            full = result_dict['full']
            reduced = result_dict['reduced']
            
            npz_file = get_npz_filepath('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/', session)
            dat = np.load(npz_file,allow_pickle=True)
            unit_info = dat['unit_info'].all()
            cluster_id = unit_info['cluster_id']
            electrode_id = unit_info['electrode_id']
            brain_area = unit_info['brain_area']
            channel_id = unit_info['channel_id']


            variables = np.copy(full.var_list)
            keep = np.ones(variables.shape[0],dtype=bool)
            k = 0
            for var in full.var_list:
                if var.startswith('neu_'):
                    keep[k] = False
                k+=1
            
            variables = variables[keep]
            
            table_report_tmp = np.zeros(1,dtype=dict_type)

            
            table_report_tmp['session'][0] = session
            table_report_tmp['unit'][0] = unit
            
            # matlab indexing was used for the name
            table_report_tmp['cluster_id'][0] = cluster_id[unit-1]
            table_report_tmp['channel_id'][0] = channel_id[unit-1]
            table_report_tmp['electrode_id'][0] = electrode_id[unit-1]
            table_report_tmp['brain_area'][0] = brain_area[unit-1]
            
            for var in reduced.var_list:
                if var.startswith('neu'):
                    continue
                idx = np.where(reduced.covariate_significance['covariate'] == var)[0]
                if reduced.covariate_significance['p-val'][idx]>0.001:
                    table_report_tmp[var][0] = True
                    
            # if you want to stack another neuron you can do this
            # insert the info and then
            table_report = np.hstack((table_report,table_report_tmp))
            
            
            # this saves it as a struct
           
                
                
                
        except:
            print("can't open ",os.path.join('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/results_jp/',gam_sess,dill_name))
        
            
    
savemat('table_report.mat',{'table_report':table_report})            