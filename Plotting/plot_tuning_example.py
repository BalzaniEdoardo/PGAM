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


session = 'm53s91'
# load file
with open('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/results_jp/gam_m91s2/gam_fit_m91s2_c1_all_1.0000.dill') as fh:
    result_dict = dill.load(fh)
    
full = result_dict['gam_full']
reduced = result_dict['gam_reduced']

sele_glm  = np.argmax(result_dict['glm_pr2_test'])
glm_res = result_dict['glm_res_dict'][sele_glm]

sm_handler_glm = smooths_handler()

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
        
    sm_handler_glm = add_smooth(sm_handler_glm, x, var, knots, session, np.ones(len(x)), time_bin=0.006, lam=50)
    


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
for var in sm_handler_glm.smooths_var:
    ax_dict[var] = plt.subplot(4,4,k)
    
    X = sm_handler_glm[var].X.toarray()
    X = X[:,:-1] - np.mean(X[:,:-1],axis=0)
    X = np.hstack((np.ones((X.shape[0],1)),X))
    plt.title(var)
    pred = np.dot(X[:,1:],glm_res._results.params[full.index_dict[var]])
    plt.plot(pred,label='glm')
    
    k += 1
    
    

k = 1
for var in sm_handler_gam.smooths_var:
    ax = ax_dict[var]
    fX,fX_p,fX_m = reduced.smooth_compute(sm_handler_gam[var]._x,var,0.99)
    # X = sm_handler_gam[var].X.toarray()
    # X = X[:,:-1] - np.mean(X[:,:-1],axis=0)
    # X = np.hstack((np.ones((X.shape[0],1)),X))

    # pred = np.dot(X[:,1:],reduced.beta[reduced.index_dict[var]])
    xx = np.arange(fX.shape[0])
    ax.plot(xx,fX,color='r',label='gam reduced')
    ax.fill_between(xx,fX_m,fX_p,color='r',alpha=0.3)
    
    k += 1
    
k = 1
for var in sm_handler_gam_full.smooths_var:
    ax = ax_dict[var]
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
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('tuning_example.pdf')
                    