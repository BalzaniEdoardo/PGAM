#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:18:36 2021

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
from statsmodels.distributions import ECDF

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}

path_gen = get_paths_class()
info_selectivity = np.load('response_strength_info.npy')
dat = np.load('eval_matrix_and_info.npz',allow_pickle=True)
        #eval_matrix=eval_matrix,info=info_matrix,index_list=index_list)
info_evals = dat['info']
eval_matrix = dat['eval_matrix']
var_list = dat['index_list']

variable = 'rad_vel'
monkey = 'm53'


# slice the variable of interest
tuning_matrix = eval_matrix[:, var_list == variable]


session_list = []#['m53s95','m53s98','m53s114','m53s115','m53s105',]
for session in np.unique(info_selectivity['session']):
    
    if not monkey in session:
        continue
    man_type_sess = np.unique(info_selectivity[info_selectivity['session']==session]['manipulation type'])
    if ('controlgain' in man_type_sess) or ('ptb' in man_type_sess in man_type_sess):
        continue
    session_list += [session]
        
    
man_type = 'odd'

similarity_all = np.zeros((2, 0))

# cnt_session = 0
for session in session_list:
    
    
    sele = ((info_evals['session'] == session) & 
            (info_evals['manipulation type']==man_type))
    
    sele_selctivity = ((info_selectivity['session']==session)*
                       (info_selectivity['manipulation type']==man_type))
    
    
    odd_info = info_selectivity[sele_selctivity * 
                            (info_selectivity['manipulation value']==1)]
    even_info = info_selectivity[sele_selctivity * 
                            (info_selectivity['manipulation value']==0)]
    
    # extract variables that are tuned to both
    tuned_odd = []
    tuned_even = []
    for row in odd_info:
        if row[variable]:
            tuned_odd += [row['unit']]
            
    for row in even_info:
        if row[variable]:
            tuned_even += [row['unit']]
            
    tuned_both = list(set(tuned_even).intersection(tuned_odd))
    
    # compute the similarity
    sim_session = np.zeros((2, len(tuned_both)))
    
    
    bool_odd = sele * (info_evals['manipulation value']==1)
    bool_even = sele * (info_evals['manipulation value']==0)
    
    shuffle_id = np.random.permutation(tuned_both)
    
    cc = 0
    for unit in tuned_both:
        tun_odd = tuning_matrix[bool_odd * (info_evals['unit id']==unit),:]
        tun_even = tuning_matrix[bool_even * (info_evals['unit id']==unit),:]
        tun_shuffle = tuning_matrix[bool_even * (info_evals['unit id']==shuffle_id[cc]),:]
        
        
        # in the case the unit has not been processed
        if tun_odd.shape[0] == 0 or tun_odd.shape[0] == 0:
            continue
        sim_session[0,cc] = sts.pearsonr(tun_odd[0],tun_even[0])[0]
        sim_session[1,cc] = sts.pearsonr(tun_odd[0],tun_shuffle[0])[0]
        
        cc+=1
    
    evals = eval_matrix[sele]
    resp = evals[:, var_list==variable]
    similarity_all = np.hstack((similarity_all,sim_session))
    # plt.plot(resp[0,:])
    # plt.plot(resp[1,:])
    # break


plt.figure(figsize=[4.73, 4.8 ])
plt.title('similarity '+variable)
x = np.linspace(-1,1,100)
ecdf_true = ECDF(similarity_all[0,:])
ecdf_sh = ECDF(similarity_all[1,:])
plt.plot(x,ecdf_true(x),label='odd/even',color='k')
plt.plot(x,ecdf_sh(x),label='shuffle',color=(0.5,)*3)
plt.legend()
plt.savefig('Figs/%s_sim_index_%s.png'%(monkey_dict[monkey],variable))

    
    