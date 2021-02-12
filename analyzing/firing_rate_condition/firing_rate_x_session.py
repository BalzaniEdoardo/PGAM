#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:34:01 2021

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
import venn

npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/eval_matrix_and_info.npz',allow_pickle=True)
info = dat['info']




condition_dict = {
    'controlgain':[1,1.5,2],
    'ptb':[0,1],
    'density':[0.005,0.0001]
    }

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}


dtype_dict = {'names':['monkey','session','brain area','channel id','electrode id','cluster id','condition','value','unit','rate [Hz]'],
                               'formats':['U30','U30','U30',int,int,int,'U30',float, int,float]}

firing_rate = np.zeros(0,dtype=dtype_dict)

for cond_sel in condition_dict.keys():
    val_sel = condition_dict[cond_sel][1]
    session_list= np.unique(info[(info['manipulation type'] == cond_sel)*(info['manipulation value'] == val_sel)]['session'])

    for session in session_list:
        print(session)
        inputData = np.load(os.path.join(npz_folder,session+'.npz'),allow_pickle=True)
        concat = inputData['data_concat'].all()
        y = concat['Yt']
        trial_idx = concat['trial_idx']
        info_trial = inputData['info_trial'].all().trial_type
        unit_info = inputData['unit_info'].all()
        
        cont_rate_filter = unit_info['cR']
        isi_v_filter = unit_info['isiV']
        presence_rate_filter = unit_info['presence_rate']
        unit_type = unit_info['unit_type']
    
        cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
        presence_rate_filter = presence_rate_filter > 0.9
        isi_v_filter = isi_v_filter < 0.2
        combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
        
        for cond in condition_dict.keys():
            for value in condition_dict[cond]:
                sel = np.where(info_trial[cond] == value)[0]
                keep = np.zeros(y.shape[0],dtype=bool)
                for tr in sel:
                    keep[trial_idx==tr] = True
                
                if len(sel) == 0:
                    continue
                
                tmp = np.zeros(combine_filter.sum(),dtype=dtype_dict)
                tmp['condition'] = cond
                tmp['session'] = session
                tmp['monkey'] = monkey_dict[session.split('s')[0]]
                tmp['value'] = value
                tmp['unit'] = np.arange(1,1+y.shape[1])[combine_filter]
                tmp['brain area'] = unit_info['brain_area'][combine_filter]
                
                tmp['electrode id'] = unit_info['electrode_id'][combine_filter]
                tmp['channel id'] = unit_info['channel_id'][combine_filter]
                tmp['cluster id'] = unit_info['cluster_id'][combine_filter]
    
                
                spk = y[:,combine_filter]
                spk = spk[keep]
                tmp['rate [Hz]'] = spk.mean(axis=0)/0.006
                firing_rate = np.hstack((firing_rate, tmp))
                
        np.save('firing_rate_x_cond.npy',firing_rate)
                
        
        