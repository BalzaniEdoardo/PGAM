#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:05:38 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(thisPath)),'GAM_Library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import scipy.stats as sts

# plot input distribution for all session and save the histograms
bs_fold = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

first = False
pattern = '^m\d+s\d+.npz$'

hist_size = 400
hist_matrix = np.zeros((0,hist_size),dtype=int)
edge_matrix = np.zeros((0,hist_size+1),dtype=np.float32)
info = np.zeros(0,dtype={'names':('session','variable'),'formats':('U20','U20')})
for root, dirs, names in os.walk(bs_fold):
    for name in names:
        if not re.match(pattern, name):
            continue
        print(name)
        
        dat = np.load(os.path.join(root,name), allow_pickle=True)
        session = name.split('.npz')[0]
        
        var_names = dat['var_names']
        concat = dat['data_concat'].all()
        Xt = concat['Xt']
        
        cnt_var = 0
        for var in var_names:
            if var.startswith('t_') or var.startswith('lfp_'):
                continue
            cnt_var += 1
        
        tmp_info = np.zeros(cnt_var, dtype={'names':('session','variable'),'formats':('U20','U20')})
        tmp_hist = np.zeros((cnt_var, hist_size))
        tmp_edge = np.zeros((cnt_var, hist_size+1))

        cc = 0
        for var in var_names:
            if var.startswith('t_') or var.startswith('lfp_'):
                continue
            idx = np.where(var_names == var)[0][0]
            xx = Xt[:,idx]
            if var == 'ang_target':
                xx = xx[np.abs(xx) <= 50]
            
            xx = xx[~np.isnan(xx)]
            if 'eye' in var:
                xx = xx[np.abs(xx)<np.nanpercentile(np.abs(xx),99)]
                xx = sts.zscore(xx)
            tmp_hist[cc,:],tmp_edge[cc,:] = np.histogram(xx,bins=hist_size)
            tmp_info[cc]['variable'] = var
            tmp_info[cc]['session'] = session
            cc+=1
        
        hist_matrix = np.vstack((hist_matrix, tmp_hist))
        edge_matrix = np.vstack((edge_matrix, tmp_edge))
        info = np.hstack((info,tmp_info))
        # break
np.savez('input_hist_acc.npz',hist=hist_matrix,edge=edge_matrix,info=info)
    #     first = True
    #     break
    # if first:
    #     break
    
        