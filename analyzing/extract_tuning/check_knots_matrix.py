#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:23:39 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
from spline_basis_toolbox import *
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

path_gen = get_paths_class()

def unpack_name(name):
    session = re.findall('m\d+s\d+',name)[0]
    unitID = int(re.findall('_c\d+_',name)[0].split('_')[1].split('c')[1])
    man_type = re.findall('_c\d+_[a-z]+_',name)[0].split('_')[2]
    man_val = float(re.findall('\d+.\d\d\d\d',name)[0])
    return session,unitID,man_type,man_val

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}

    
path_to_gam = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'
dtype_dict = {
    'names': [
        'monkey','session', 'brain area','unit id', 'electrode id', 'cluster id','channel id', 'manipulation type','manipulation value', 'variable','first knot', 'last knot','knots num'
        ],
    'formats':
        [
        'U20','U20', 'U20', int, int, int,int,'U20',float, 'U20', float, float, int
            ]
    }

if os.path.exists('knot_num_and_range.npy'):
    info_all = np.load('knot_num_and_range.npy')
else:
    info_all = np.zeros(0,dtype=dtype_dict)

pattern = '^fit_results_m\d+s\d+_c\d+_[a-z]+_\d+.\d\d\d\d.dill$'
session_prev = ''
npz_path = None
first=True
for (root,dirs,files) in os.walk(path_to_gam):
    
    for name in files:
        if not re.match(pattern, name):
            continue
        session,unitID,man_type,man_val = unpack_name(name)
        if ((info_all['unit id'] == unitID) * (info_all['session'] == session)
            * (info_all['manipulation type'] == man_type)
            * (info_all['manipulation value'] == man_val)).sum() != 0:
            continue
        
        npz_path = path_gen.search_npz(session)

        if npz_path is None:
            continue
        if session != session_prev:
            print('sessoin',session)
            session_prev = session
            

           
            dat = np.load(os.path.join(npz_path, session+'.npz'), allow_pickle=True)
            unit_info = dat['unit_info'].all()
        
            # check filters for unit quality not needed because gam fit are only 
            # for qualty uints, but just in case
            cont_rate_filter = (unit_info['cR'] < 0.2) | (unit_info['unit_type']=='multiunit')
            presence_rate_filter = unit_info['presence_rate'] > 0.9
            isi_v_filter = unit_info['isiV'] < 0.2
            pp_size = isi_v_filter.shape[0]
            combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
        
        if not combine_filter[unitID - 1]:
            print('BAD unit: ',session, unitID)
            continue
        
        # open fits
        with open(os.path.join(root, name), 'rb') as fh:
            gam_res = dill.load(fh)
            full = gam_res['full']
            del gam_res
        
        cnt_vars = 0
        for var in full.var_list:
            if var == 'spike_hist' or var.startswith('neu'):
                continue
            cnt_vars += 1
        
        info_neu = np.zeros(cnt_vars, dtype=dtype_dict)
        info_neu['session'] = session
        info_neu['monkey'] = monkey_dict[session.split('s')[0]]
        info_neu['unit id'] = unitID
        info_neu['electrode id'] = unit_info['electrode_id'][unitID-1]
        info_neu['cluster id'] = unit_info['cluster_id'][unitID-1]
        info_neu['channel id'] = unit_info['channel_id'][unitID-1]
        info_neu['brain area'] = unit_info['brain_area'][unitID-1]
        
        info_neu['manipulation type'] = man_type
        info_neu['manipulation value'] = man_val

        

        
        cc = 0
        for var in full.var_list:
            if var == 'spike_hist' or var.startswith('neu'):
                continue
            
            knots = full.smooth_info[var]['knots'][0]
            knots_0 = knots[4]
            knots_end = knots[-4]
            knots_num = np.unique(knots).shape[0]
            info_neu['variable'][cc] = var
            info_neu['first knot'][cc] = knots_0
            info_neu['last knot'][cc] = knots_end
            info_neu['knots num'][cc] = knots_num
            cc += 1
            
        info_all = np.hstack((info_all,info_neu))

np.save('knot_num_and_range.npy',info_all)
            
                
        
            
            
            
        
        