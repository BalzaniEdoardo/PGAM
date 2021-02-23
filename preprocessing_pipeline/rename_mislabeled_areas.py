#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:51:44 2021

@author: edoardo
"""
## script to control that kernel are not forced to zero by the algorithm
import numpy as np
import sys, os, dill
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

from datetime import datetime
import zipfile



def saveCompressed(fh, **namedict):
     with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                          allowZip64=True) as zf:
         for k, v in namedict.items():
             with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                 np.lib.npyio.format.write_array(buf,
                                                 np.asanyarray(v),
                                                 allow_pickle=True)


user_paths = get_paths_class()
#datetime(2018,9,20)
date_list = [datetime(2018,9,25),datetime(2018,9,26)]

for date in date_list:
    session = user_paths.monkey_info.get_session(date)
    
    dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,allow_pickle=True)
    unit_info = dat['unit_info'].all()
    info_trial = dat['info_trial'].all()
    data_concat = dat['data_concat'].all()
    var_names = dat['var_names']
    time_bin = float(dat['time_bin'])
    pre_trial_dur = float(dat['pre_trial_dur'])
    post_trial_dur = float(dat['post_trial_dur'])
    unit_info['brain_area'][unit_info['brain_area']=='MST'] = 'VIP'
    
    saveCompressed(os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/','%s.npz'%session),unit_info=unit_info,
                   info_trial=info_trial,
                   data_concat=data_concat,
             var_names=var_names,
             time_bin=time_bin,post_trial_dur=post_trial_dur,
             pre_trial_dur=pre_trial_dur, force_zip64=True)