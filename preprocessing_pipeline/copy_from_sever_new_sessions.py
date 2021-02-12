#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:22:52 2020

@author: edoardo
"""
import numpy as np
import pathlib
import os,re,sys
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc')
from path_class import get_paths_class
from datetime import datetime
path_gen = get_paths_class()
pattern = '^m\d+s\d+.npz$'

fld_name = 'Pre-processing X E'
min_date = datetime(2020,12,20)
cp_dir = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel'
concat_dir = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'
# for root, dirs, files in os.walk("/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET", topdown=False):
#     for name in files:
#         if not re.match(pattern,name):
#             continue

for session in ['m53s51']:
        # session = name.split('.')[0]
    if 'm91' in session:
            continue

    path = path_gen.get_path('cluster_array_data',session)
    path = os.path.dirname(os.path.dirname(path))
    full_path = os.path.join(path,fld_name)
    print(full_path)
    path_concat = os.path.join(concat_dir,session+'.npz')
    if os.path.exists(path_concat):
        continue
    
    try:
        for fh in os.listdir(full_path):
            
            if 'GAM' in fh:
                continue
            
            full_name = os.path.join(full_path, fh)
            
            
            fname = pathlib.Path(full_name)
            mtime = datetime.fromtimestamp(fname.stat().st_mtime)
            print(fh)
            if mtime < min_date:
                continue
            full_name = full_name.replace(' ', '\ ')
            if os.path.exists(os.path.join(cp_dir, fh)):
                continue
            os.system('cp %s %s' % (full_name, cp_dir))
            
    except FileNotFoundError:
        print(session,'not found')
        print(full_path)
    # bs_path = os.path.dirname(full_path)
    
    
    # if not  os.path.exists(path):
    #     print('NOT',path)
    # if not os.path.exists(full_path):
    #     print(session)
    #     print(full_path,)
        # full_path = full_path.replace(' ', '\ ')
        # os.system('cp %s %s' % (full_path, root))