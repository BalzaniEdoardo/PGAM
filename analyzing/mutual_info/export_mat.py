#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:38:33 2021

@author: edoardo
"""
import numpy as np
import dill
import os,re
from copy import deepcopy
from scipy.io import savemat


fld_file = '/scratch/jpn5/mutual_info_lfp'
lst_done = os.listdir(fld_file)
# mutual_info_and_tunHz_m53s42.dill

first = True
for fh in lst_done:
    if not re.match('^mutual_info_and_tunHz_m\d+s\d+.dill$',fh):
        continue
    print(fh)
    with open(os.path.join(fld_file,fh),'rb') as fh:
        res = dill.load(fh)
        mi = res['mutual_info']
        tun = res['tuning_Hz']
    if first:
        mutual_info = deepcopy(mi)
        tuning = deepcopy(tun)
        first = False
    else:
        mutual_info = np.hstack((mutual_info,mi))
        tuning = np.hstack((tuning, tun))
  
np.save('/scratch/jpn5/mutual_info_lfp/tuning_func_LFP.npy',tuning)
np.save('/scratch/jpn5/mutual_info_lfp/mutual_info_LFP.npy',
        mutual_info)
print('done python')
savemat('/scratch/jpn5/mutual_info_lfp/tuning_func_LFP.mat',{'tuning':tuning})
savemat('//scratch/jpn5/mutual_info_lfp/mutual_info_LFP.mat',
        {'mutual_info':mutual_info})

