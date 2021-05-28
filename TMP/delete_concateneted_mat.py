#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:46:09 2021

@author: edoardo
"""
import os
import re
import numpy as np
from send2trash import send2trash
concat_dir = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

concat_sess = []
lst = os.listdir(concat_dir)

for name in lst:
    concat_sess += [name.split('.')[0]]
    
del_list = []
pattern = 'm\d+s\d+'
for name in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'):
    if name.startswith('.'):
        continue
    src =  re.search(pattern,name)
    if src is None:
        continue
    session = name[src.start():src.end()]
    if session in concat_sess:
        del_list += [session]
        send2trash(os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/',name))
    
del_list = np.unique(del_list)