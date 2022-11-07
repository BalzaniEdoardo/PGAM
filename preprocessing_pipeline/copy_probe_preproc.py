#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:38:46 2022

@author: edoardo
"""

import os
import shutil

base_path = '/Volumes/server/Data/Monkey2_newzdrive/Jimmy/U-probe/'
dest = '/Volumes/Balsip HD/dataset_firefly/'
for root, dirs, files in os.walk(base_path):
    if not (('PRE' in root.upper()) and ('PROC' in root.upper())):
        continue
    # if not '7a' in root:
    #     continue
    print(root)
    print(files)
    print('\n')
    
    for fh in files:
        if fh.endswith('.mat'):
            shutil.copy(os.path.join(root,fh),dest)
    