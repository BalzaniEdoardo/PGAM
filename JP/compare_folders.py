# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:52:33 2022

@author: lab
"""

import os
import numpy as np
import pandas as pd
dir_list_A = ['D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\C',
            'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\F',
            'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\S',
            'D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\N']

dir_list_B = ['T:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\C',
            'T:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\F',
            'T:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\S',
            'T:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\N']


size_A= 0
names_A  = []
sizes_A = []

for bsdir in dir_list_A:
    for root, dirs, files in os.walk(bsdir, topdown=False):
        print(root)
        for name in files:
            fp = os.path.join(root, name)
            if (not name.startswith('gam_fit_useCoup')) and ( name.endswith('.mat')):
                names_A.append(name)
                sizes_A.append(os.stat(fp).st_size)
                
            size_A += os.stat(fp).st_size

idx = np.argsort(names_A)
names_A = np.array(names_A)[idx]
sizes_A = np.array(sizes_A)[idx]            
            
size_B = 0
names_B  = []
sizes_B = []
for bsdir in dir_list_B:
    for root, dirs, files in os.walk(bsdir, topdown=False):
        print(root)
        for name in files:
            fp = os.path.join(root, name)
            if (not name.startswith('gam_fit_useCoup')) and ( name.endswith('.mat')):
                names_B.append(name)
                sizes_B.append(os.stat(fp).st_size)
            size_B += os.stat(fp).st_size
            
idx = np.argsort(names_B)
names_B = np.array(names_B)[idx]
sizes_B = np.array(sizes_B)[idx]


table = np.zeros(max(len(sizes_A),len(sizes_A)),dtype={'names':
                                                       ('comp 56 file','comp 53 file',
                                                       'comp 56 size','comp 53 size'),
                                                       'formats':('U200','U200',int,int)}
                 )
    
table['comp 56 size'][:len(sizes_A)] = sizes_A
table['comp 53 size'][:len(sizes_B)] = sizes_B
table['comp 56 file'][:len(sizes_A)] = names_A
table['comp 53 file'][:len(sizes_B)] = names_B

df = pd.DataFrame(table)
writer = pd.ExcelWriter('T:\MOUSE-ASD-NEURONS\data\compare_pc56_and_pc53.xlsx',engine='xlsxwriter')
df.to_excel(writer)
writer.save()
writer.close()