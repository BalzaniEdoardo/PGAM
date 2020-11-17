#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:01:36 2019

@author: edoardo
"""

import numpy as np
import sys, os
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(os.path.dirname(folder_name))
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
sys.path.append(os.path.join(main_dir, 'firefly_utils'))
from utils_loading import unpack_preproc_data, add_smooth
import itertools
import pandas as pd
from path_class import get_paths_class

user_paths = get_paths_class()


session = 'm53s110'

fhName = os.path.join(user_paths.get_path('local_concat'),'%s.npz'%session)
save_fld = ''


par_list = [ 'cR', 'presence_rate', 'isiV','unit_type','info_trial']

(cont_rate_filter, presence_rate_filter, isi_v_filter,
     unit_type,info_trial) = unpack_preproc_data(fhName, par_list)




# get the unit to include as input covariates
cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate_filter > 0.9
isi_v_filter = isi_v_filter < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

neuron_use = np.arange(1,isi_v_filter.shape[0]+1)[combine_filter]


# create all the conditions that you are interested to fit

#cond_dict = {'all':[True]}
#cond_dict = {'all':[True], 'ptb':[0, 1]}
#cond_dict = {'all':[True], 'controlgain':[1, 1.5, 2]}
cond_dict = {'all':[True], 'density':[0.0001, 0.005]}

dict_type = {'names':('neuron', 'condition', 'value'),'formats':(int,'U30',float)}

cond_list = np.zeros(0,dtype=dict_type)

for condition in cond_dict.keys():
    for value in cond_dict[condition]:
        if any(info_trial[condition]==value):
            tmp_cond_list = np.zeros(neuron_use.shape[0],dtype=dict_type)
            tmp_cond_list['neuron'] = neuron_use
            tmp_cond_list['condition'] = condition
            tmp_cond_list['value'] = value
            cond_list = np.hstack((cond_list,tmp_cond_list))
        else:
            print('trial of type %s %s not present'%(condition,value))

print(pd.DataFrame(tmp_cond_list)[:10])
np.save(os.path.join(save_fld, 'condition_list_%s.npy'%session),cond_list)

    