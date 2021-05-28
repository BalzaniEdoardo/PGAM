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
from copy import deepcopy

cond_dict_all = {}
user_paths = get_paths_class()
sess_nopr = []
FIRST = True

for fh in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'):
    session = fh.split('.npz')[0]
    # print(session)
    # if session !='m53s42':
    #     continue
    # if session != 'm44s174':
    #     continue
#session = 'm53s91'
    sess_keep = ['m73s5']
    #
    if not session in sess_keep:
        continue

    fhName = os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/','%s.npz'%session)
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
    
    
    if np.sum(presence_rate_filter) == 0:
        sess_nopr += [session]
    # create all the conditions that you are interested to fit
    
    #cond_dict = {'all':[True]}
    #cond_dict = {'all':[True], 'ptb':[0, 1]}
    #cond_dict = {'all':[True], 'controlgain':[1, 1.5, 2]}
    cond_dict = {'all':[True]
                  }
                  #'ptb':[0,1]} #'density':[0.0001, 0.005]}
    # cond_dict = {'all': [True] }
    # cond_dict.pop('all')

    # if not (('m53' in session) or ('m44' in session )):
    #     continue
    # cond_dict = {'controlgain':[2,1.5]}
    dict_type = {'names':('neuron', 'condition', 'value'),'formats':(int,'U30',float)}
    
    cond_list = np.zeros(0,dtype=dict_type)
    if FIRST:
        for k in info_trial.dtype.names:
            cond_dict_all[k] = []
        FIRST = False
    for k in info_trial.dtype.names:
        cond_dict_all[k] = np.hstack((cond_dict_all[k],np.unique(info_trial[k][~np.isnan(info_trial[k])])))
        # print(k,np.unique(info_trial[k][~np.isnan(info_trial[k])]))
    
    for condition in cond_dict.keys():
        unq_cond = np.unique(info_trial[condition])
        unq_cond = unq_cond[unq_cond!=-1]
        unq_cond = unq_cond[~np.isnan(unq_cond)]
        if len(unq_cond) == 1 and condition != 'all':
            continue
        if not condition in ['all','reward','density']:
            xxxx = 1
        for value in cond_dict[condition]:
            
            if any(info_trial[condition]==value):
                tmp_cond_list = np.zeros(neuron_use.shape[0],dtype=dict_type)
                tmp_cond_list['neuron'] = neuron_use
                tmp_cond_list['condition'] = condition
                tmp_cond_list['value'] = value
                cond_list = np.hstack((cond_list,tmp_cond_list))
                
            else:
                print('trial of type %s %s not present'%(condition,value))
    # for condition in ['odd']:
    #
    #     for value in [0,1]:
    #
    #         if True:
    #             tmp_cond_list = np.zeros(neuron_use.shape[0],dtype=dict_type)
    #             tmp_cond_list['neuron'] = neuron_use
    #             tmp_cond_list['condition'] = condition
    #             tmp_cond_list['value'] = value
    #             cond_list = np.hstack((cond_list,tmp_cond_list))
    #
    #         else:
    #             print('trial of type %s %s not present'%(condition,value))
                
    # print(pd.DataFrame(tmp_cond_list)[:10])
    shape = cond_list.shape[0]
    all_num = (cond_list['condition'] == 'all').sum()
    
    quotient = shape//all_num
    one_every = min(quotient,5)
    # print(one_every)
    # if cond_list['condition']
    # if one_every > 0:
    #     cond_list2 = deepcopy(cond_list)
    #     cond_list2['neuron'] = -1
    #     cond_list2['condition'] = ''
    #     cond_list2['value'] = np.nan
    #     idx_all = np.where(cond_list['condition'] == 'all')[0]
    #     idx_other = np.where(cond_list['condition'] != 'all')[0]
    #     new_idx_all = np.arange(0,idx_all.shape[0]*one_every,one_every)
    #     new_idx_other = np.array(list(set(np.arange(0,cond_list.shape[0])).difference(set(new_idx_all))))
    #
    #     cond_list2['neuron'][new_idx_all] = cond_list['neuron'][idx_all]
    #     cond_list2['condition'][new_idx_all] = cond_list['condition'][idx_all]
    #     cond_list2['value'][new_idx_all] = cond_list['value'][idx_all]
    #
    #     cond_list2['neuron'][new_idx_other] = cond_list['neuron'][idx_other]
    #     cond_list2['condition'][new_idx_other] = cond_list['condition'][idx_other]
    #     cond_list2['value'][new_idx_other] = cond_list['value'][idx_other]
    #
    #     cond_list = cond_list2
    #
    # 
    # if 'ptb' in cond_list['condition'] or 'controlgain' in cond_list['condition']:
    # if not 'density' in cond_list['condition']:
    #     continue
    np.save(os.path.join(save_fld, 'condition_list_%s.npy'%session),cond_list)

    print('sbatch --array=1-%d gam_fit_%s.sh'%(len(cond_list),session))

    # print('\n',session,'unit num',len(tmp_cond_list),'\n')
    
for k in cond_dict_all.keys():
    cond_dict_all[k] = np.unique(cond_dict_all[k])

