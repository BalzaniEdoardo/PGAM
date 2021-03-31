#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:17:36 2021

@author: edoardo
"""
import numpy as np
import os,sys,re

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')

from GAM_library import *
from data_handler import *

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno',
               'm72':'Marco'}

fh_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'
fh_list = os.listdir(fh_folder)

dtype_dict ={'names':('monkey','session','neuron','channel_id','electrode_id','cluster_id','brain_area',
                      'unit_type','manipulation_type','manipulation_value','firing_rate'),
             'formats':('U30','U20',int,int,int,int,'U20','U20','U20',float,float)}

if os.path.exists('firing_rate_info2.npy'):
    table_info = np.load('firing_rate_info2.npy')
else:
    table_info = np.zeros(0,dtype_dict)

done_sess = np.unique(table_info['session'])
for fh in fh_list:
    if not re.match('^m\d+s\d+.npz$',fh):
        continue
    
    
    if fh.split('.')[0] in  done_sess:
        continue
    print(fh)
    dat = np.load(os.path.join(fh_folder,fh),allow_pickle=True)
   
    unit_info = dat['unit_info'].all()
    unit_type = unit_info['unit_type']
    brain_area = unit_info['brain_area']
    channel_id = unit_info['channel_id']
    electrode_id = unit_info['electrode_id']
    cluster_id = unit_info['cluster_id']
    
    
    isiV = unit_info['isiV'] # % of isi violations 
    cR =  unit_info['cR'] # contamination rate
    presence_rate = unit_info['presence_rate'] # measure of stability of the firing in time
    
    # std filters for unit quality
    cont_rate_filter = (cR < 0.2) | (unit_type == 'multiunit')
    presence_rate_filter = presence_rate > 0.9
    isi_v_filter = isiV < 0.2
    combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
    
    # data
    concat = dat['data_concat'].all()
    spikes = concat['Yt']
    trial_idx = concat['trial_idx']
    
    # apply filters
    unit_type = unit_type[combine_filter]
    brain_area = brain_area[combine_filter]
    channel_id = channel_id[combine_filter]
    electrode_id = electrode_id[combine_filter]
    cluster_id = cluster_id[combine_filter]
    neuron_list = np.arange(1,spikes.shape[1]+1)[combine_filter]
    spikes = spikes[:,combine_filter]
    
    trial_type = dat['info_trial'].all().trial_type
    
    # create existing conditions
    cond_dict = {'all':[True],'odd':[0,1]}
    subsel_type = trial_type[trial_type['all']]
    for cond in ['reward','density','ptb','controlgain','replay']:
        # print(cond, np.unique(subsel_type[cond]))
        if len(np.unique(subsel_type[cond])) > 1:
            cond_dict[cond] = np.unique(subsel_type[cond])
    
    
    # create
    select_tr = np.zeros(spikes.shape[0],dtype=bool)
    for cond in cond_dict.keys():
        for val in cond_dict[cond]:
            select_tr = select_tr & False
            if cond != 'odd':
                tr_list = np.where(trial_type[cond] == val)[0]
            if cond == 'odd':
                all_trs = np.arange(trial_type.shape[0],dtype=int)
                all_trs = all_trs[trial_type['all']==1]
                if val == 1:
                    tr_list = all_trs[1::2]
                else:
                    tr_list = all_trs[::2]
                del all_trs
                
            for tr in tr_list:
                select_tr[trial_idx==tr] = True
            
            firing_rate = spikes[select_tr,:].mean(axis=0)/0.006
            tmp = np.zeros(spikes.shape[1],dtype=dtype_dict)
            tmp['monkey'] = monkey_dict[fh.split('s')[0]]
            tmp['session'] = fh.split('.')[0]
            tmp['neuron'] = neuron_list
            tmp['channel_id'] = channel_id
            tmp['electrode_id'] = electrode_id
            tmp['cluster_id'] = cluster_id
            tmp['brain_area'] = brain_area
            tmp['unit_type'] = unit_type
            tmp['manipulation_type'] = cond
            tmp['manipulation_value'] = val
            tmp['firing_rate'] = firing_rate
            
            table_info = np.hstack((table_info, tmp))

    np.save('firing_rate_info2.npy',table_info)
            
            
                