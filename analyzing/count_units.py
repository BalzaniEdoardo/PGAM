#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:28:36 2020

@author: edoardo
"""
import numpy as np
import os,re


bs_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET'
pattern = '^m\d+s\d+.npz$'   

SU_count = {'PPC':0,'PFC':0,'MST':0,'VIP':0}
MU_count = {'PPC':0,'PFC':0,'MST':0,'VIP':0}
for root, dirs, files in os.walk(bs_fld, topdown=False):
    for fh in files:
        if not re.match(pattern, fh):
            continue
        print(fh)
        dat = np.load(os.path.join(root,fh),allow_pickle=True)
        unit_info = dat['unit_info'].all()
        
        unit_type = unit_info['unit_type']
        cont_rate_filter = unit_info['cR']
        presence_rate_filter = unit_info['presence_rate']
        isi_v_filter = unit_info['isiV']
        
        cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
        presence_rate_filter = presence_rate_filter > 0.9
        isi_v_filter = isi_v_filter < 0.2
        su_filter = unit_type == 'singleunit'
        
        mu_filter = unit_type == 'multiunit'
        
        combine_filter = (su_filter) * (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
        combine_filter_MU = (mu_filter) * (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)


        brain_area = unit_info['brain_area'][combine_filter]
        
        for ba in np.unique(brain_area):
            SU_count[ba] = SU_count[ba] + (brain_area == ba).sum()
            
        brain_area = unit_info['brain_area'][combine_filter_MU]
        
        for ba in np.unique(brain_area):
            MU_count[ba] = MU_count[ba] + (brain_area == ba).sum()
    

