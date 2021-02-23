#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:23:46 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy')

dtype_dict = {
    'names':('monkey','session','sender brain area','receiver brain area','sender unit id','receiver unit id',
             'manipulation type','value 1','value 2','value 3','sign 1','sign 2','sign 3'),
    'formats':('U30','U30','U30','U30',int,int,'U30',float,float,float,bool,bool,bool)
    }


conditions = ['odd']

monkey = 'Schro'

# sel = dat['monkey'] == 'Schro'
# dat = dat[sel]

table_result = np.zeros(0,dtype=dtype_dict)
for monkey in np.unique(dat['monkey']):
    dat_m  = dat[dat['monkey'] == monkey]
    for session in np.unique(dat_m['session']):
        dat_cond = dat_m[dat_m['session'] == session]
        # sel cond
        found = False
        for cond in np.unique(dat_cond['manipulation type']):
            if cond in conditions:
                found = True
                break
        assert(found)
        found = False
        dat_cond = dat_cond[dat_cond['manipulation type'] == cond]
        # cond = dat_cond[0]['manipulation type']
        print(cond.upper(),session)
            # dat_cond = dat_s[dat_s['manipulation type'] == cond]
        man_vals = np.unique(dat_cond['manipulation value'])
        dat_dict = {}
        for val in man_vals:
            dat_dict[val] = dat_cond[dat_cond['manipulation value'] == val]
            dat_dict[val] = np.sort(dat_dict[val],order=('sender unit id','receiver unit id'))
        
        table_result_tmp = np.zeros(dat_dict[man_vals[0]].shape, dtype=dtype_dict)
        cnt = 0
        for row in dat_dict[man_vals[0]]:
            if cnt % 500 ==0:
                print('%d/%d'%(cnt+1,dat_dict[man_vals[0]].shape[0]))
            table_result_tmp['monkey'][cnt] = row['monkey']
            table_result_tmp['session'][cnt] = row['session']
            table_result_tmp['sender brain area'][cnt] = row['sender brain area']
            table_result_tmp['receiver brain area'][cnt] = row['receiver brain area']
            table_result_tmp['manipulation type'][cnt] = row['manipulation type']
            
            table_result_tmp['receiver unit id'][cnt] = row['receiver unit id']
            table_result_tmp['sender unit id'][cnt] = row['sender unit id']
            
            table_result_tmp['value 1'][cnt] = row['manipulation value']
            table_result_tmp['sign 1'][cnt] = row['is significant']
            
            val_id = 1
            for val in dat_dict.keys():
                
                sel = (dat_dict[val]['sender unit id'] == row['sender unit id']) & \
                      (dat_dict[val]['receiver unit id'] == row['receiver unit id']) &\
                      (dat_dict[val]['session'] == row['session'])
                      
                row_other = dat_dict[val][sel]
                assert(row_other.shape[0] <= 1)
                if row_other.shape[0] == 0:
                    table_result_tmp['value %d'%val_id][cnt] = np.nan
                    table_result_tmp['sign %d'%val_id][cnt] = False
                
                else:
                    table_result_tmp['value %d'%val_id][cnt] = row_other['manipulation value']
                    table_result_tmp['sign %d'%val_id][cnt] = row_other['is significant']
                    
                val_id += 1
    
            cnt += 1
        table_result = np.hstack((table_result,table_result_tmp))
        print('################\n\n')
np.save('oddEven_coupling_x_cond.npy',table_result) 
    # cnt = 0
    # for row in dat_cond:
    #     table_result_tmp['monkey'] = row['monkey']
    #     table_result_tmp['monkey'] = row['monkey']
    #     table_result_tmp['monkey'] = row['monkey']
    #     table_result_tmp['monkey'] = row['monkey']
    #     table_result_tmp['monkey'] = row['monkey']
    #     table_result_tmp['monkey'] = row['monkey']
    #     table_result_tmp['monkey'] = row['monkey']
        
    # break