#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:26:29 2020

@author: edoardo
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts

variabels = [
 'rad_vel_resp_magn_full',
 'ang_vel_resp_magn_full',
 'rad_path_resp_magn_full',
 'ang_path_resp_magn_full',
 'rad_target_resp_magn_full',
 'ang_target_resp_magn_full',
 'lfp_beta_resp_magn_full',
 'lfp_alpha_resp_magn_full',
 'lfp_theta_resp_magn_full',
 't_move_resp_magn_full',
 't_flyOFF_resp_magn_full',
 't_stop_resp_magn_full',
 't_reward_resp_magn_full',
 'eye_vert_resp_magn_full',
 'eye_hori_resp_magn_full',
 'spike_hist_resp_magn_full',
 ]

info=['session','unit',
 'cluster_id',
 'electrode_id',
 'channel_id',
 'brain_area']
color_dict = {'MST':'g','PPC':'b','PFC':'r'}
tab = loadmat('/Users/edoardo/Work/Code/GAM_code/plotting/table_report.mat')
nunits = tab['table_report']['t_reward'].shape[1]
table_numpy = np.zeros((len(variabels),nunits))
monkey = np.zeros(nunits,dtype='U20')
brain_area = np.zeros(nunits,dtype='U20')

var_lab = []
for unt in range(nunits):
    cc = 0
    for var in variabels:
        var_lab += [' '.join(var.split('_')[:2])]
        table_numpy[cc,unt] = tab['table_report'][var][0,unt][0,0]
        cc+=1
    session = tab['table_report']['session'][0,unt][0]
    monkey[unt] = session.split('s')[0]
    brain_area[unt] = tab['table_report']['brain_area'][0,unt][0]
    

for mnk in np.unique(monkey):
    if mnk == 'm91':
        continue
    sele = monkey == mnk
    table_mnk = table_numpy[:,sele]
    brain_area_mnk = brain_area[sele]
    perc25 = np.nanpercentile(table_mnk,1,axis=1)
    perc75 = np.nanpercentile(table_mnk,99,axis=1)
    keep = ((table_mnk.T > perc25) * (table_mnk.T < perc75)).sum(axis=1)==16
    table_mnk = table_mnk[:,keep]
    table_mnk = sts.zscore(table_mnk,axis=1)
    brain_area_mnk = brain_area_mnk[keep]

    for ba in ['MST','PPC','PFC']:
        seleba = brain_area_mnk == ba
        if seleba.sum() == 0:
            continue
        table_ba = table_mnk[:,seleba]
        
        mn = np.nanmedian(table_ba,axis=1)
        print(mn,mnk,ba)
        YERR = np.zeros((2,perc25.shape[0]))
        YERR[0,:] = -np.nanpercentile(table_ba,25,axis=1)  + mn
        YERR[1,:] =  np.nanpercentile(table_ba,75,axis=1) - mn

        plt.errorbar(range(mn.shape[0]),mn,yerr=YERR,color=color_dict[ba])
plt.xticks(range(mn.shape[0]),var_lab,rotation=45)
plt.tight_layout()