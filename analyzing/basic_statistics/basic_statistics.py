#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:30:47 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy

session = 'm53s113'
dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,allow_pickle=True)

concat = dat['data_concat'].all()
trial_idx = concat['trial_idx']
spikes = concat['Yt']
X = concat['Xt']

var_names = dat['var_names']


unit_info = dat['unit_info'].all()
unit_type = unit_info['unit_type']
isiV = unit_info['isiV'] # % of isi violations 
cR =  unit_info['cR'] # contamination rate
presence_rate = unit_info['presence_rate'] #



cont_rate_filter = (cR < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate > 0.9
isi_v_filter = isiV < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

units = np.arange(1,spikes.shape[1]+1)
units = units[combine_filter]
spikes = spikes[:,combine_filter]

t_tlyOFF = X[:, var_names=='t_flyOFF'].flatten()
untit_spike_times = {}

plt.figure(figsize=(10,6))
plt.suptitle(session)
cc=1
cnt_unit = 0
for unit in range(spikes.shape[1]):
    spk = spikes[:,unit]
    t_spk = []
    dur_trial = []
    if cc > 8:
        plt.figure(figsize=(10,6))
        cc=1
    plt.subplot(2,4,cc)
    
    for tr in np.unique(trial_idx):
        sele = trial_idx==tr
        ton_idx = np.where(t_tlyOFF[sele] == 1)[0][0] - 50
        t_spk += [np.where(spk[sele][ton_idx:] >0)[0]*0.006]
        # dur_trial += [sele.sum()]
    plt.eventplot(t_spk,lw=1.5,color='k')
    plt.title('unit %d'%units[cnt_unit])
    if cc > 4:
        plt.xlabel('time[sec]')
    if cc%4==1:
        plt.ylabel('trial num')
    cc+=1
    cnt_unit+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('%s_raster.png'%session)