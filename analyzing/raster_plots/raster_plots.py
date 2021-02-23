#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:24:51 2021

@author: edoardo
"""
import numpy as np
import sys,os,dill,inspect
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
mainDir = os.path.dirname(os.path.dirname(thisPath))
sys.path.append(os.path.join(mainDir,'GAM_library'))
sys.path.append(os.path.join(mainDir,'firefly_utils'))
sys.path.append(os.path.join(mainDir,'preprocessing_pipeline','util_preproc'))
from spike_times_class import spike_counts
from lfp_class import lfp_class
from copy import deepcopy
from scipy.io import loadmat,savemat
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PIL import ImageColor
from data_handler import *
from path_class import get_paths_class
from scipy.io import loadmat
gen_path = get_paths_class()


def dict_to_vec(dictionary):
    return np.hstack(list(dictionary.values()))


# session = 'm53s91'
session_list = ['m53s113', 'm53s83', 'm53s114', 'm53s86',
                'm53s128', 'm53s123', 'm53s124', 'm53s47', 'm53s46', 'm53s31']

session = 'm72s1'

# base_file = os.path.join'/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/%s/%s.mat'
# base_file =  '/Users/edoardo/Downloads/%s/%s.mat'
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'
use_left_eye = ['m53s83','m53s84','m53s86','m53s90','m53s92','m53s133','m53s134']

trajectory_all_session = np.zeros(0,dtype=object)

dtype_initial_cond = {'names':('x_fly','y_fly','rad_vel','ang_vel','ang_target','rad_target'),'formats':(float,float,float,float,float,float)}
init_cond = np.zeros(0,dtype=dtype_initial_cond)
dtype_dict = {'names':('session','all', 'reward', 'density', 'ptb', 'microstim', 'landmark', 'replay','controlgain','firefly_fullON'),
            'formats':('U20',bool,int,float,int,int,int,int,float,int)}
info_all_session = np.zeros(0,dtype=dtype_dict)

path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/%s.mat'%session
dat = loadmat(path)

#
# trials_behv = dat[behav_dat_key].flatten()
# flyon_full = trials_behv['logical'][0]['firefly_fullON'][0,0].flatten()
pre_trial_dur = 0.2
post_trial_dur = 0.2
is_phase = False
if session in use_left_eye:
    use_eye = 'left'
else:
    use_eye = 'right'
    
exp_data = data_handler(dat, behav_dat_key, spike_key, None, behav_stat_key, pre_trial_dur=pre_trial_dur,
                        post_trial_dur=post_trial_dur,
                        extract_lfp_phase=False,
                        use_eye=use_eye,extract_fly_and_monkey_xy=True)

exp_data.set_filters('all',True)
window_sec = 0.4


events = ['t_move','t_flyOFF', 't_reward','t_stop']



raster = {}
for event in events:
    raster[event] = {}
    unit = 10
    time_event = getattr(exp_data.behav.events, event)
    
    
    
    for unit in range(exp_data.spikes.spike_times.shape[0]):
        raster[event][unit] = []
        
    for tr in np.where(exp_data.filter)[0]:
        if tr > 1500:
            break
        # print(tr)
        tp0 = exp_data.behav.time_stamps[tr][0]
        tp1 = exp_data.behav.time_stamps[tr][-1]
        if event == 't_flyOFF':
            ddt = 0.3
        else:
            ddt = 0
            
        t_event = time_event[tr] - ddt
        if any(np.isnan(t_event)):
            continue
        assert((t_event-tp0) >= window_sec)
        assert((tp1-t_event) >= window_sec)
    
        
        
        for unit in range(exp_data.spikes.spike_times.shape[0]):
            t_spk = exp_data.spikes.spike_times[unit,tr] - t_event
            if event == 't_flyOFF':
                t_spk = t_spk[(t_spk <= 2*(window_sec))*(t_spk >= 0)]
            else:
                t_spk = t_spk[(t_spk >= (-window_sec)) & (t_spk <= (window_sec))]
            
            raster[event][unit] += [t_spk]
    
    # print(t_event-tp0)


# brain_area = unit_info['brain_area']
color = {'MST':'g','PPC':'b','PFC':'r','VIP':'k'}
ccplot = 41
fig = None
first = True
pltcnt = 0
for unit in range(exp_data.spikes.spike_times.shape[0]):
    brain_area = exp_data.spikes.brain_area[unit]
    if ccplot == 41:
            ccplot = 1
            pltcnt +=1
            if not fig is None:
                plt.tight_layout()
                plt.savefig('%s_%d_mean_rate.png'%(session,pltcnt))
            # if not first:
            #     break
            # first=False
            fig = plt.figure(figsize=(10,12))  
            
    for event in events:
        print(unit,ccplot)

            
        plt.subplot(10,4,ccplot)
        if ccplot <= 4:
            plt.title(event)
        if ccplot % 4 == 1:
            plt.ylabel('unit %d'%(unit+1))
        
        vv = np.hstack((raster[event][unit]))
        if event !='t_flyOFF':
            cnt,edg = np.histogram(vv,range=(-window_sec,window_sec),bins=15)
        else:
            cnt,edg = np.histogram(vv,range=(0,2*window_sec),bins=15)
        plt.plot(edg[:-1],cnt/(len(raster[event][unit])*(edg[1]-edg[0])),color=color[brain_area])
        # plt.eventplot(raster[event][unit],color=color[brain_area[unit]],lw=1.5)
        plt.plot([0,0],[0,max(cnt/(len(raster[event][unit])*(edg[1]-edg[0])))],'--', color=(0.5,)*3, lw=2)
        if ccplot <= 36:
            plt.xticks([])
        else:
            plt.xlabel('time [sec]')
        ccplot+=1
plt.tight_layout()

# plt.savefig('%s_%d_mean_rate.png'%(session,pltcnt+1))


## session m53s50
# PPC_example = [136, 44, 99,17]
# PFC_example = [61,62]


## session m53s51
# PPC_example = []
# PFC_example = [15]

## session m72s2
# PPC_example = []
# PFC_example = [6]


## session m72s1
PPC_example = []
PFC_example = [2]

# plt.figure(figsize=(10,10))  

# brain_area = unit_info['brain_area']
color = {'MST':'g','PPC':'b','PFC':'r','VIP':'k'}
# plt.close('all')

for unit in (PPC_example + PFC_example):
    brain_area = exp_data.spikes.brain_area[unit-1]
    ccplot = 1
    plt.figure(figsize=(10,4))
    plt.suptitle(brain_area + ' example unit')
    for event in events:
        
        ax = plt.subplot(1,4,ccplot)
        plt.title(event)
        plt.eventplot(raster[event][unit-1],color=color[brain_area],lw=2)
        plt.plot([0,0],[0,len(raster[event][unit-1])],'--', color=(0.5,)*3, lw=2)
        if event == 't_flyOFF':
            plt.plot([0.3,0.3],[0,len(raster[event][unit-1])],'--', color=(0.5,)*3, lw=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ccplot!=1:
            plt.yticks([])
        else:
            plt.ylabel('trials')
            
        ccplot+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])



with open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_m72s2/fit_results_m72s2_c6_odd_1.0000.dill','rb') as fh:
    res = dill.load(fh)