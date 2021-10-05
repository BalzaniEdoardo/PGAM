#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:04:09 2021

@author: edoardo
"""

import sys,os
import numpy as np
import matplotlib.pylab as plt
import re
if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc')

    sess_list = ['m53s113']
    JOB = 0
    clust = False
    area = 'MST'
else:
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')

    sess_list = []
    for fhn in os.listdir('/scratch/jpn5/dataset_firefly'):
        if re.match('^m\d+s\d+.npz$',fhn):
            sess_list += [fhn.split('.')[0]]
    clust = True
    area = sys.argv[2]
    JOB = int(sys.argv[1]) - 1

from GAM_library import *
import dill
from utils_loading import unpack_preproc_data
from knots_constructor import *




session = sess_list[JOB]

cond_type = 'all'
cond_value = True
if not clust:
    fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
else:
    fhName = '/scratch/jpn5/dataset_firefly/%s.npz' % session



par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']

(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
  trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
  cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
  channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)


# # par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
# #         'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
# #         'unit_type','channel_id','electrode_id','cluster_id']

# # (Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
# #   trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
# #   cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
# #   channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)

# # pair trials

# base_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'
# print('loading session %s...' % session)
# pre_trial_dur = 0.0
# post_trial_dur = 0.0


# # keys in the mat file generated by the preprocessing script of  K.
# behav_stat_key = 'behv_stats'
# spike_key = 'units'
# behav_dat_key = 'trials_behv'
# lfp_key = 'lfps'

# dat = loadmat(os.path.join(base_file, '%s.mat' % (session)))


# exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur,
#                         post_trial_dur=post_trial_dur,
#                         lfp_beta=None, lfp_alpha=None,
#                         lfp_theta=None, extract_lfp_phase=False,
#                         use_eye='right',  extract_fly_and_monkey_xy=True)

# exp_data.set_filters('all', True)
# # impose all replay trials
# exp_data.filter = exp_data.filter + exp_data.info.get_replay(0, skip_not_ok=False)

# t_targ = dict_to_vec(exp_data.behav.events.t_targ)
# t_move = dict_to_vec(exp_data.behav.events.t_move)

# t_start = np.min(np.vstack((t_move, t_targ)), axis=0) - pre_trial_dur
# t_stop = dict_to_vec(exp_data.behav.events.t_end) + post_trial_dur

# # bin_ts = time_stamps_rebin(exp_data.behav.time_stamps, binwidth_ms=20)
# exp_data.spikes.bin_spikes(exp_data.behav.time_stamps, t_start=t_start, t_stop=t_stop, select=exp_data.filter)

# var_names = ('rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target',
#              'eye_vert', 'eye_hori',
#              'rad_acc', 'ang_acc')

# time_pts, rate, sm_traj, raw_traj, fly_pos, cov_dict = exp_data.GPFA_YU_preprocessing_noTW(t_start, t_stop,
#                                                                                            var_list=var_names,binwidth_ms=None)

# spikes, var_dict, trial_idx = exp_data.concatenate_inputs('t_move','t_stop','t_reward',t_start=t_start,t_stop=t_stop)


# brain_area = exp_data.spikes.brain_area


# cLeft = np.array([-130,200])
# cCenter = np.array([0,200])
# cRight = np.array([130,200])

# radius = 40
# radiius_center = 25
# end_pt = np.zeros(sm_traj.shape)
# for k in range(sm_traj.shape[0]):
#     last_nonNan = np.where(~np.isnan(sm_traj[k, 0]))[0][-1]
#     end_pt[k, 0] = sm_traj[k, 0][last_nonNan]
#     end_pt[k, 1] = sm_traj[k, 1][last_nonNan]


# selLeft = np.linalg.norm(end_pt - cLeft,axis=1) < radius
# selCenter = np.linalg.norm(end_pt - cCenter,axis=1) < radiius_center
# selRight = np.linalg.norm(end_pt - cRight,axis=1) < radius
# print(selLeft.sum(), selCenter.sum(), selRight.sum())

# plt.figure()
# for kk in np.where(selLeft)[0]:
#     plt.plot(sm_traj[kk,0],sm_traj[kk,1],'b')

# for kk in np.where(selCenter)[0]:
#     plt.plot(sm_traj[kk,0],sm_traj[kk,1],'r')

# for kk in np.where(selRight)[0]:
#     plt.plot(sm_traj[kk,0],sm_traj[kk,1],'g')

# plt.savefig('separate_traj.png')


# rateLeft = np.zeros((0,81))
# for key in np.where(selLeft)[0]:
#     rateLeft = np.vstack((rateLeft,rate[key].T))

# rateCenter = np.zeros((0,81))
# for key in np.where(selRight)[0]:
#     rateCenter = np.vstack((rateCenter,rate[key].T))

# rateRight = np.zeros((0,81))
# for key in np.where(selCenter)[0]:
#     rateRight = np.vstack((rateRight,rate[key].T))

