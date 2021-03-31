#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:08:31 2021

@author: edoardo
"""
import numpy as np
import sys,os,dill
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
sys.path.append('/scratch/eb162/GAM_library/')
from spike_times_class import spike_counts
from behav_class import behavior_experiment,load_trial_types
from lfp_class import lfp_class
from copy import deepcopy
from scipy.io import loadmat,savemat
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PIL import ImageColor
from sklearn.decomposition import PCA

def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        sm_x[sel] = np.convolve(x[sel],filter,mode='same')
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk

def compute_median_IEI(X,varnames,event_list,trial_idx,dt=0.006,flyON_dur=50):
    """
    IEI = inter event interval
    flyON_dur = duration of target on in time points (50 tp correspond 50*dt sec, in 6ms bins it is 300ms)
    :return:
    """
    tr_with_event = {}
    tr_without_event = {}
    event_dict = {}
    for event in event_list:
        if event == 't_flyON':
            event_dict[event] = 't_flyOFF'
        else:
            event_dict[event] = event
        tr_with_event[event_dict[event]] = []
        tr_without_event[event_dict[event]] = []

        event_col = np.where(varnames == event_dict[event])[0][0]
        for tr in np.unique(trial_idx):
            if any(X[trial_idx == tr, event_col]):
                tr_with_event[event_dict[event]] += [tr]
            else:
                tr_without_event[event_dict[event]] += [tr]
        tr_with_event[event_dict[event]] = set(tr_with_event[event_dict[event]])

    median_iei = np.zeros(len(event_list) - 1)
    for k in range(len(event_list) - 1):
        ev1 = event_list[k]
        ev2 = event_list[k+1]
        if ev1 == 't_flyON':
            delta_tp = flyON_dur
            ev1 = 't_flyOFF'
        elif ev2 == 't_flyON':
            delta_tp = -flyON_dur
            ev2 = 't_flyOFF'
        else:
            delta_tp = 0
        ev1_col = np.where(varnames == ev1)[0][0]
        ev2_col = np.where(varnames == ev2)[0][0]
        trial_use = np.sort(list(tr_with_event[ev1].intersection(tr_with_event[ev2])))
        iei = np.zeros(trial_use.shape[0])
        cc = 0
        for tr in trial_use:
            iei[cc] = (np.where(X[trial_idx==tr, ev2_col])[0][0] - np.where(X[trial_idx==tr, ev1_col])[0][0] + delta_tp)*dt
            cc += 1


        median_iei[k] = np.median(iei)
    return median_iei,tr_without_event

def compute_aligned_rate(spk_mat, X, varnames, h, trial_idx, event_align,dt=0.006,flyON_dur=50,num_tp_x_bin=None,
                         post_tr_tp=50):
    # this code implies that reward is the last event
    assert(event_align[-1] == 't_reward')
    # get median iei
    median_iei, tr_without_event = compute_median_IEI(X,varnames,event_align,trial_idx,dt=dt,flyON_dur=flyON_dur)

    # set time points number for aligned trials
    num_tp = post_tr_tp
    if num_tp_x_bin is None:
        num_tp_x_bin = []
        time_bounds = np.cumsum(np.hstack(([0],median_iei)))
        for k in range(median_iei.shape[0]):
            t0 = time_bounds[k]
            t1 = time_bounds[k+1]
            num_tp_x_bin += [int(np.ceil((t1 - t0) / dt))]
            num_tp += num_tp_x_bin[-1] - 1


    else:
        time_bounds = np.cumsum(np.hstack(([0], median_iei)))
        for k in range(median_iei.shape[0]):
            num_tp += num_tp_x_bin[k] - 1
    num_tp_x_bin += [post_tr_tp]


    # extract firing rate estimate
    firing_rate_est = pop_spike_convolve(spk_mat, trial_idx, h)/dt

    unq_trials = np.unique(trial_idx)
    trial_rescaled_rate = np.zeros((spk_mat.shape[1],unq_trials.shape[0], num_tp)) * np.nan
    rescaled_vel = np.zeros((unq_trials.shape[0], num_tp)) * np.nan
    rescaled_ang = np.zeros((unq_trials.shape[0], num_tp)) * np.nan
    rescaled_ev = np.zeros((unq_trials.shape[0], num_tp)) * np.nan
    rescaled_eh = np.zeros((unq_trials.shape[0], num_tp)) * np.nan
    rescaled_rd = np.zeros((unq_trials.shape[0], num_tp)) * np.nan
    rescaled_ad = np.zeros((unq_trials.shape[0], num_tp)) * np.nan

    timeSTOP = np.zeros((unq_trials.shape[0])) * np.nan
    timeSTART = np.zeros((unq_trials.shape[0])) * np.nan
    timeOFF = np.zeros((unq_trials.shape[0])) * np.nan
    isREW = np.zeros((unq_trials.shape[0]),dtype=bool)
    rad_targ_vec = np.zeros(unq_trials.shape[0])
    cnt_tr = 0
    trial_list = []
    for tr in unq_trials:
        if cnt_tr % 100 == 0:
            print('%d/%d'%(cnt_tr+1,unq_trials.shape[0]))
        skip_trial = False
        is_rew = not (tr in tr_without_event['t_reward'])
        # tstart = np.where(X[trial_idx == tr, np.where(varnames == event_align[0])[0][0]] == 1)[0][0] * dt
        rad_vel = X[trial_idx == tr, np.where(varnames == 'rad_vel')[0][0]]
        ang_vel = X[trial_idx == tr, np.where(varnames == 'ang_vel')[0][0]]
        eye_vert = X[trial_idx == tr, np.where(varnames == 'eye_vert')[0][0]]
        eye_hori = X[trial_idx == tr, np.where(varnames == 'eye_hori')[0][0]]

        rad_targ = X[trial_idx == tr, np.where(varnames == 'rad_target')[0][0]]
        ang_targ = X[trial_idx == tr, np.where(varnames == 'ang_target')[0][0]]
        try:
            rad_targ_vec[cnt_tr] = rad_targ[~np.isnan(rad_targ)][0]
        except:
            rad_targ_vec[cnt_tr] = np.nan
        if tr == unq_trials[0]:
            time_rescale_all = []
        cc = 0
        trial_list += [tr]

        for k in range(median_iei.shape[0]):
            ev1 = event_align[k]
            ev2 = event_align[k+1]

            if ev1 == 't_flyON':
                delta_tp_0 = flyON_dur
                delta_tp_1 = 0
                ev1 = 't_flyOFF'
            elif ev2 == 't_flyON':
                delta_tp_1 = flyON_dur
                delta_tp_0 = 0
                ev2 = 't_flyOFF'
            else:
                delta_tp_0 = 0
                delta_tp_1 = 0

            time_points = num_tp_x_bin[k]
            try:
                tp_0 = np.where(X[trial_idx==tr, np.where(varnames==ev1)[0][0]] == 1)[0][0] - delta_tp_0
            except:
                print('skip tr',cnt_tr)
                rescaled_vel[cnt_tr, cc:cc + time_points - 1] = np.nan
                rescaled_ang[cnt_tr, cc:cc + time_points - 1] = np.nan
                rescaled_ev[cnt_tr, cc:cc + time_points - 1] = np.nan
                rescaled_eh[cnt_tr, cc:cc + time_points - 1] = np.nan
                rescaled_ad[cnt_tr, cc:cc + time_points - 1] = np.nan
                rescaled_rd[cnt_tr, cc:cc + time_points - 1] = np.nan
                cc += (time_points - 1)
                continue

            if ev2 == 't_reward' and (not is_rew):
                # this considers reward as last event
                sele_tr = np.where(trial_idx == tr)[0]
                sele_tr = sele_tr[tp_0:]
                tp_1 = tp_0 + min(int(median_iei[-1] / dt), sele_tr.shape[0])
            else:
                tp_1 = np.where(X[trial_idx==tr, np.where(varnames==ev2)[0][0]] == 1)[0][0] - delta_tp_1

            if tp_0 >= tp_1-1:
                skip_trial = True
                break

            time_rescale = np.linspace(time_bounds[k], time_bounds[k+1], time_points)[:-1]

            if tr == unq_trials[0]:
                time_rescale_all = np.hstack((time_rescale_all,time_rescale))


            rates = firing_rate_est[trial_idx == tr,:]
            rates = rates[tp_0: tp_1, :]
            rad_vel_tmp = rad_vel[tp_0:tp_1]
            ang_vel_tmp = ang_vel[tp_0:tp_1]
            eye_hori_tmp = eye_hori[tp_0:tp_1]
            eye_vert_tmp = eye_vert[tp_0:tp_1]
            rad_targ_tmp = rad_targ[tp_0:tp_1]
            ang_targ_tmp = ang_targ[tp_0:tp_1]

            time_tr = np.linspace(time_rescale[0], time_rescale[-1], rates.shape[0])

            interp_vel = interp1d(time_tr, rad_vel_tmp, kind='linear')
            interp_ang = interp1d(time_tr, ang_vel_tmp, kind='linear')
            interp_ev = interp1d(time_tr, eye_vert_tmp, kind='linear')
            interp_eh = interp1d(time_tr, eye_hori_tmp, kind='linear')
            interp_ad = interp1d(time_tr, ang_targ_tmp, kind='linear')
            interp_rd = interp1d(time_tr, rad_targ_tmp, kind='linear')


            rescaled_vel[cnt_tr,cc:cc+time_points-1] = interp_vel(time_rescale)
            rescaled_ang[cnt_tr,cc:cc+time_points-1] = interp_ang(time_rescale)
            rescaled_ev[cnt_tr,cc:cc+time_points-1] = interp_ev(time_rescale)
            rescaled_eh[cnt_tr,cc:cc+time_points-1] = interp_eh(time_rescale)
            rescaled_ad[cnt_tr,cc:cc+time_points-1] = interp_ad(time_rescale)
            rescaled_rd[cnt_tr,cc:cc+time_points-1] = interp_rd(time_rescale)



            for neu in range(rates.shape[1]):
                interp = interp1d(time_tr, rates[:,neu], kind='linear')
                trial_rescaled_rate[neu, cnt_tr, cc:cc+time_points-1] = interp(time_rescale)

            cc += (time_points-1)

        # set the post trial duration
        if skip_trial:
            cnt_tr += 1
            continue
        rad_vel_tmp = rad_vel[tp_1:tp_1 + post_tr_tp]
        ang_vel_tmp = ang_vel[tp_1:tp_1 + post_tr_tp]
        eye_hori_tmp = eye_hori[tp_1:tp_1 + post_tr_tp]
        eye_vert_tmp = eye_vert[tp_1:tp_1 + post_tr_tp]
        rad_targ_tmp = rad_targ[tp_1:tp_1 + post_tr_tp]
        ang_targ_tmp = ang_targ[tp_1:tp_1 + post_tr_tp]

        rates = firing_rate_est[trial_idx == tr, :]
        rates = rates[tp_1: tp_1 + post_tr_tp, :]
        time_tr = np.linspace(time_rescale[-1], time_rescale[-1]+post_tr_tp*dt, post_tr_tp+1)[1:]
        time_rescale = time_tr

        if tr == unq_trials[0]:
            time_rescale_all = np.hstack((time_rescale_all, time_rescale))

        if is_rew and rates.shape[0] != post_tr_tp:
            raise ValueError

        elif rates.shape[0] != post_tr_tp:
            trial_rescaled_rate[neu, cnt_tr, cc:] = np.nan
            rescaled_vel[cnt_tr, cc:] = np.nan
            true_size = rates.shape[0]
        else:
            true_size = post_tr_tp

        if true_size == 0:
            cnt_tr += 1
            continue
        time_tr = time_tr[:true_size]
        time_rescale = time_tr
        try:
            pass
        except:
            xxx=1
        
        if rad_vel_tmp.shape[0] == 1:
            # interp_vel = interp1d(time_tr, rad_vel_tmp, kind='linear')
            # interp_ang = interp1d(time_tr, ang_vel_tmp, kind='linear')
            # interp_ev = interp1d(time_tr, eye_vert_tmp, kind='linear')
            # interp_eh = interp1d(time_tr, eye_hori_tmp, kind='linear')
            # interp_ad = interp1d(time_tr, ang_targ_tmp, kind='linear')
            # interp_rd = interp1d(time_tr, rad_targ_tmp, kind='linear')
        
    
            rescaled_vel[cnt_tr, cc:cc + true_size] = rad_vel_tmp[0]
            rescaled_ang[cnt_tr, cc:cc + true_size] = ang_vel_tmp[0]
            rescaled_ev[cnt_tr, cc:cc + true_size] = eye_vert_tmp[0]
            rescaled_eh[cnt_tr, cc:cc + true_size] = eye_hori_tmp[0]
            rescaled_ad[cnt_tr, cc:cc + true_size] = ang_targ_tmp[0]
            rescaled_rd[cnt_tr, cc:cc + true_size] = rad_targ_tmp[0]
            for neu in range(rates.shape[1]):
                # interp = interp1d(time_tr, rates[:, neu], kind='linear')
                trial_rescaled_rate[neu, cnt_tr, cc:cc+true_size] = rates[0, neu]

        else:
            interp_vel = interp1d(time_tr, rad_vel_tmp, kind='linear')
            interp_ang = interp1d(time_tr, ang_vel_tmp, kind='linear')
            interp_ev = interp1d(time_tr, eye_vert_tmp, kind='linear')
            interp_eh = interp1d(time_tr, eye_hori_tmp, kind='linear')
            interp_ad = interp1d(time_tr, ang_targ_tmp, kind='linear')
            interp_rd = interp1d(time_tr, rad_targ_tmp, kind='linear')
        
    
            rescaled_vel[cnt_tr, cc:cc + true_size] = interp_vel(time_rescale)
            rescaled_ang[cnt_tr, cc:cc + true_size] = interp_ang(time_rescale)
            rescaled_ev[cnt_tr, cc:cc + true_size] = interp_ev(time_rescale)
            rescaled_eh[cnt_tr, cc:cc + true_size] = interp_eh(time_rescale)
            rescaled_ad[cnt_tr, cc:cc + true_size] = interp_ad(time_rescale)
            rescaled_rd[cnt_tr, cc:cc + true_size] = interp_rd(time_rescale)
    
    
            for neu in range(rates.shape[1]):
                interp = interp1d(time_tr, rates[:, neu], kind='linear')
                trial_rescaled_rate[neu, cnt_tr, cc:cc+true_size] = interp(time_rescale)

        timeOFF[cnt_tr] = np.where(X[trial_idx == tr, np.where(varnames == 't_flyOFF')[0][0]] == 1)[0][0] * dt
        timeSTART[cnt_tr] = np.where(X[trial_idx == tr, np.where(varnames == 't_move')[0][0]] == 1)[0][0] * dt
        timeSTOP[cnt_tr] = np.where(X[trial_idx == tr, np.where(varnames == 't_stop')[0][0]] == 1)[0][0] * dt
        isREW[cnt_tr] = is_rew
        cnt_tr += 1

    return (np.array(time_rescale_all), time_bounds, trial_rescaled_rate,num_tp_x_bin,
            np.array(trial_list),rad_targ_vec,rescaled_vel,timeOFF,timeSTART,timeSTOP,isREW,rescaled_ang,rescaled_ev,rescaled_eh,rescaled_ad,rescaled_rd)


# solo condizioni (density, PTB, controlgain)
# ciclo nella cartella e filtro le sezioni

filtwidth = 10
# num_timept_x_bin = [50, 102, 102, 74, 50]
num_timept_x_bin=[50, 205, 74, 50]
# num_timept_x_bin=[ 205, 74, 50]
# event_align = ['t_move','t_stop','t_reward']
event_align = ['t_flyON','t_flyOFF','t_stop','t_reward']
# event_align = ['t_flyON','t_flyOFF', 't_ptb', 't_stop', 't_reward']
file_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'

done_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/multi_event_aligned'
# mi_path = '/Users/edoardo/Work/Code/Angelaki-Savin/Analysis_Scripts/mutual_info/mutual_info_results.npy'
# session = 'm53s91'

flag_first = True

skipUntil = True

# extract sessions
sess_list = []
for root, dirs, files in os.walk('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc'):
    for fh_name in files:
        for cond in ['controlgain','ptb','density']:
            if cond in fh_name:
                sess_list += [fh_name.split('results_')[1].split('_')[0]]


sess_list = np.unique(sess_list)

fl = True
for session in sess_list:
    if session != 'm53s40' and fl:
        continue
    
    # else:
    #     fl=False
    # if not (session.startswith('m') and session.endswith('.npz') and (not 'eyeTrack' in session)):
    #     continue
    # session = session.split('.')[0]


    # if session != 'm53s130' and skipUntil:
    #     continue
    # else:
    #     skipUntil = False
    # if session == 'm53s130':
    #     continue
    # if session == 'm53s110':
    #     break
    # fix param
    # if not session.startswith('m91'):
    #     continue
    print('extracting ', session)
    firefly_ON = 0.3 # duration of fly on
    # gaussian filter kernel generation
    t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
    h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
    h = h / np.sum(h)


    list_done = os.listdir(done_path)


    print('COMPUTING SESSION %s\n\n'%session,event_align[0])
    try:
        dat = np.load(os.path.join(file_fld,session+'.npz'),allow_pickle=True)
    except:
        print('could not open %s'%session)
        continue

    concat = dat['data_concat'].all()
    yt = concat['Yt']
    X = concat['Xt']
    trial_idx = concat['trial_idx']
    var_names = dat['var_names']
    trial_type = dat['info_trial'].all().trial_type
    # compute_median_IEI(concat['Xt'],dat['var_names'],event_align,trial_idx)
    # firing_rate_est = pop_spike_convolve(yt,trial_idx,h)
    if flag_first:
        (time_rescale_all,time_bounds, trial_rescaled_rate,num_timept_x_bin,trial_list,
         rad_targ_vec,rescaled_vel,timeOFF,timeSTART,timeSTOP,isREW,rescaled_ang,rescaled_ev,rescaled_eh,rescaled_ad,rescaled_rd) = compute_aligned_rate(yt, X, var_names, h, trial_idx, event_align,dt=0.006,flyON_dur=50,num_tp_x_bin=num_timept_x_bin)
        flag_first = False
    else:
        (time_rescale_all,time_bounds, trial_rescaled_rate,tmp,trial_list,
         rad_targ_vec,rescaled_vel,timeOFF,timeSTART,timeSTOP,isREW,rescaled_ang,rescaled_ev,rescaled_eh,rescaled_ad,rescaled_rd) = compute_aligned_rate(yt, X, var_names, h, trial_idx, event_align,dt=0.006,flyON_dur=50,num_tp_x_bin=num_timept_x_bin)

    if event_align[0] != 't_flyON':
        np.savez('/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/multi_event_aligned/%s_multiresc_trials.npz'%session,event_align=event_align,
            time_rescale=time_rescale_all,time_bounds=time_bounds,rescaled_rate=trial_rescaled_rate,trial_list=trial_list,
            structure='neuron x trial x time point',trial_rad_targ=rad_targ_vec,rescaled_vel=rescaled_vel,
                 timeSTART=timeSTART, timeOFF=timeOFF, timeSTOP=timeSTOP,isREW=isREW,resc_ang_vel=rescaled_ang,
                 resc_eye_vert=rescaled_ev,resc_eye_hori=rescaled_eh,resc_ang_targ=rescaled_ad,resc_rad_targ=rescaled_rd,trial_type=trial_type)
    else:
        np.savez(
            '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/multi_event_aligned/flyON_%s_multiresc_trials.npz' % session,
            event_align=event_align,
            time_rescale=time_rescale_all, time_bounds=time_bounds, rescaled_rate=trial_rescaled_rate,
            trial_list=trial_list,
            structure='neuron x trial x time point', trial_rad_targ=rad_targ_vec, rescaled_vel=rescaled_vel,
            timeMOVE=timeSTART,timeOFF=timeOFF,timeSTOP=timeSTOP,isREW=isREW,resc_ang_vel=rescaled_ang, resc_eye_vert=rescaled_ev,resc_eye_hori=rescaled_eh,resc_ang_targ=rescaled_ad,resc_rad_targ=rescaled_rd,
            trial_type=trial_type)

