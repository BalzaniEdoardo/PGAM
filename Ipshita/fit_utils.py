#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:02:18 2021

@author: edoardo
"""
from scipy.io import loadmat
import numpy as np
import sys
sys.path.append('../GAM_library')
sys.path.append('../firefly_utils')
from GAM_library import *
from data_handler import *
import scipy.stats as sts
from copy import deepcopy




def construct_knots_highSpatialDensity(dat, varType, varName, neuNum=0, portNum=0, history_filt_len=199):
    # Standard params for the B-splines
    is_cyclic = False  # no angles or period variables
    kernel_len = 165  # this is used for event variables
    order = 4  # cubic spline
    penalty_type = 'der'  # derivative based penalization
    der = 2  # degrees of the derivative

    is_temporal_kernel = (varType == 'eventVar') | (varType == 'logVar')

    if varName == 'spike_hist':
        kernel_direction = 1  # Causal filter
    else:
        kernel_direction = 0  # acausal filter

    # get the variable
    if varName == 'spike_hist':
        x = dat['spkMat'][neuNum, :]
    else:
        x = np.squeeze(dat[varType][varName][0, 0])

    if varName == 'licks':
        x = x[portNum]

    if (is_temporal_kernel) & (varName != 'spike_hist'):
        knots = np.linspace(-kernel_len, kernel_len, 6)
        knots = np.hstack(([knots[0]] * 3,
                           knots,
                           [knots[-1]] * 3
                           ))

    elif varName == 'spike_hist':
        if history_filt_len > 20:
            kernel_len = history_filt_len
            knots = np.hstack(([(10) ** -6] * 3, np.linspace((10) ** -6, kernel_len // 2, 10), [kernel_len // 2] * 3))
            penalty_type = 'der'
            der = 2
            is_temporal_kernel = True
        else:
            knots = np.linspace((10) ** -6, kernel_len // 2, 6)
            penalty_type = 'EqSpaced'
            order = 1  # too few time points for a cubic splines

    elif varName == 'x':
        knots = np.linspace(0.85, 4.2, 5)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 1) | (x > 4.2)] = np.nan

    elif varName == 'y':
        knots = np.linspace(2, 110, 25)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 2) | (x > 110)] = np.nan


    elif varName == 'vel':

        knots = np.hstack((np.linspace(0, 30, 11)[:-1], np.linspace(30, 55, 6)))
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 0) | (x > 55)] = np.nan


    elif varName == 'freq':
        knots = np.hstack((np.linspace(2, 60, 25)))
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 2) | (x > 60)] = np.nan

    return knots, x, is_cyclic, order, \
           kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der
           


def construct_knots_medSpatialDensity(dat, varType, varName, neuNum=0, portNum=0, history_filt_len=199):
    # Standard params for the B-splines
    is_cyclic = False  # no angles or period variables
    kernel_len = 165  # this is used for event variables
    order = 4  # cubic spline
    penalty_type = 'der'  # derivative based penalization
    der = 2  # degrees of the derivative

    is_temporal_kernel = (varType == 'eventVar') | (varType == 'logVar')

    if varName == 'spike_hist':
        kernel_direction = 1  # Causal filter
    else:
        kernel_direction = 0  # acausal filter

    # get the variable
    if varName == 'spike_hist':
        x = dat['spkMat'][neuNum, :]
    else:
        x = np.squeeze(dat[varType][varName][0, 0])

    if varName == 'licks':
        x = x[portNum]

    if (is_temporal_kernel) & (varName != 'spike_hist'):
        knots = np.linspace(-kernel_len, kernel_len, 6)
        knots = np.hstack(([knots[0]] * 3,
                           knots,
                           [knots[-1]] * 3
                           ))

    elif varName == 'spike_hist':
        if history_filt_len > 20:
            kernel_len = history_filt_len
            knots = np.hstack(([(10) ** -6] * 3, np.linspace((10) ** -6, kernel_len // 2, 10), [kernel_len // 2] * 3))
            penalty_type = 'der'
            der = 2
            is_temporal_kernel = True
        else:
            knots = np.linspace((10) ** -6, kernel_len // 2, 6)
            penalty_type = 'EqSpaced'
            order = 1  # too few time points for a cubic splines

    elif varName == 'x':
        knots = np.linspace(0.85, 4.2, 5)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 1) | (x > 4.2)] = np.nan

    elif varName == 'y':
        knots = np.linspace(2, 110, 15)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 2) | (x > 110)] = np.nan


    elif varName == 'vel':

        knots = np.hstack((np.linspace(0, 10, 4)[:-1], np.linspace(10, 55, 6)))
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 0) | (x > 55)] = np.nan


    elif varName == 'freq':
        knots = np.hstack((np.linspace(2, 60, 15)))
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 2) | (x > 60)] = np.nan

    return knots, x, is_cyclic, order, \
           kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der
           

def construct_knots_lowSpatialDensity(dat, varType, varName, neuNum=0, portNum=0, history_filt_len=199):
    # Standard params for the B-splines
    is_cyclic = False  # no angles or period variables
    kernel_len = 165  # this is used for event variables
    order = 4  # cubic spline
    penalty_type = 'der'  # derivative based penalization
    der = 2  # degrees of the derivative

    is_temporal_kernel = (varType == 'eventVar') | (varType == 'logVar')

    if varName == 'spike_hist':
        kernel_direction = 1  # Causal filter
    else:
        kernel_direction = 0  # acausal filter

    # get the variable
    if varName == 'spike_hist':
        x = dat['spkMat'][neuNum, :]
    else:
        x = np.squeeze(dat[varType][varName][0, 0])

    if varName == 'licks':
        x = x[portNum]

    if (is_temporal_kernel) & (varName != 'spike_hist'):
        knots = np.linspace(-kernel_len, kernel_len, 6)
        knots = np.hstack(([knots[0]] * 3,
                           knots,
                           [knots[-1]] * 3
                           ))

    elif varName == 'spike_hist':
        if history_filt_len > 20:
            kernel_len = history_filt_len
            knots = np.hstack(([(10) ** -6] * 3, np.linspace((10) ** -6, kernel_len // 2, 10), [kernel_len // 2] * 3))
            penalty_type = 'der'
            der = 2
            is_temporal_kernel = True
        else:
            knots = np.linspace((10) ** -6, kernel_len // 2, 6)
            penalty_type = 'EqSpaced'
            order = 1  # too few time points for a cubic splines

    elif varName == 'x':
        knots = np.linspace(0.85, 4.2, 5)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 1) | (x > 4.2)] = np.nan

    elif varName == 'y':
        knots = np.linspace(2, 110, 10)
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 2) | (x > 110)] = np.nan


    elif varName == 'vel':

        knots = np.hstack((np.linspace(0, 10, 4)[:-1], np.linspace(10, 55, 6)))
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 0) | (x > 55)] = np.nan


    elif varName == 'freq':
        knots = np.hstack((np.linspace(2, 60, 10)))
        knots = np.hstack((
            [knots[0]] * 3,
            knots,
            [knots[-1]] * 3
        ))
        x[(x < 2) | (x > 60)] = np.nan

    return knots, x, is_cyclic, order, \
           kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der
           
def pseudo_r2_comp(spk, fit, sm_handler, family, use_tp=None):
    exog, _ = sm_handler.get_exog_mat(fit.var_list)
    if use_tp is None:
        use_tp = np.ones(exog.shape[0],dtype=bool)
    
    exog = exog[use_tp]
    spk = spk[use_tp]
    lin_pred = np.dot(exog, fit.beta)
    mu = fit.family.fitted(lin_pred)
    res_dev_t = fit.family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t ** 2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t ** 2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2

def compute_tuning(spk, fit, exog, var, sm_handler, filter_trials,dt=0.006):
    
    mu = np.dot(exog[filter_trials], fit.beta)
    sigma2 = np.einsum('ij,jk,ik->i', exog[filter_trials], fit.cov_beta, exog[filter_trials],
                       optimize=True)

    # convert to rate space
    lam_s = np.exp(mu + sigma2 * 0.5)
    sigm2_s = (np.exp(sigma2) - 1) * np.exp(2 * mu + sigma2)
    lam_s = lam_s
    sigm2_s = sigm2_s

    if fit.smooth_info[var]['is_temporal_kernel'] and fit.smooth_info[var]['is_event_input']:
    
        
        reward = np.squeeze(sm_handler[var]._x)[filter_trials]
        # set everything to -1
        time_kernel = np.ones(reward.shape[0]) * np.inf
        rew_idx = np.where(reward == 1)[0]
        
    
        # temp kernel where 161 timepoints long
        size_kern = fit.smooth_info[var]['time_pt_for_kernel'].shape[0]
        if size_kern %2 == 0:
            size_kern += 1
        half_size = (size_kern - 1) // 2
        timept = np.arange(-half_size,half_size+1) * fit.time_bin
    
        temp_bins = np.linspace(timept[0], timept[-1], 15)
        dt = temp_bins[1] - temp_bins[0]
    
        tuning = np.zeros(temp_bins.shape[0])
        var_tuning = np.zeros(temp_bins.shape[0])
        sc_based_tuning = np.zeros(temp_bins.shape[0])
        tot_s_vec = np.zeros(temp_bins.shape[0])
        x_axis = deepcopy(temp_bins)
    
        for ind in rew_idx:
            if (ind < half_size) or (ind >= time_kernel.shape[0] - half_size):
                continue
            time_kernel[ind - half_size:ind + half_size+1] = timept
    
        cc = 0
        for t0 in temp_bins:
            idx = (time_kernel >= t0) * (time_kernel < t0 + dt)
            tuning[cc] = np.mean(lam_s[idx])
            var_tuning[cc] = np.nanpercentile(sigm2_s[idx], 90)
            sc_based_tuning[cc] = spk[filter_trials][idx].mean()
            tot_s_vec[cc] = np.sum(idx)

            cc += 1
    else:
        # this gives error for 2d variable
        vels = np.squeeze(sm_handler[var]._x)[filter_trials]
        if len(vels.shape) > 1:
            print('Mutual info not implemented for multidim variable')
            raise ValueError
            
        knots = fit.smooth_info[var]['knots'][0]
        vel_bins = np.linspace(knots[0], knots[-2], 16)
        dv = vel_bins[1] - vel_bins[0]
    
        tuning = np.zeros(vel_bins.shape[0]-1)
        var_tuning = np.zeros(vel_bins.shape[0]-1)
        sc_based_tuning = np.zeros(vel_bins.shape[0]-1)
        tot_s_vec = np.zeros(vel_bins.shape[0]-1)
        x_axis = 0.5*(vel_bins[:-1]+vel_bins[1:])
    
        cc = 0
    
        for v0 in vel_bins[:-1]:
    
            idx = (vels >= v0) * (vels < v0 + dv)
            tuning[cc] = np.nanmean(lam_s[idx])
            var_tuning[cc] = np.nanpercentile(sigm2_s[idx], 90)
            sc_based_tuning[cc] = spk[filter_trials][idx].mean()
            tot_s_vec[cc] = np.sum(idx)

            cc += 1
    return x_axis, tuning/dt, sc_based_tuning/dt

        
def partition_trials(tstart, tend):
    T = tstart.shape[0]
    trial_idx = np.zeros(T)
    idx_start = np.where(tstart)[0]
    idx_end = np.where(tend)[0]

    if idx_start[0] > idx_end[0]:
        idx_end = idx_end[1:]
    if idx_start[-1] > idx_end[-1]:
        idx_start = idx_start[:-1]

    curId = 0
    for k in range(0, idx_start.shape[0]):
        
        if k+1 < idx_start.shape[0]:
            trEnd = int((idx_end[k] + idx_start[k+1]) * 0.5)
        else:
            trEnd = tstart.shape[0]
        if k > 0:
            trStart = int((idx_end[k-1] + idx_start[k]) * 0.5)
        else:
            trStart = 0
        # print(trStart,trEnd)
        trial_idx[trStart:trEnd] = curId
        curId += 1
        
    return trial_idx