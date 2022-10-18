#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:33:36 2022

@author: edoardo
"""
import numpy as np
from GAM_library import *
from copy import deepcopy
import scipy.stats as sts
from scipy.integrate import simps
from numba import njit

def pseudo_r2_comp(spk, fit, sm_handler, family, use_tp=None, exog=None):
    if exog is None:
        exog, _ = sm_handler.get_exog_mat_fast(fit.var_list)
    exog_cut = deepcopy(exog)
    if use_tp is None:
        use_tp = np.ones(exog.shape[0], dtype=bool)

    exog_cut = exog_cut[use_tp]
    spk = spk[use_tp]
    lin_pred = np.dot(exog_cut, fit.beta)
    mu = fit.family.fitted(lin_pred)
    res_dev_t = fit.family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t ** 2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t ** 2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2, exog


def compute_tuning(spk, fit, exog, var, sm_handler, filter_trials, trial_idx, dt=0.006,bins=15):
    # print('comp tun')
    mu = np.dot(exog[filter_trials], fit.beta)
    sigma2 = np.einsum('ij,jk,ik->i', exog[filter_trials], fit.cov_beta, exog[filter_trials],
                       optimize=True)
    # convert to rate space
    lam_s = np.exp(mu + sigma2 * 0.5)
    if fit.smooth_info[var]['is_temporal_kernel'] and fit.smooth_info[var]['is_event_input']:
        # print('temp')
        x_axis, tuning, sc_based_tuning, counts = compute_tuning_temporal(var, lam_s, sm_handler, fit, spk, filter_trials, trial_idx, bins, dt)
    else:
        # print('spat')
        x_axis, tuning, sc_based_tuning,counts = compute_tuning_spatial(var, lam_s, sm_handler, fit, spk, filter_trials, bins,dt)
    return x_axis, tuning, sc_based_tuning, counts

def mutual_info_est(spk_counts, exog, fit, var, sm_handler, filter_trials, trial_idx, dt=0.006,bins=15):
    # print('here')
    temp_bins, tuning, sc_based_tuning, count_bins = compute_tuning(spk_counts, fit, exog, var, sm_handler, filter_trials,trial_idx, dt=0.006, bins=bins)
    tuning = tuning * dt
    smooth_info = fit.smooth_info
    entropy_s = np.zeros(temp_bins.shape[0]) * np.nan
    for cc in range(tuning.shape[0]):
        entropy_s[cc] = sts.poisson.entropy(tuning[cc])

    if (smooth_info[var]['kernel_direction'] == 1) and\
            (smooth_info[var]['is_temporal_kernel']) and (smooth_info[var]['is_event_input']):

        sel = temp_bins > 0
        temp_bins = temp_bins[sel]
        count_bins = count_bins[sel]
        tuning = tuning[sel]
        sc_based_tuning = sc_based_tuning[sel]
        entropy_s = entropy_s[sel]

    elif (smooth_info[var]['kernel_direction'] == -1) and\
            (smooth_info[var]['is_temporal_kernel']) and (smooth_info[var]['is_event_input']):
        sel = temp_bins < 0
        temp_bins = temp_bins[sel]
        count_bins = count_bins[sel]
        tuning = tuning[sel]
        sc_based_tuning = sc_based_tuning[sel]
        entropy_s = entropy_s[sel]

    prob_s = count_bins / count_bins.sum()
    mean_lam = np.sum(prob_s * tuning)

    try:
        mi = (sts.poisson.entropy(mean_lam) - (prob_s * entropy_s).sum()) * np.log2(np.exp(1)) / dt
    except:
        mi = np.nan

    tmp_val = empty_container()
    tmp_val.x = temp_bins
    tmp_val.y_raw = sc_based_tuning
    tmp_val.y_model = tuning / dt

    return mi, tmp_val


def alignRateForMI(y, lam_s, var, sm_handler, smooth_info, time_bin, filter_trials, trial_idx):
    """
    Slow aligment method.
    """
    reward = np.squeeze(sm_handler[var]._x)[filter_trials]
    # temp kernel where 161 timepoints long
    size_kern = smooth_info[var]['time_pt_for_kernel'].shape[0]
    if size_kern % 2 == 0:
        size_kern += 1
    half_size = (size_kern - 1) // 2
    timept = np.arange(-half_size, half_size + 1) * time_bin
    if (var.startswith('neu')) or var == 'spike_hist':
        nbin = timept.shape[0]
        if nbin % 2 == 0:
            nbin += 1
    else:
        nbin = 15
    temp_bins = np.linspace(timept[0], timept[-1], nbin)
    # sum spikes
    tuning = np.zeros(temp_bins.shape[0])
    count_bins = np.zeros(temp_bins.shape[0])
    sc_based_tuning = np.zeros(temp_bins.shape[0])
    for tr in np.unique(trial_idx):
        select = trial_idx == tr
        rwd_tr = reward[select]
        lam_s_tr = lam_s[select]
        y_tr = y[select]
        for ii in np.where(rwd_tr==1)[0]:
            i0 = max(0, ii-half_size)
            i1 = min(len(rwd_tr), ii+half_size+1)
            d0 = ii - i0
            d1 = i1 - ii
            tmpmu = lam_s_tr[i0:i1]
            tmpy = y_tr[i0:i1]
            iidx = np.array(np.round(nbin//2 + (-d0 + np.arange(0,d1+d0))*time_bin / (temp_bins[1]-temp_bins[0])),dtype=int)
            for cc in np.unique(iidx):
                tuning[cc] = tuning[cc] + tmpmu[iidx == cc].sum()
                count_bins[cc] = count_bins[cc] + (iidx == cc).sum()
                sc_based_tuning[cc] = sc_based_tuning[cc] + tmpy[iidx==cc].sum()


    tuning = tuning / count_bins
    sc_based_tuning = sc_based_tuning / count_bins

    entropy_s = np.zeros(temp_bins.shape[0])*np.nan
    for cc in range(tuning.shape[0]):
        entropy_s[cc] = sts.poisson.entropy(tuning[cc])

    if (var.startswith('neu')) or var == 'spike_hist':
        sel = temp_bins > 0
        temp_bins = temp_bins[sel]
        count_bins = count_bins[sel]
        tuning = tuning[sel]
        sc_based_tuning = sc_based_tuning[sel]
        entropy_s = entropy_s[sel]
    prob_s = count_bins / count_bins.sum()
    mean_lam = np.sum(prob_s * tuning)

    try:
        mi = (sts.poisson.entropy(mean_lam) - (prob_s * entropy_s).sum()) * np.log2(np.exp(1)) / time_bin
    except:
        mi = np.nan

    tmp_val = empty_container()
    tmp_val.x = temp_bins
    tmp_val.y_raw = sc_based_tuning / time_bin
    tmp_val.y_model = tuning / time_bin

    return mi, tmp_val


def find_first_x_bin(result, num_events, bins):
    flag = False
    skip_until = 0#nb.int64([0])
    for idx, val in np.ndenumerate(num_events):
        # print(idx,type(idx))
        ii = idx[0]#nb.int64(idx)
        if ii < skip_until:
            continue
        if (val != 0):
            result[idx] = 1
            skip_until = ii + bins
            flag = True
    return result,flag

def compute_tuning_spatial(var, lam_s, sm_handler, fit, spk, filter_trials, bins,dt):
    # this gives error for 2d variable
    vels = np.squeeze(sm_handler[var]._x)[filter_trials]
    if len(vels.shape) > 1:
        print('Mutual info not implemented for multidim variable')
        raise ValueError

    knots = fit.smooth_info[var]['knots'][0]
    vel_bins = np.linspace(knots[0], knots[-2], bins + 1)
    dv = vel_bins[1] - vel_bins[0]

    tuning = np.zeros(vel_bins.shape[0] - 1)
    sc_based_tuning = np.zeros(vel_bins.shape[0] - 1)
    tot_s_vec = np.zeros(vel_bins.shape[0] - 1)
    x_axis = 0.5 * (vel_bins[:-1] + vel_bins[1:])
    ii = 0

    for v0 in vel_bins[:-1]:
        idx = (vels >= v0) * (vels < v0 + dv)
        tuning[ii] = np.nanmean(lam_s[idx])
        sc_based_tuning[ii] = spk[filter_trials][idx].mean()
        tot_s_vec[ii] = np.sum(idx)

        ii += 1
    return x_axis, tuning / dt, sc_based_tuning / dt, tot_s_vec

def compute_tuning_temporal(var, lam_s, sm_handler, fit, spk, filter_trials,trial_idx, bins, dt):
    
    events = np.array(np.squeeze(sm_handler[var]._x)[filter_trials], dtype=np.int64)
    events_analyze = np.zeros(events.shape, dtype=np.int64)
    filter_len = np.int64(fit.smooth_info[var]['time_pt_for_kernel'].shape[0])
    flag = True
    cc = 0

    ## params for extraction
    if filter_len % 2 == 0:
        filter_len += 1
    half_size = (filter_len - 1) // 2
    timept = np.arange(-half_size, half_size + 1) * fit.time_bin

    if bins % 2 == 0:
        bins += 1
    temp_bins = np.linspace(timept[0], timept[-1], bins)
    #dt = temp_bins[1] - temp_bins[0]
    tuning = np.zeros(temp_bins.shape[0])
    sc_based_tuning = np.zeros(temp_bins.shape[0])
    tot_s_vec = np.zeros(temp_bins.shape[0])

    while flag:
        events_analyze, flag = find_first_x_bin(events_analyze, events, filter_len)
        if not flag:
            break
        tuning, sc_based_tuning, tot_s_vec = compute_average(spk[filter_trials],lam_s, events_analyze, temp_bins, sc_based_tuning, tuning, tot_s_vec,half_size,timept,
                                                             trial_idx[filter_trials])
        events = events - events_analyze
        events_analyze *= 0
        cc += 1
    tuning = tuning / tot_s_vec
    sc_based_tuning = sc_based_tuning / tot_s_vec
    return temp_bins, tuning / dt, sc_based_tuning / dt, tot_s_vec

def compute_average(spk, lam_s, events, temp_bins, sc_based_tuning, 
                    tuning, tot_s_vec,half_size,timept, trial_idx):
    rew_idx = np.where(events == 1)[0]
    time_kernel = np.ones(events.shape[0]) * np.inf
    for ind in rew_idx:
        if (ind < half_size) or (ind >= time_kernel.shape[0] - half_size):
            continue
        time_kernel[ind - half_size:ind + half_size + 1] = timept

    
    dt = temp_bins[1] - temp_bins[0]
    for tr in np.unique(trial_idx):
        cc = 0
        sel = trial_idx == tr
        time_kernel_tr = time_kernel[sel]
        lam_tr = lam_s[sel]
        spk_tr = spk[sel]
        for t0 in temp_bins:
            idx = (time_kernel_tr >= t0) * (time_kernel_tr < t0 + (dt))
            tuning[cc] = tuning[cc] + np.sum(lam_tr[idx])
            tot_s_vec[cc] = tot_s_vec[cc] + np.sum(idx)
            sc_based_tuning[cc] = sc_based_tuning[cc] + spk_tr[idx].sum()
            cc += 1
    return tuning, sc_based_tuning, tot_s_vec

def sum_trial(ev_sender, ev_reciever, rate_reciever, DT, num_DT):
    num_bin = int(np.ceil(num_DT/DT))
    counts_DT = np.zeros(num_bin)
    tp_DT = np.zeros(num_bin)
    idx_spk = np.where(ev_sender)[0]
    for ii in idx_spk:
        mn = max(ii-num_DT/2, 0)
        mx = min(ii+num_DT/2, len(idx_spk))
        cc = num_bin - mn
        for k in range(mn,mx):
            counts_DT[cc] = counts_DT[cc] + ev_reciever[k+ii]
            tp_DT[cc] = tp_DT[cc] + 1
            cc += 1
    return counts_DT, tp_DT

def postprocess_results(neuron_id,counts, full_fit, reduced_fit, train_bool,
                        sm_handler, family, trial_idx,var_zscore_par=None,info_save={},bins=30):


    dtypes = {
        'neuron_id':'U100',
        'fr':float,
        'full_pseudo_r2_train':float,
        'full_pseudo_r2_eval':float,
        'reduced_pseudo_r2_train':float,
        'reduced_pseudo_r2_eval':float,
        'variable':'U100',
        'pval':float,
        'reduced_pval':float,
        'mutual_info':float,
        'x_rate_Hz':object,
        'model_rate_Hz':object,
        'raw_rate_Hz':object,
        'reduced_x_rate_Hz':object,
        'reduced_model_rate_Hz':object,
        'reduced_raw_rate_Hz':object,
        'eval_x_rate_Hz': object,
        'eval_model_rate_Hz': object,
        'eval_raw_rate_Hz': object,
        'eval_reduced_x_rate_Hz': object,
        'eval_reduced_model_rate_Hz': object,
        'eval_reduced_raw_rate_Hz': object,
        'kernel_strength':float,
        'signed_kernel_strength':float,
        'reduced_kernel_strength':float,
        'reduced_signed_kernel_strength':float,
        'x_kernel':object,
        'y_kernel':object,
        'y_kernel_mCI':object,
        'y_kernel_pCI':object,
        'reduced_x_kernel':object,
        'reduced_y_kernel':object,
        'reduced_y_kernel_mCI':object,
        'reduced_y_kernel_pCI':object,
        'beta_full':object,
        'beta_reduced':object,
        'intercept_full':float,
        'intercept_reduced':float,
    }
    for name in info_save.keys():
        # set object as a type for unknown info save
        dtypes[name] = object
    dtype_dict = {'names':[], 'formats':[]}
    for name in dtypes.keys():
        dtype_dict['names'] += [name]
        dtype_dict['formats'] += [dtypes[name]]

    results = np.zeros(len((full_fit.var_list)), dtype=dtype_dict)
    for name in info_save.keys():
        results[name] = info_save[name]
    
    results['neuron_id'] = neuron_id
    results['fr'] = counts.mean()/full_fit.time_bin
    
    cs_table = full_fit.covariate_significance
    if not reduced_fit is None:
        cs_table_red = reduced_fit.covariate_significance
    else:
        cs_table_red = None

    exog_full = None
    exog_reduced = None
    if var_zscore_par is None:
        var_zscore_par = {}
        for var in full_fit.var_list:
            var_zscore_par[var]={'loc':0,'scale':1}

    for cc in range(len(full_fit.var_list)):
        var = full_fit.var_list[cc]
        print('processing: ', var)
        cs_var = cs_table[cs_table['covariate'] == var]
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                cs_var_red = cs_table_red[cs_table_red['covariate'] == var]



        results['variable'][cc] = var
        # results['trial_type'][cc] = trial_type
        results['full_pseudo_r2_train'][cc] = full_fit.pseudo_r2
        results['full_pseudo_r2_eval'][cc], exog_full = pseudo_r2_comp(counts, full_fit, sm_handler, family,
                                                                       use_tp=~(train_bool), exog=exog_full)
        if not reduced_fit is None:
            results['reduced_pseudo_r2_train'][cc] = reduced_fit.pseudo_r2
            results['reduced_pseudo_r2_eval'][cc], exog_reduced = pseudo_r2_comp(counts, reduced_fit, sm_handler,
                                                                                 family,
                                                                                 use_tp=~(train_bool),
                                                                                 exog=exog_reduced)
        results['pval'][cc] = cs_var['p-val']
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                results['reduced_pval'][cc] = cs_var_red['p-val']
            else:
                results['reduced_pval'][cc] = np.nan
        try:
            mi_full, tun_full = mutual_info_est(counts, exog_full, full_fit, var, sm_handler, train_bool, trial_idx, dt=full_fit.time_bin, bins=bins)

        except SystemError:
            mi_full = np.nan
            tun_full = empty_container()
            tun_full.x = np.nan
            tun_full.y_raw = np.nan
            tun_full.y_model = np.nan


        results['mutual_info'][cc] = mi_full
        if ~np.isnan(var_zscore_par[var]['loc']):
            xx = tun_full.x * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
        else:
            xx = tun_full.x

        results['x_rate_Hz'][cc] = xx
        results['model_rate_Hz'][cc] = tun_full.y_model
        results['raw_rate_Hz'][cc] = tun_full.y_raw

        try:
            mi_full, tun_full = mutual_info_est(counts, exog_full, full_fit, var, sm_handler, ~train_bool, trial_idx, dt=full_fit.time_bin, bins=bins)

        except:
            mi_full = np.nan
            tun_full = empty_container()
            tun_full.x = np.nan
            tun_full.y_raw = np.nan
            tun_full.y_model = np.nan


        if ~np.isnan(var_zscore_par[var]['loc']):
            xx = tun_full.x * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
        else:
            xx = tun_full.x

        results['eval_x_rate_Hz'][cc] = xx
        results['eval_model_rate_Hz'][cc] = tun_full.y_model
        results['eval_raw_rate_Hz'][cc] = tun_full.y_raw


        # compute kernel strength
        if full_fit.smooth_info[var]['is_temporal_kernel']:
            dim_kern = full_fit.smooth_info[var]['basis_kernel'].shape[0]
            x = np.zeros(dim_kern)
            x[(dim_kern - 1) // 2] = 1
            xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
            fX, fminus, fplus = full_fit.smooth_compute([x], var, 0.99)
            if (full_fit.smooth_info[var]['kernel_direction'] == 1) and\
                (full_fit.smooth_info[var]['is_event_input']):
                fminus = fminus[(dim_kern - 1) // 2:] - fX[0]
                fplus = fplus[(dim_kern - 1) // 2:] - fX[0]
                fX = fX[(dim_kern - 1) // 2:] - fX[0]
                xx2 = xx2[(dim_kern - 1) // 2:]
            else:
                fplus = fplus - fX[-1]
                fminus = fminus - fX[-1]
                fX = fX - fX[-1]

            results['kernel_strength'][cc] = simps(fX ** 2, dx=full_fit.time_bin) / (full_fit.time_bin * fX.shape[0])
            results['signed_kernel_strength'][cc] = simps(fX, dx=full_fit.time_bin) / (full_fit.time_bin * fX.shape[0])

        else:
            knots = full_fit.smooth_info[var]['knots']
            xmin = knots[0].min()
            xmax = knots[0].max()
            func = lambda x: (full_fit.smooth_compute([x], var, 0.99)[0] -
                              full_fit.smooth_compute([x], var, 0.99)[0].mean()) ** 2
            xx = np.linspace(xmin, xmax, 500)
            xx2 = np.linspace(xmin, xmax, 100)
            dx = xx[1] - xx[0]
            fX, fminus, fplus = full_fit.smooth_compute([xx2], var, 0.99)
            results['kernel_strength'][cc] = simps(func(xx), dx=dx) / (xmax - xmin)
        results['y_kernel'][cc] = fX
        results['y_kernel_pCI'][cc] = fplus
        results['y_kernel_mCI'][cc] = fminus

        if ~np.isnan(var_zscore_par[var]['loc']):
            xx2 = xx2 * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
        results['x_kernel'][cc] = xx2
        results['beta_full'][cc] = full_fit.beta[full_fit.index_dict[var]]
        results['intercept_full'][cc] = full_fit.beta[0]

        if not (reduced_fit is None):
            results['intercept_reduced'][cc] = reduced_fit.beta[0]
            if var in reduced_fit.var_list:

                results['beta_reduced'][cc] = reduced_fit.beta[reduced_fit.index_dict[var]]

                # compute kernel strength
                if reduced_fit.smooth_info[var]['is_temporal_kernel']:
                    dim_kern = reduced_fit.smooth_info[var]['basis_kernel'].shape[0]
                    x = np.zeros(dim_kern)
                    x[(dim_kern - 1) // 2] = 1
                    xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
                    fX, fminus, fplus = reduced_fit.smooth_compute([x], var, 0.99)
                    if (full_fit.smooth_info[var]['kernel_direction'] == 1) and\
                         (full_fit.smooth_info[var]['is_event_input']):
                        fminus = fminus[(dim_kern - 1) // 2:] - fX[0]
                        fplus = fplus[(dim_kern - 1) // 2:] - fX[0]
                        fX = fX[(dim_kern - 1) // 2:] - fX[0]
                        xx2 = xx2[(dim_kern - 1) // 2:]
                    else:
                        fplus = fplus - fX[-1]
                        fminus = fminus - fX[-1]
                        fX = fX - fX[-1]
                    results['reduced_kernel_strength'][cc] = simps(fX ** 2, dx=full_fit.time_bin) / (full_fit.time_bin * fX.shape[0])
                    results['reduced_signed_kernel_strength'][cc] = simps(fX, dx=full_fit.time_bin) / (full_fit.time_bin * fX.shape[0])

                else:
                    knots = full_fit.smooth_info[var]['knots']
                    xmin = knots[0].min()
                    xmax = knots[0].max()
                    func = lambda x: (reduced_fit.smooth_compute([x], var, 0.99)[0] -
                                      reduced_fit.smooth_compute([x], var, 0.99)[0].mean()) ** 2
                    xx = np.linspace(xmin, xmax, 500)
                    xx2 = np.linspace(xmin, xmax, 100)
                    dx = xx[1] - xx[0]
                    fX, fminus, fplus = reduced_fit.smooth_compute([xx2], var, 0.99)
                    results['reduced_kernel_strength'][cc] = simps(func(xx), dx=dx) / (xmax - xmin)

                results['reduced_y_kernel'][cc] = fX
                results['reduced_y_kernel_pCI'][cc] = fplus
                results['reduced_y_kernel_mCI'][cc] = fminus

                if ~np.isnan(var_zscore_par[var]['loc']):
                    xx2 = xx2 * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
                results['reduced_x_kernel'][cc] = xx2
                # results['beta_full'][cc] = full_fit.beta[full_fit.index_dict[var]]
                # results['intercept_full'][cc] = full_fit.beta[0]
                # results['intercept_reduced'][cc] = reduced_fit.beta[0]

            try:
                mi_red, tun_red = mutual_info_est(counts, exog_reduced, reduced_fit, var, sm_handler, train_bool,trial_idx,
                                                    dt=full_fit.time_bin, bins=20)
                
            except:
                mi_red = np.nan
                tun_red = empty_container()
                tun_red.x = np.nan
                tun_red.y_raw = np.nan
                tun_red.y_model = np.nan

            if ~np.isnan(var_zscore_par[var]['loc']):
                xx = tun_red.x * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
            else:
                xx = tun_red.x
            results['reduced_x_rate_Hz'][cc] = xx
            results['reduced_model_rate_Hz'][cc] = tun_red.y_model
            results['reduced_raw_rate_Hz'][cc] = tun_red.y_raw

            try:
                mi_red, tun_red = mutual_info_est(counts, exog_reduced, reduced_fit, var, sm_handler, ~train_bool,trial_idx,
                                                    dt=full_fit.time_bin, bins=20)
                
            except:
                tun_red = empty_container()
                tun_red.x = np.nan
                tun_red.y_raw = np.nan
                tun_red.y_model = np.nan

            if ~np.isnan(var_zscore_par[var]['loc']):
                xx = tun_red.x * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
            else:
                xx = tun_red.x
            results['eval_reduced_x_rate_Hz'][cc] = xx
            results['eval_reduced_model_rate_Hz'][cc] = tun_red.y_model
            results['eval_reduced_raw_rate_Hz'][cc] = tun_red.y_raw

    return results
@njit
def compute_event_trig_counts(ev_receiver, pred_rate_receiver, idx_spk, rate_DT,
                              counts_DT, tp_DT, tot_tp, delta_step,dt_ms,skip_t0):
    """
    Compute a fast spike triggered average from idx_spk (the sender spike) and pred_rate_receiver (the receiver pgam
    rate) given the receiver spikes (ev_receiver).
    rate_DT, counts_DT and tp_DT are container for the spike trig averages of size num_tp
    """


    for ii in idx_spk:
        mn = max(ii - tot_tp, 0)
        mx = min(ii + tot_tp, len(ev_receiver))
        cc = tot_tp - ii + mn
        for k in range(mn,mx):
            if skip_t0 and k == ii:
                continue
            counts_DT[delta_step[cc]] = counts_DT[delta_step[cc]] + ev_receiver[k]
            rate_DT[delta_step[cc]] = rate_DT[delta_step[cc]] + pred_rate_receiver[k]
            tp_DT[delta_step[cc]] = tp_DT[delta_step[cc]] + 1
            cc += 1
    return rate_DT, counts_DT, tp_DT


def compute_tuning_temporal_fast(var, lam_s, sm_handler, fit, spk, filter_trials, trial_idx, bins, dt):

    spk_sender = np.array(np.squeeze(sm_handler[var]._x)[filter_trials], dtype=np.int64)

    filter_len = np.int64(fit.smooth_info[var]['time_pt_for_kernel'].shape[0])
    if filter_len%2 == 1:
        filter_len = filter_len - 1

    tot_tp = filter_len // 2
    delta_step = np.array(np.arange(filter_len) // (np.floor(filter_len/bins) + 1), dtype=int)
    num_tp = np.unique(delta_step).shape[0]
    counts_DT = np.zeros(num_tp)
    rate_DT = np.zeros(num_tp)
    tp_DT = np.zeros(num_tp)


    skip_t0 = var == 'spike_hist'

    dt_ms = fit.time_bin
    edges = np.zeros(num_tp)
    cc = 0
    for val in np.unique(delta_step):
        edges[cc] = ( np.where(delta_step==val)[0].mean()-tot_tp) * dt
        cc += 1

    spk_filt = spk[filter_trials]
    lam_s_filt = lam_s[filter_trials]
    trial_idx_filt = trial_idx[filter_trials]

    for tr in np.unique(trial_idx_filt):
        sel = trial_idx_filt == tr
        spk_tr = spk_filt[sel]
        lam_s_tr = lam_s_filt[sel]
        idx_spk = np.where(spk_sender[sel])[0]


        rate_DT, counts_DT, tp_DT = compute_event_trig_counts(spk_tr, lam_s_tr, idx_spk, rate_DT,
                              counts_DT, tp_DT, tot_tp, delta_step, dt_ms, skip_t0)

    rate_DT = rate_DT / tp_DT
    counts_DT = counts_DT / tp_DT
    return edges, rate_DT / dt, counts_DT / dt, tp_DT