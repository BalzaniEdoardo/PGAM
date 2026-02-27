#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:33:36 2022

@author: edoardo
"""
import operator
from copy import deepcopy

import numpy as np
import scipy.stats as sts
from GAM_library import *
from numba import njit
from scipy.integrate import simpson as simps


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
    resid_deviance = np.sum(res_dev_t**2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t**2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2, exog


def compute_tuning(
    spk, fit, exog, var, sm_handler, filter_trials, trial_idx, dt=0.006, bins=15
):
    # print('comp tun')
    mu = np.dot(exog[filter_trials], fit.beta)

    # convert to rate space (the sima2 works for log-normal, only log link)
    if type(fit.family.link) == sm.genmod.families.links.log:
        sigma2 = np.einsum(
            "ij,jk,ik->i",
            exog[filter_trials],
            fit.cov_beta,
            exog[filter_trials],
            optimize=True,
        )
        lam_s = np.exp(mu + sigma2 * 0.5)
    else:  # do not consider uncertainty on the beta...
        lam_s = fit.family.fitted(mu)

    if (
        fit.smooth_info[var]["is_temporal_kernel"]
        and fit.smooth_info[var]["is_event_input"]
    ):
        # print('temp')
        x_axis, tuning, sc_based_tuning, counts = compute_tuning_temporal(
            var, lam_s, sm_handler, fit, spk, filter_trials, trial_idx, bins, dt
        )
        x_axis = x_axis.reshape(1, -1)
        tuning = tuning.reshape(1, -1)
        sc_based_tuning = sc_based_tuning.reshape(1, -1)
        counts = counts.reshape(1, -1)
    else:
        # print('spat')
        x_axis, tuning, sc_based_tuning, counts = compute_tuning_spatial(
            var, lam_s, sm_handler, fit, spk, filter_trials, bins, dt
        )
    return x_axis, tuning, sc_based_tuning, counts


def mutual_info_est(
    spk_counts,
    exog,
    fit,
    var,
    sm_handler,
    filter_trials,
    trial_idx,
    dt=0.006,
    bins=15,
    min_occupancy_sec=0.2,
):
    # print('here')
    temp_bins, tuning, sc_based_tuning, count_bins = compute_tuning(
        spk_counts,
        fit,
        exog,
        var,
        sm_handler,
        filter_trials,
        trial_idx,
        dt=dt,
        bins=bins,
    )
    tuning = tuning * dt
    smooth_info = fit.smooth_info
    D = np.prod(tuning.shape)
    entropy_s = np.zeros(D) * np.nan
    bl_use = (
        count_bins.flatten() > min_occupancy_sec / dt
    )  # at least 500ms to estimate rate
    for cc in range(D):
        if bl_use[cc]:
            entropy_s[cc] = sts.poisson.entropy(tuning.flatten()[cc])

    if (
        (smooth_info[var]["kernel_direction"] == 1)
        and (smooth_info[var]["is_temporal_kernel"])
        and (smooth_info[var]["is_event_input"])
    ):
        sel = temp_bins > 0
        temp_bins = temp_bins[sel].reshape(1, -1)
        count_bins = count_bins[sel].reshape(1, -1)
        tuning = tuning[sel].reshape(1, -1)
        sc_based_tuning = sc_based_tuning[sel].reshape(1, -1)
        entropy_s = entropy_s[sel[0]]
        bl_use = bl_use[sel[0]]

    elif (
        (smooth_info[var]["kernel_direction"] == -1)
        and (smooth_info[var]["is_temporal_kernel"])
        and (smooth_info[var]["is_event_input"])
    ):
        sel = temp_bins < 0
        temp_bins = temp_bins[sel]
        count_bins = count_bins[sel]
        tuning = tuning[sel]
        sc_based_tuning = sc_based_tuning[sel]
        entropy_s = entropy_s[sel[0]]
        bl_use = bl_use[sel[0]]

    prob_s = count_bins.flatten()[bl_use] / count_bins.flatten()[bl_use].sum()
    mean_lam = np.sum(prob_s * tuning.flatten()[bl_use])

    try:
        mi = (
            (sts.poisson.entropy(mean_lam) - np.nansum(prob_s * entropy_s[bl_use]))
            * np.log2(np.exp(1))
            / dt
        )
    except:
        mi = np.nan

    tmp_val = empty_container()
    tmp_val.x = temp_bins
    tmp_val.y_raw = sc_based_tuning
    tmp_val.y_model = tuning / dt

    return mi, tmp_val


def alignRateForMI(
    y, lam_s, var, sm_handler, smooth_info, time_bin, filter_trials, trial_idx
):
    """
    Slow aligment method.
    """
    reward = np.squeeze(sm_handler[var]._x)[filter_trials]
    # temp kernel where 161 timepoints long
    size_kern = smooth_info[var]["time_pt_for_kernel"].shape[0]
    if size_kern % 2 == 0:
        size_kern += 1
    half_size = (size_kern - 1) // 2
    timept = np.arange(-half_size, half_size + 1) * time_bin
    if (var.startswith("neu")) or var == "spike_hist":
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
        for ii in np.where(rwd_tr == 1)[0]:
            i0 = max(0, ii - half_size)
            i1 = min(len(rwd_tr), ii + half_size + 1)
            d0 = ii - i0
            d1 = i1 - ii
            tmpmu = lam_s_tr[i0:i1]
            tmpy = y_tr[i0:i1]
            iidx = np.array(
                np.round(
                    nbin // 2
                    + (-d0 + np.arange(0, d1 + d0))
                    * time_bin
                    / (temp_bins[1] - temp_bins[0])
                ),
                dtype=int,
            )
            for cc in np.unique(iidx):
                tuning[cc] = tuning[cc] + tmpmu[iidx == cc].sum()
                count_bins[cc] = count_bins[cc] + (iidx == cc).sum()
                sc_based_tuning[cc] = sc_based_tuning[cc] + tmpy[iidx == cc].sum()

    tuning = tuning / count_bins
    sc_based_tuning = sc_based_tuning / count_bins

    entropy_s = np.zeros(temp_bins.shape[0]) * np.nan
    for cc in range(tuning.shape[0]):
        entropy_s[cc] = sts.poisson.entropy(tuning[cc])

    if (var.startswith("neu")) or var == "spike_hist":
        sel = temp_bins > 0
        temp_bins = temp_bins[sel]
        count_bins = count_bins[sel]
        tuning = tuning[sel]
        sc_based_tuning = sc_based_tuning[sel]
        entropy_s = entropy_s[sel]
    prob_s = count_bins / count_bins.sum()
    mean_lam = np.sum(prob_s * tuning)

    try:
        mi = (
            (sts.poisson.entropy(mean_lam) - (prob_s * entropy_s).sum())
            * np.log2(np.exp(1))
            / time_bin
        )
    except:
        mi = np.nan

    tmp_val = empty_container()
    tmp_val.x = temp_bins
    tmp_val.y_raw = sc_based_tuning / time_bin
    tmp_val.y_model = tuning / time_bin

    return mi, tmp_val


def find_first_x_bin(result, num_events, bins):
    flag = False
    skip_until = 0  # nb.int64([0])
    for idx, val in np.ndenumerate(num_events):
        # print(idx,type(idx))
        ii = idx[0]  # nb.int64(idx)
        if ii < skip_until:
            continue
        if val != 0:
            result[idx] = 1
            skip_until = ii + bins
            flag = True
    return result, flag


# Modified from numpy.histogramdd
def multidim_digitize(sample, bins=10):
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    # dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of the "
                " sample x."
            )
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    _range = (None,) * D

    # Create edge arrays
    for i in range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    "`bins[{}]` must be positive, when an integer".format(i)
                )
            smin, smax = np.lib.histograms._get_outer_edges(sample[:, i], _range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                    "`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    "`bins[{}]` must be monotonically increasing, when an array".format(
                        i
                    )
                )
        else:
            raise ValueError("`bins[{}]` must be a scalar or 1d array".format(i))

        nbin[i] = len(edges[i]) - 1

        # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side="right") - 1
        for i in range(D)
    )

    # grab the indexes which are out of range (Ncount == -1 and len(edges))
    # clip, then ad nans
    invalid = np.zeros(sample.shape[0], dtype=bool)
    for i, nct in enumerate(Ncount):
        invalid += (nct == -1) | (nct == nbin[i])

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    if len(nbin) > 1:
        digit = np.ravel_multi_index(Ncount, nbin, mode="clip")
    else:
        digit = Ncount[0]
    cedges = edges.copy()
    for i in range(D):
        cedges[i] = 0.5 * (cedges[i][:-1] + cedges[i][1:])

    XY = np.meshgrid(*cedges)
    digit = np.asarray(digit, dtype=float)
    digit[invalid] = np.nan
    return digit, XY


def compute_tuning_spatial(var, lam_s, sm_handler, fit, spk, filter_trials, bins, dt):
    # check dimensionality of the basis

    # this gives error for 2d variable
    vels = sm_handler[var]._x[:, filter_trials].T
    vel_bins = []
    for i in range(vels.shape[1]):
        knots = fit.smooth_info[var]["knots"][i]
        vel_bins.append(np.linspace(knots[0], knots[-1] - 0.000001, bins + 1))
    digit_vels, XY = multidim_digitize(vels, np.asarray(vel_bins))

    tuning = np.zeros(XY[0].shape) * np.nan
    sc_based_tuning = np.zeros(XY[0].shape) * np.nan
    tot_s_vec = np.zeros(XY[0].shape)

    if vels.shape[1] == 1:
        tuning = tuning.reshape(1, -1)
        sc_based_tuning = sc_based_tuning.reshape(1, -1)
        tot_s_vec = tot_s_vec.reshape(1, -1)

    for ii in np.unique(digit_vels):
        if np.isnan(ii):
            continue
        if ii == -1 or ii == np.prod(tuning.shape):
            continue
        idx = digit_vels == ii
        if vels.shape[1] > 1:
            row, col = np.unravel_index(int(ii), tuning.shape)
        else:
            row = 0
            col = int(ii)
        tuning[row, col] = np.nanmean(lam_s[idx])
        sc_based_tuning[row, col] = spk[filter_trials][idx].mean()
        tot_s_vec[row, col] = np.sum(idx)

    return XY, tuning / dt, sc_based_tuning / dt, tot_s_vec


def compute_tuning_temporal(
    var, lam_s, sm_handler, fit, spk, filter_trials, trial_idx, bins, dt
):
    events = np.array(np.squeeze(sm_handler[var]._x)[filter_trials], dtype=np.int64)
    events_analyze = np.zeros(events.shape, dtype=np.int64)
    filter_len = np.int64(fit.smooth_info[var]["time_pt_for_kernel"].shape[0])
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
    # dt = temp_bins[1] - temp_bins[0]
    tuning = np.zeros(temp_bins.shape[0])
    sc_based_tuning = np.zeros(temp_bins.shape[0])
    tot_s_vec = np.zeros(temp_bins.shape[0])

    while flag:
        events_analyze, flag = find_first_x_bin(events_analyze, events, filter_len)
        if not flag:
            break
        tuning, sc_based_tuning, tot_s_vec = compute_average(
            spk[filter_trials],
            lam_s,
            events_analyze,
            temp_bins,
            sc_based_tuning,
            tuning,
            tot_s_vec,
            half_size,
            timept,
            trial_idx[filter_trials],
        )
        events = events - events_analyze
        events_analyze *= 0
        cc += 1
    tuning = tuning / tot_s_vec
    sc_based_tuning = sc_based_tuning / tot_s_vec
    return temp_bins, tuning / dt, sc_based_tuning / dt, tot_s_vec


def compute_average(
    spk,
    lam_s,
    events,
    temp_bins,
    sc_based_tuning,
    tuning,
    tot_s_vec,
    half_size,
    timept,
    trial_idx,
):
    rew_idx = np.where(events == 1)[0]
    time_kernel = np.ones(events.shape[0]) * np.inf
    for ind in rew_idx:
        if (ind < half_size) or (ind >= time_kernel.shape[0] - half_size):
            continue
        time_kernel[ind - half_size : ind + half_size + 1] = timept

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


@njit
def replace_nan_2D(tuning, tuning_fill, rows, cols, neigh=1):
    for cc in range(rows.shape[0]):
        val = 0
        cnt = 0
        row = rows[cc]
        col = cols[cc]
        for dr in range(-neigh, neigh + 1):
            if (row + dr < 0) or (row + dr >= tuning.shape[0]):
                continue
            for dc in range(-neigh, neigh + 1):
                if (col + dc < 0) or (col + dc >= tuning.shape[1]):
                    continue
                if ~np.isnan(tuning[row + dr, col + dc]):
                    val += tuning[row + dr, col + dc]
                    cnt += 1

        tuning_fill[row, col] = val / cnt
    return tuning_fill


def multi_int(Z, x_list):
    """
    Numerical estimate a N-dimensional integral on a hyper-rectangle.
    Calls a recursion that integrates on all coordinates.

    Z : (samples )^N dimensional tensor of F evaluated in x_list
        Z[i,j,...k] = F[x_list[0][i], x_list[1][j], ...., x_list[N-1][k]]
    x_list: N-dimensioinal list of the hyperrecangle coordinates
    """
    xx = x_list[0]
    if len(Z.shape) == 1:
        zz = Z
    else:
        zz = np.zeros(Z.shape[0])
        for i in range(Z.shape[0]):
            zz[i] = multi_int(Z[i], x_list[1:])
    return simps(zz, xx)


def prediction_and_kernel_str(fit, var, var_zscore_par):
    if fit is None:
        return (np.nan,) * 6
    if not var in fit.var_list:
        return (np.nan,) * 6
    if fit.smooth_info[var]["is_temporal_kernel"]:
        return temporal_prediction_and_kernel_str(fit, var)
    else:
        return spatial_prediction_and_kernel_str(fit, var, var_zscore_par)


def spatial_prediction_and_kernel_str(fit, var, var_zscore_par):
    knots = fit.smooth_info[var]["knots"]
    D = len(knots)
    dx = np.zeros(D)
    x_list = []
    x_list_simps = []
    for i in range(D):
        xmin = knots[i].min()
        xmax = knots[i].max()
        xx = np.linspace(xmin, xmax, 200)
        xx2 = np.linspace(xmin, xmax, 100)
        dx[i] = xx[1] - xx[0]
        x_list.append(xx2)
        x_list_simps.append(xx)

        # simps([simps(zz_x,x) for zz_x in zz],y)
    XY = np.meshgrid(*x_list)
    shapeXY = XY[0].shape
    XY = [XY[k].flatten() for k in range(len(XY))]
    fX, fminus, fplus = fit.smooth_compute(XY, var, 0.99)

    # xx2 = xx2 * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
    XY = np.array(
        [
            XY[k].reshape(shapeXY) * var_zscore_par[var]["scale"]
            + var_zscore_par[var]["loc"]
            for k in range(len(XY))
        ]
    )
    fX = fX.reshape(shapeXY)
    fminus = fminus.reshape(shapeXY)
    fplus = fplus.reshape(shapeXY)

    # compute integral
    func = (
        lambda x: (
            fit.smooth_compute(x, var, 0.99)[0]
            - fit.smooth_compute(x, var, 0.99)[0].mean()
        )
        ** 2
    )
    XY2 = np.meshgrid(*x_list_simps)
    Z = func([XY2[k].flatten() for k in range(len(XY))])
    kern_str = multi_int(Z.reshape(XY2[0].shape), x_list_simps)
    return XY, fX, fminus, fplus, kern_str, np.nan


def temporal_prediction_and_kernel_str(fit, var):
    dim_kern = fit.smooth_info[var]["basis_kernel"].shape[0]
    x = np.zeros(dim_kern)
    x[(dim_kern - 1) // 2] = 1
    xx2 = np.arange(x.shape[0]) * fit.time_bin - np.where(x)[0][0] * fit.time_bin
    fX, fminus, fplus = fit.smooth_compute([x], var, 0.99)
    if (fit.smooth_info[var]["kernel_direction"] == 1) and (
        fit.smooth_info[var]["is_event_input"]
    ):
        sel = xx2 > 0
        fminus = fminus[sel] - fX[0]
        fplus = fplus[sel] - fX[0]
        fX = fX[sel] - fX[0]
        xx2 = xx2[sel]
    elif (fit.smooth_info[var]["kernel_direction"] == -1) and (
        fit.smooth_info[var]["is_event_input"]
    ):
        sel = xx2 < 0
        fminus = fminus[sel] - fX[-1]
        fplus = fplus[sel] - fX[-1]
        fX = fX[sel] - fX[-1]
        xx2 = xx2[sel]
    else:
        fplus = fplus - fX[-1]
        fminus = fminus - fX[-1]
        fX = fX - fX[-1]
    xx2 = xx2.reshape(1, -1)
    kern_str = simps(fX**2, dx=fit.time_bin) / (fit.time_bin * fX.shape[0])
    signed_kern_str = simps(fX, dx=fit.time_bin) / (fit.time_bin * fX.shape[0])
    return xx2, fX, fminus, fplus, kern_str, signed_kern_str


def postprocess_results(
    neuron_id,
    counts,
    full_fit,
    reduced_fit,
    train_bool,
    sm_handler,
    family,
    trial_idx,
    var_zscore_par=None,
    info_save={},
    bins=30,
):
    dtypes = {
        "neuron_id": "U100",
        "variable": "U100",
        "fr": float,
        "full_pseudo_r2_train": float,
        "full_pseudo_r2_eval": float,
        "reduced_pseudo_r2_train": float,
        "reduced_pseudo_r2_eval": float,
        "pval": float,
        "reduced_pval": float,
        "x_rate_Hz": object,
        "y_rate_Hz_model": object,
        "y_rate_Hz_raw": object,
        "reduced_x_rate_Hz": object,
        "reduced_y_rate_Hz_model": object,
        "reduced_y_rate_Hz_raw": object,
        "eval_x_rate_Hz": object,
        "eval_y_rate_Hz_model": object,
        "eval_y_rate_Hz_raw": object,
        "eval_reduced_x_rate_Hz": object,
        "eval_reduced_y_rate_Hz_model": object,
        "eval_reduced_y_rate_Hz_raw": object,
        "kernel_strength": float,
        "signed_kernel_strength": float,
        "reduced_kernel_strength": float,
        "reduced_signed_kernel_strength": float,
        "x_kernel": object,
        "y_kernel": object,
        "y_kernel_mCI": object,
        "y_kernel_pCI": object,
        "reduced_x_kernel": object,
        "reduced_y_kernel": object,
        "reduced_y_kernel_mCI": object,
        "reduced_y_kernel_pCI": object,
        "beta_full": object,
        "beta_reduced": object,
        "intercept_full": float,
        "intercept_reduced": float,
        "mutual_info": float,
        "penalization": object,
    }
    for name in info_save.keys():
        # set object as a type for unknown info save
        dtypes[name] = object

    dtype_dict = {"names": [], "formats": []}
    for name in dtypes.keys():
        dtype_dict["names"] += [name]
        dtype_dict["formats"] += [dtypes[name]]

    results = np.zeros(len((full_fit.var_list)), dtype=dtype_dict)
    for name in info_save.keys():
        results[name] = info_save[name]

    results["neuron_id"] = neuron_id
    results["fr"] = counts.mean() / full_fit.time_bin

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
            var_zscore_par[var] = {"loc": 0, "scale": 1}

    for cc in range(len(full_fit.var_list)):
        var = full_fit.var_list[cc]
        print('processing: ', var)
        cs_var = cs_table[cs_table["covariate"] == var]
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                cs_var_red = cs_table_red[cs_table_red["covariate"] == var]

        results["variable"][cc] = var
        results["penalization"][cc] = sm_handler[var].lam
        # results['trial_type'][cc] = trial_type
        results["full_pseudo_r2_train"][cc] = full_fit.pseudo_r2
        results["full_pseudo_r2_eval"][cc], exog_full = pseudo_r2_comp(
            counts, full_fit, sm_handler, family, use_tp=~(train_bool), exog=exog_full
        )
        if not reduced_fit is None:
            results["reduced_pseudo_r2_train"][cc] = reduced_fit.pseudo_r2
            results["reduced_pseudo_r2_eval"][cc], exog_reduced = pseudo_r2_comp(
                counts,
                reduced_fit,
                sm_handler,
                family,
                use_tp=~(train_bool),
                exog=exog_reduced,
            )
        results["pval"][cc] = cs_var["p-val"]
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                results["reduced_pval"][cc] = cs_var_red["p-val"]
            else:
                results["reduced_pval"][cc] = np.nan
        try:
            mi_full, tun_full = mutual_info_est(
                counts,
                exog_full,
                full_fit,
                var,
                sm_handler,
                train_bool,
                trial_idx,
                dt=full_fit.time_bin,
                bins=bins,
            )

        except SystemError:
            mi_full = np.nan
            tun_full = empty_container()
            tun_full.x = np.nan
            tun_full.y_raw = np.nan
            tun_full.y_model = np.nan

        results["mutual_info"][cc] = mi_full
        if ~np.isnan(var_zscore_par[var]["loc"]):
            xx_scaled = []
            for kk in range(len(tun_full.x)):
                xx = (
                    tun_full.x[kk] * var_zscore_par[var]["scale"]
                    + var_zscore_par[var]["loc"]
                )
                xx_scaled.append(xx)
        else:
            xx_scaled = tun_full.x

        results["x_rate_Hz"][cc] = xx_scaled
        results["y_rate_Hz_model"][cc] = tun_full.y_model
        results["y_rate_Hz_raw"][cc] = tun_full.y_raw

        try:
            mi_full, tun_full = mutual_info_est(
                counts,
                exog_full,
                full_fit,
                var,
                sm_handler,
                ~train_bool,
                trial_idx,
                dt=full_fit.time_bin,
                bins=bins,
            )
        except:
            mi_full = np.nan
            tun_full = empty_container()
            tun_full.x = np.nan
            tun_full.y_raw = np.nan
            tun_full.y_model = np.nan

        if ~np.isnan(var_zscore_par[var]["loc"]):
            xx_scaled = []
            for kk in range(len(tun_full.x)):
                xx = (
                    tun_full.x[kk] * var_zscore_par[var]["scale"]
                    + var_zscore_par[var]["loc"]
                )
                xx_scaled.append(xx)
        else:
            xx_scaled = tun_full.x

        results["eval_x_rate_Hz"][cc] = xx_scaled
        results["eval_y_rate_Hz_model"][cc] = tun_full.y_model
        results["eval_y_rate_Hz_raw"][cc] = tun_full.y_raw

        # compute kernel strength
        (
            results["x_kernel"][cc],
            results["y_kernel"][cc],
            results["y_kernel_mCI"][cc],
            results["y_kernel_pCI"][cc],
            results["kernel_strength"][cc],
            results["signed_kernel_strength"][cc],
        ) = prediction_and_kernel_str(full_fit, var, var_zscore_par)

        (
            results["reduced_x_kernel"][cc],
            results["reduced_y_kernel"][cc],
            results["reduced_y_kernel_mCI"][cc],
            results["reduced_y_kernel_pCI"][cc],
            _,
            _,
        ) = prediction_and_kernel_str(reduced_fit, var, var_zscore_par)

        results["beta_full"][cc] = full_fit.beta[full_fit.index_dict[var]]
        results["intercept_full"][cc] = full_fit.beta[0]

        if not (reduced_fit is None):
            results["intercept_reduced"][cc] = reduced_fit.beta[0]
            try:
                mi_red, tun_red = mutual_info_est(
                    counts,
                    exog_reduced,
                    reduced_fit,
                    var,
                    sm_handler,
                    train_bool,
                    trial_idx,
                    dt=full_fit.time_bin,
                    bins=bins,
                )
                if ~np.isnan(var_zscore_par[var]["loc"]):
                    xx_scaled = []
                    for kk in range(len(tun_red.x)):
                        xx = (
                            tun_red.x[kk] * var_zscore_par[var]["scale"]
                            + var_zscore_par[var]["loc"]
                        )
                        xx_scaled.append(xx)

                else:
                    xx_scaled = tun_red.x
            except:
                mi_red = np.nan
                tun_red = empty_container()
                tun_red.x = np.nan
                tun_red.y_raw = np.nan
                tun_red.y_model = np.nan
                xx_scaled = np.nan

            results["reduced_x_rate_Hz"][cc] = xx_scaled
            results["reduced_y_rate_Hz_model"][cc] = tun_red.y_model
            results["reduced_y_rate_Hz_raw"][cc] = tun_red.y_raw

            try:
                mi_red, tun_red = mutual_info_est(
                    counts,
                    exog_reduced,
                    reduced_fit,
                    var,
                    sm_handler,
                    ~train_bool,
                    trial_idx,
                    dt=full_fit.time_bin,
                    bins=bins,
                )

                if ~np.isnan(var_zscore_par[var]["loc"]):
                    xx_scaled = []
                    for kk in range(len(tun_red.x)):
                        xx = (
                            tun_red.x[kk] * var_zscore_par[var]["scale"]
                            + var_zscore_par[var]["loc"]
                        )
                        xx_scaled.append(xx)
                else:
                    xx_scaled = tun_red.x
            except:
                tun_red = empty_container()
                tun_red.x = np.nan
                tun_red.y_raw = np.nan
                tun_red.y_model = np.nan
                xx_scaled = np.nan

            results["eval_reduced_x_rate_Hz"][cc] = xx_scaled
            results["eval_reduced_y_rate_Hz_model"][cc] = tun_red.y_model
            results["eval_reduced_y_rate_Hz_raw"][cc] = tun_red.y_raw

    return results


def sum_trial(ev_sender, ev_reciever, rate_reciever, DT, num_DT):
    num_bin = int(np.ceil(num_DT / DT))
    counts_DT = np.zeros(num_bin)
    tp_DT = np.zeros(num_bin)
    idx_spk = np.where(ev_sender)[0]
    for ii in idx_spk:
        mn = max(ii - num_DT / 2, 0)
        mx = min(ii + num_DT / 2, len(idx_spk))
        cc = num_bin - mn
        for k in range(mn, mx):
            counts_DT[cc] = counts_DT[cc] + ev_reciever[k + ii]
            tp_DT[cc] = tp_DT[cc] + 1
            cc += 1
    return counts_DT, tp_DT


@njit
def compute_event_trig_counts(
    ev_receiver,
    pred_rate_receiver,
    idx_spk,
    rate_DT,
    counts_DT,
    tp_DT,
    tot_tp,
    delta_step,
    dt_ms,
    skip_t0,
):
    """
    Compute a fast spike triggered average from idx_spk (the sender spike) and pred_rate_receiver (the receiver pgam
    rate) given the receiver spikes (ev_receiver).
    rate_DT, counts_DT and tp_DT are container for the spike trig averages of size num_tp
    """

    for ii in idx_spk:
        mn = max(ii - tot_tp, 0)
        mx = min(ii + tot_tp, len(ev_receiver))
        cc = tot_tp - ii + mn
        for k in range(mn, mx):
            if skip_t0 and k == ii:
                continue
            counts_DT[delta_step[cc]] = counts_DT[delta_step[cc]] + ev_receiver[k]
            rate_DT[delta_step[cc]] = rate_DT[delta_step[cc]] + pred_rate_receiver[k]
            tp_DT[delta_step[cc]] = tp_DT[delta_step[cc]] + 1
            cc += 1
    return rate_DT, counts_DT, tp_DT


def compute_tuning_temporal_fast(
    var, lam_s, sm_handler, fit, spk, filter_trials, trial_idx, bins, dt
):
    spk_sender = np.array(np.squeeze(sm_handler[var]._x)[filter_trials], dtype=np.int64)

    filter_len = np.int64(fit.smooth_info[var]["time_pt_for_kernel"].shape[0])
    if filter_len % 2 == 1:
        filter_len = filter_len - 1

    tot_tp = filter_len // 2
    delta_step = np.array(
        np.arange(filter_len) // (np.floor(filter_len / bins) + 1), dtype=int
    )
    num_tp = np.unique(delta_step).shape[0]
    counts_DT = np.zeros(num_tp)
    rate_DT = np.zeros(num_tp)
    tp_DT = np.zeros(num_tp)

    skip_t0 = var == "spike_hist"

    dt_ms = fit.time_bin
    edges = np.zeros(num_tp)
    cc = 0
    for val in np.unique(delta_step):
        edges[cc] = (np.where(delta_step == val)[0].mean() - tot_tp) * dt
        cc += 1

    spk_filt = spk[filter_trials]
    lam_s_filt = lam_s[filter_trials]
    trial_idx_filt = trial_idx[filter_trials]

    for tr in np.unique(trial_idx_filt):
        sel = trial_idx_filt == tr
        spk_tr = spk_filt[sel]
        lam_s_tr = lam_s_filt[sel]
        idx_spk = np.where(spk_sender[sel])[0]

        rate_DT, counts_DT, tp_DT = compute_event_trig_counts(
            spk_tr,
            lam_s_tr,
            idx_spk,
            rate_DT,
            counts_DT,
            tp_DT,
            tot_tp,
            delta_step,
            dt_ms,
            skip_t0,
        )

    rate_DT = rate_DT / tp_DT
    counts_DT = counts_DT / tp_DT
    return edges, rate_DT / dt, counts_DT / dt, tp_DT
