import numpy as np
from copy import deepcopy
from scipy.integrate import simps
import matplotlib.pylab as plt
from numba import jit
import numba as nb
import inspect,os
dir_fh = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_gamlib = os.path.join(os.path.dirname(dir_fh),'GAM_library')
import GAM_library as gl

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


def compute_tuning(spk, fit, exog, var, sm_handler, filter_trials, dt=0.006,bins=15):
    mu = np.dot(exog[filter_trials], fit.beta)
    sigma2 = np.einsum('ij,jk,ik->i', exog[filter_trials], fit.cov_beta, exog[filter_trials],
                       optimize=True)
    # convert to rate space
    lam_s = np.exp(mu + sigma2 * 0.5)
    if fit.smooth_info[var]['is_temporal_kernel'] and fit.smooth_info[var]['is_event_input']:
        x_axis, tuning, sc_based_tuning, counts = compute_tuning_temporal(var, lam_s, sm_handler, fit, spk, filter_trials, bins, dt)
    else:
        x_axis, tuning, sc_based_tuning,counts = compute_tuning_spatial(var, lam_s, sm_handler, fit, spk, filter_trials, bins,dt)
    return x_axis, tuning, sc_based_tuning, counts

def mutual_info_est(spk_counts, exog, fit, var, sm_handler, filter_trials, dt=0.006,bins=15):
    temp_bins, tuning, sc_based_tuning, count_bins = compute_tuning(spk_counts, fit, exog, var, sm_handler, filter_trials, dt=0.006, bins=bins)
    tuning = tuning * dt
    smooth_info = fit.smooth_info
    entropy_s = np.zeros(temp_bins.shape[0]) * np.nan
    for cc in range(tuning.shape[0]):
        entropy_s[cc] = sts.poisson.entropy(tuning[cc])

    if (smooth_info[var]['kernel_direction'] == 1) and\
            (smooth_info[var]['is_temporal_kernel']) and (smooth_info[var]['is_event_input']):

        sel = temp_bins >= 0
        temp_bins = temp_bins[sel]
        count_bins = count_bins[sel]
        tuning = tuning[sel]
        sc_based_tuning = sc_based_tuning[sel]
        entropy_s = entropy_s[sel]

    elif (smooth_info[var]['kernel_direction'] == -1) and\
            (smooth_info[var]['is_temporal_kernel']) and (smooth_info[var]['is_event_input']):
        sel = temp_bins <= 0
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

    tmp_val = gl.empty_container()
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


@jit(nb.types.Tuple((nb.int64[:], nb.bool_))(nb.int64[:], nb.int64[:], nb.int64),nopython=True)
def find_first_x_bin(result, num_events, bins):
    flag = False
    skip_until = nb.int64([0])
    for idx, val in np.ndenumerate(num_events):
        ii = nb.int64(idx)
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

def compute_tuning_temporal(var, lam_s, sm_handler, fit, spk, filter_trials, bins, dt):
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
        print(events.sum())
        events_analyze, flag = find_first_x_bin(events_analyze, events, filter_len)
        if not flag:
            break
        tuning, sc_based_tuning, tot_s_vec = compute_average(spk[filter_trials],lam_s, events_analyze, temp_bins, sc_based_tuning, tuning, tot_s_vec,half_size,timept)
        events = events - events_analyze
        events_analyze *= 0
        cc += 1
    tuning = tuning / tot_s_vec
    sc_based_tuning = sc_based_tuning / tot_s_vec
    return temp_bins, tuning / dt, sc_based_tuning / dt, tot_s_vec

def compute_average(spk, lam_s, events, temp_bins, sc_based_tuning, tuning, tot_s_vec,half_size,timept):
    rew_idx = np.where(events == 1)[0]
    time_kernel = np.ones(events.shape[0]) * np.inf
    for ind in rew_idx:
        if (ind < half_size) or (ind >= time_kernel.shape[0] - half_size):
            continue
        time_kernel[ind - half_size:ind + half_size + 1] = timept

    cc = 0
    dt = temp_bins[1] - temp_bins[0]
    for t0 in temp_bins:
        idx = (time_kernel >= t0) * (time_kernel < t0 + (dt))
        tuning[cc] = tuning[cc] + np.sum(lam_s[idx])
        tot_s_vec[cc] = tot_s_vec[cc] + np.sum(idx)
        sc_based_tuning[cc] = sc_based_tuning[cc] + spk[idx].sum()
        cc += 1
    return tuning, sc_based_tuning, tot_s_vec



def postprocess_results(counts, full_fit, reduced_fit, filter_trials,
                        sm_handler, family, var_zscore_par,info_save={}):


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

    cs_table = full_fit.covariate_significance
    if not reduced_fit is None:
        cs_table_red = reduced_fit.covariate_significance
    else:
        cs_table_red = None

    exog_full = None
    exog_reduced = None

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
                                                                       use_tp=~(filter_trials), exog=exog_full)
        if not reduced_fit is None:
            results['reduced_pseudo_r2_train'][cc] = reduced_fit.pseudo_r2
            results['reduced_pseudo_r2_eval'][cc], exog_reduced = pseudo_r2_comp(counts, reduced_fit, sm_handler,
                                                                                 family,
                                                                                 use_tp=~(filter_trials),
                                                                                 exog=exog_reduced)
        results['pval'][cc] = cs_var['p-val']
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                results['reduced_pval'][cc] = cs_var_red['p-val']
            else:
                results['reduced_pval'][cc] = np.nan
        try:
            mi_full, tun_full = mutual_info_est(spk_counts, exog_full, full_fit, var, sm_handler, filter_trials, dt=full_fit.time_bin, bins=20)

        except:
            mi_full = np.nan
            tun_full = gl.empty_container()
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
            mi_full, tun_full = mutual_info_est(spk_counts, exog_full, full_fit, var, sm_handler, ~filter_trials, dt=full_fit.time_bin, bins=20)

        except:
            mi_full = np.nan
            tun_full = gl.empty_container()
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

                    results['reduced_kernel_strength'][cc] = simps(fX ** 2, dx=0.006) / (0.006 * fX.shape[0])
                    results['reduced_signed_kernel_strength'][cc] = simps(fX, dx=0.006) / (0.006 * fX.shape[0])

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
                mi_red, tun_red = mutual_info_est(spk_counts, exog_reduced, reduced_fit, var, sm_handler, filter_trials,
                                                    dt=full_fit.time_bin, bins=20)
            except:
                mi_red = np.nan
                tun_red = gl.empty_container()
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
                mi_red, tun_red = mutual_info_est(spk_counts, exog_reduced, reduced_fit, var, sm_handler, ~filter_trials,
                                                    dt=full_fit.time_bin, bins=20)
            except:
                mi_red = np.nan
                tun_red = gl.empty_container()
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


if __name__ == '__main__':
    import scipy.stats as sts
    import statsmodels.api as sm
    import sys
    import gam_data_handlers as gdh
    ## inputs parameters
    num_events = 6000
    time_points = 3 * 10 ** 5  # 30 mins at 0.006 ms resolution
    rate = 5. * 0.006  # Hz rate of the final kernel
    variance = 5.  # spatial input and nuisance variance
    int_knots_num = 20  # num of internal knots for the spline basis
    order = 4  # spline order

    ## assume 200 trials
    trial_ids = np.repeat(np.arange(200), time_points // 200)

    ## create temporal input
    idx = np.random.choice(np.arange(time_points), num_events, replace=False)
    events = np.zeros(time_points)
    events[idx] = 1

    rv = sts.multivariate_normal(mean=[0, 0], cov=variance * np.eye(2))
    samp = rv.rvs(time_points)
    spatial_var = samp[:, 0]
    nuisance_var = samp[:, 1]

    # truncate X to avoid jumps in the resp function
    sele_idx = np.abs(spatial_var) < 5
    spatial_var = spatial_var[sele_idx]
    nuisance_var = nuisance_var[sele_idx]
    while spatial_var.shape[0] < time_points:
        tmpX = rv.rvs(10 ** 4)
        sele_idx = np.abs(tmpX[:, 0]) < 5
        tmpX = tmpX[sele_idx, :]

        spatial_var = np.hstack((spatial_var, tmpX[:, 0]))
        nuisance_var = np.hstack((nuisance_var, tmpX[:, 1]))
    spatial_var = spatial_var[:time_points]
    nuisance_var = nuisance_var[:time_points]

    # create a resp function
    knots = np.hstack(([-5] * 3, np.linspace(-5, 5, 8), [5] * 3))
    beta = np.arange(10)
    beta = beta / np.linalg.norm(beta)
    beta = np.hstack((beta[5:], beta[:5][::-1]))
    resp_func = lambda x: np.dot(gdh.splineDesign(knots, x, order, der=0), beta)

    filter_used_conv = sts.gamma.pdf(np.linspace(0, 20, 100), a=2) - sts.gamma.pdf(np.linspace(0, 20, 100), a=5)
    filter_used_conv = np.hstack((np.zeros(101), filter_used_conv)) * 2
    # mean of the spike counts depending on spatial_var and events
    log_mu0 = resp_func(spatial_var)
    for tr in np.unique(trial_ids):
        log_mu0[trial_ids == tr] = log_mu0[trial_ids == tr] + np.convolve(events[trial_ids == tr], filter_used_conv,
                                                                          mode='same')

    # adjust mean rate
    const = np.log(np.mean(np.exp(log_mu0)) / rate)
    log_mu0 = log_mu0 - const

    # generate spikes
    spk_counts = np.random.poisson(np.exp(log_mu0))

    #fit
    sm_handler = gdh.smooths_handler()
    knots = np.hstack(([-5] * 3, np.linspace(-5, 5, 15), [5] * 3))
    sm_handler.add_smooth('spatial', [spatial_var], knots=[knots], ord=4, is_temporal_kernel=False,
                          trial_idx=trial_ids, is_cyclic=[False], penalty_type='der', der=2)

    sm_handler.add_smooth('nuisance', [nuisance_var], knots=[knots], ord=4, is_temporal_kernel=False,
                          trial_idx=trial_ids, is_cyclic=[False], penalty_type='der', der=2)

    sm_handler.add_smooth('temporal', [events], knots=None, ord=4, is_temporal_kernel=True,
                          trial_idx=trial_ids, is_cyclic=[False], penalty_type='der', der=2,
                          knots_num=10, kernel_length=500, kernel_direction=1)

    # split trial in train and eval
    train_trials = trial_ids % 10 != 0
    eval_trials = ~train_trials

    link = sm.genmod.families.links.log()
    poissFam = sm.genmod.families.family.Poisson(link=link)

    # create the pgam model
    pgam = gl.general_additive_model(sm_handler,
                                  sm_handler.smooths_var,  # list of coovarate we want to include in the model
                                  spk_counts,  # vector of spike counts
                                  poissFam  # poisson family with exponential link from statsmodels.api
                                  )

    # with with all covariate, remove according to stat testing, and then refit
    full, reduced = pgam.fit_full_and_reduced(sm_handler.smooths_var,
                                              th_pval=0.001,  # pval for significance of covariate icluseioon
                                              max_iter=10 ** 2,  # max number of iteration
                                              use_dgcv=True,  # learn the smoothing penalties by dgcv
                                              trial_num_vec=trial_ids,
                                              filter_trials=train_trials)

    exog, idx = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
    xx,tn,sctn,cnt = compute_tuning(spk_counts, full, exog, 'temporal', sm_handler, train_trials, dt=0.006)
    xx2,tn2,sctn2,cnt = compute_tuning(spk_counts, full, exog, 'spatial', sm_handler, train_trials, dt=0.006)
    mi,tun = mutual_info_est(spk_counts, exog, full, 'temporal', sm_handler, train_trials, dt=0.006,bins=20)
    var_zscore_par = {}
    for var in sm_handler.smooths_var:
        var_zscore_par[var] = {}
        var_zscore_par[var]['loc'] = 0
        var_zscore_par[var]['scale'] = 1

    results = postprocess_results(spk_counts,full,reduced,train_trials,sm_handler,poissFam,var_zscore_par=var_zscore_par,info_save={})