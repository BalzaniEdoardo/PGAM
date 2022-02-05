import numpy as np
from copy import deepcopy
from scipy.integrate import simps
import matplotlib.pylab as plt

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


def compute_tuning(spk, fit, exog, var, sm_handler, filter_trials, dt=0.006):
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
        if size_kern % 2 == 0:
            size_kern += 1
        half_size = (size_kern - 1) // 2
        timept = np.arange(-half_size, half_size + 1) * fit.time_bin

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
            time_kernel[ind - half_size:ind + half_size + 1] = timept

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

        tuning = np.zeros(vel_bins.shape[0] - 1)
        var_tuning = np.zeros(vel_bins.shape[0] - 1)
        sc_based_tuning = np.zeros(vel_bins.shape[0] - 1)
        tot_s_vec = np.zeros(vel_bins.shape[0] - 1)
        x_axis = 0.5 * (vel_bins[:-1] + vel_bins[1:])

        cc = 0

        for v0 in vel_bins[:-1]:
            idx = (vels >= v0) * (vels < v0 + dv)
            tuning[cc] = np.nanmean(lam_s[idx])
            var_tuning[cc] = np.nanpercentile(sigm2_s[idx], 90)
            sc_based_tuning[cc] = spk[filter_trials][idx].mean()
            tot_s_vec[cc] = np.sum(idx)

            cc += 1
    return x_axis, tuning / dt, sc_based_tuning / dt


def postprocess_results(counts, full_fit,reduced_fit, info_save, filter_trials,
                        sm_handler, family,var_zscore_par, use_coupling, use_subjectivePrior):

    dtype_dict = {'names': (
        'brain_area_group','animal_name','date','session_num', 'neuron_id', 'brain_region','brain_region_id',
        'fr','amp','depth','x','y','z',
        'full_pseudo_r2_train', 'full_pseudo_r2_eval',
        'reduced_pseudo_r2_train', 'reduced_pseudo_r2_eval', 'variable', 'pval', 'reduced_pval', 'mutual_info', 'x_rate_Hz',
        'model_rate_Hz', 'raw_rate_Hz','reduced_x_rate_Hz','reduced_model_rate_Hz', 'reduced_raw_rate_Hz', 'kernel_strength', 'signed_kernel_strength','reduced_kernel_strength', 'reduced_signed_kernel_strength', 'kernel_x',
        'kernel', 'kernel_mCI', 'kernel_pCI','reduced_kernel_x','reduced_kernel', 'reduced_kernel_mCI', 'reduced_kernel_pCI','beta_full','beta_reduced','intercept_full','intercept_reduced','use_coupling',
        'use_subjectivePrior'),
        'formats': ('U30','U15','U15','U3',int,'U30',int,float,float,float,float,float,float,float,float,float,float,'U40',
                    float,float,float,object,object,object,object,object,object,float,float,float,float,object,object,object,object,object,object,object,object,object,object,float,float,bool,bool)
    }
    results = np.zeros(len((full_fit.var_list)), dtype=dtype_dict)
    cs_table = full_fit.covariate_significance
    cs_table_red = reduced_fit.covariate_significance
    exog_full = None
    exog_reduced = None
    for cc in range(len(full_fit.var_list)):
        var = full_fit.var_list[cc]
        print('processing: ',var)
        cs_var = cs_table[cs_table['covariate'] == var]
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                cs_var_red = cs_table_red[cs_table_red['covariate'] == var]

        results['brain_area_group'][cc] = info_save['brain_area_group']
        results['animal_name'][cc] = info_save['animal_name']
        results['date'][cc] = info_save['date']
        results['session_num'][cc] = info_save['session_num']
        results['neuron_id'][cc] = info_save['neuron_id']
        results['brain_region'][cc] = info_save['brain_region']
        results['brain_region_id'][cc] = info_save['brain_region_id']
        results['use_coupling'][cc] = use_coupling
        results['use_subjectivePrior'][cc] = use_subjectivePrior

        results['fr'][cc] = info_save['fr']
        results['amp'][cc] = info_save['amp']
        results['depth'][cc] = info_save['depth']
        results['x'][cc] = info_save['x']
        results['y'][cc] = info_save['y']
        results['z'][cc] = info_save['z']


        results['variable'][cc] = var
        # results['trial_type'][cc] = trial_type
        results['full_pseudo_r2_train'][cc] = full_fit.pseudo_r2
        results['full_pseudo_r2_eval'][cc],exog_full = pseudo_r2_comp(counts, full_fit, sm_handler, family,
                                                            use_tp=~(filter_trials),exog=exog_full)
        if not reduced_fit is None:
            results['reduced_pseudo_r2_train'][cc] = reduced_fit.pseudo_r2
        results['reduced_pseudo_r2_eval'][cc],exog_reduced = pseudo_r2_comp(counts, reduced_fit, sm_handler, family,
                                                               use_tp=~(filter_trials),exog=exog_reduced)
        results['pval'][cc] = cs_var['p-val']
        if not reduced_fit is None:
            if var in reduced_fit.var_list:
                results['reduced_pval'][cc] = cs_var_red['p-val']
            else:
                results['reduced_pval'][cc] = np.nan

        if var in full_fit.mutual_info.keys():
            results['mutual_info'][cc] = full_fit.mutual_info[var]
        else:
            results['mutual_info'][cc] = np.nan
        if var in full_fit.tuning_Hz.__dict__.keys():
            if ~np.isnan(var_zscore_par[var]['loc']):
                xx = full_fit.tuning_Hz.__dict__[var].x * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
            else:
                xx = full_fit.tuning_Hz.__dict__[var].x
            if (full_fit.smooth_info[var]['kernel_direction'] == 1) and (full_fit.smooth_info[var]['is_temporal_kernel']):
                sel = xx > 0
            elif (full_fit.smooth_info[var]['kernel_direction'] == -1) and (full_fit.smooth_info[var]['is_temporal_kernel']):
                sel = xx < 0
            else:
                sel = np.ones(xx.shape,dtype=bool)
            results['x_rate_Hz'][cc] = xx[sel]
            results['model_rate_Hz'][cc] = full_fit.tuning_Hz.__dict__[var].y_model[sel]
            results['raw_rate_Hz'][cc] = full_fit.tuning_Hz.__dict__[var].y_raw[sel]

        # compute kernel strength
        if full_fit.smooth_info[var]['is_temporal_kernel']:
            dim_kern = full_fit.smooth_info[var]['basis_kernel'].shape[0]
            knots_num = full_fit.smooth_info[var]['knots'][0].shape[0]
            x = np.zeros(dim_kern)
            x[(dim_kern - 1) // 2] = 1
            xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
            fX, fminus, fplus = full_fit.smooth_compute([x], var, 0.99)
            if (var == 'spike_hist') or ('neuron_') in var:
                fminus = fminus[(dim_kern - 1) // 2:] - fX[0]
                fplus = fplus[(dim_kern - 1) // 2:] - fX[0]
                fX = fX[(dim_kern - 1) // 2:] - fX[0]
                xx2 = xx2[(dim_kern - 1) // 2:]
            else:
                fplus = fplus - fX[-1]
                fminus = fminus - fX[-1]
                fX = fX - fX[-1]

            results['kernel_strength'][cc] = simps(fX ** 2, dx=0.006) / (0.006 * fX.shape[0])
            results['signed_kernel_strength'][cc] = simps(fX, dx=0.006) / (0.006 * fX.shape[0])

        else:
            knots = full_fit.smooth_info[var]['knots']
            xmin = knots[0].min()
            xmax = knots[0].max()
            func = lambda x: (full_fit.smooth_compute([x], var, 0.99)[0] -
                              full_fit.smooth_compute([x], var, 0.95)[0].mean()) ** 2
            xx = np.linspace(xmin, xmax, 500)
            xx2 = np.linspace(xmin, xmax, 100)
            dx = xx[1] - xx[0]
            fX, fminus, fplus = full_fit.smooth_compute([xx2], var, 0.99)
            results['kernel_strength'][cc] = simps(func(xx), dx=dx) / (xmax - xmin)
        results['kernel'][cc] = fX
        results['kernel_pCI'][cc] = fplus
        results['kernel_mCI'][cc] = fminus


        if ~np.isnan(var_zscore_par[var]['loc']):
            xx2 = xx2 * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
        results['kernel_x'][cc] = xx2
        results['beta_full'][cc] = full_fit.beta[full_fit.index_dict[var]]
        results['intercept_full'][cc] = full_fit.beta[0]

        if not reduced_fit is None:
            results['intercept_reduced'][cc] = reduced_fit.beta[0]
            if var in reduced_fit.var_list:
                results['beta_reduced'][cc] = reduced_fit.beta[reduced_fit.index_dict[var]]

                # compute kernel strength
                if reduced_fit.smooth_info[var]['is_temporal_kernel']:
                    dim_kern = reduced_fit.smooth_info[var]['basis_kernel'].shape[0]
                    knots_num = reduced_fit.smooth_info[var]['knots'][0].shape[0]
                    x = np.zeros(dim_kern)
                    x[(dim_kern - 1) // 2] = 1
                    xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
                    fX, fminus, fplus = reduced_fit.smooth_compute([x], var, 0.99)
                    if (var == 'spike_hist') or ('neuron_') in var:
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
                                      reduced_fit.smooth_compute([x], var, 0.95)[0].mean()) ** 2
                    xx = np.linspace(xmin, xmax, 500)
                    xx2 = np.linspace(xmin, xmax, 100)
                    dx = xx[1] - xx[0]
                    fX, fminus, fplus = reduced_fit.smooth_compute([xx2], var, 0.99)
                    results['reduced_kernel_strength'][cc] = simps(func(xx), dx=dx) / (xmax - xmin)
                results['reduced_kernel'][cc] = fX
                results['reduced_kernel_pCI'][cc] = fplus
                results['reduced_kernel_mCI'][cc] = fminus

            if ~np.isnan(var_zscore_par[var]['loc']):
                xx2 = xx2 * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
            results['reduced_kernel_x'][cc] = xx2
            # results['beta_full'][cc] = full_fit.beta[full_fit.index_dict[var]]
            # results['intercept_full'][cc] = full_fit.beta[0]
            # results['intercept_reduced'][cc] = reduced_fit.beta[0]
        if not reduced_fit is None:
            if var in reduced_fit.tuning_Hz.__dict__.keys():
                if ~np.isnan(var_zscore_par[var]['loc']):
                    xx = reduced_fit.tuning_Hz.__dict__[var].x * var_zscore_par[var]['scale'] + var_zscore_par[var]['loc']
                else:
                    xx = reduced_fit.tuning_Hz.__dict__[var].x
                if (reduced_fit.smooth_info[var]['kernel_direction'] == 1) and (reduced_fit.smooth_info[var]['is_temporal_kernel']):
                    sel = xx > 0
                elif (reduced_fit.smooth_info[var]['kernel_direction'] == -1) and (reduced_fit.smooth_info[var]['is_temporal_kernel']):
                    sel = xx < 0
                else:
                    sel = np.ones(xx.shape,dtype=bool)
                results['reduced_x_rate_Hz'][cc] = xx[sel]
                results['reduced_model_rate_Hz'][cc] = reduced_fit.tuning_Hz.__dict__[var].y_model[sel]
                results['reduced_raw_rate_Hz'][cc] = reduced_fit.tuning_Hz.__dict__[var].y_raw[sel]


    return results