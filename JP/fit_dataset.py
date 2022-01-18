from create_basis import construct_knots, dict_param
from parsing_tools import parse_mat
import sys, inspect, os

basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir, 'GAM_library'))

from GAM_library import GAM_result, general_additive_model
from gam_data_handlers import smooths_handler
import numpy as np
from der_wrt_smoothing import deriv3_link, d2variance_family
import statsmodels as sm
from processing_tools import pseudo_r2_comp
from scipy.integrate import simps

try:
    job_id = int(sys.argv[1]) - 1
    table = np.load('fit_list.npy')

except:
    table = np.zeros(1, dtype={'names':('neuron_id', 'path_file'), 'formats':(int, 'U400')})
    table['neuron_id'] = 1
    table['path_file'] = 'gam_preproc_ACAd_NYU-28_2020-10-21_001.mat'
    job_id = 0

# extract input
gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(table['path_file'][job_id])
sm_handler = smooths_handler()
for inputs in construct_knots(gam_raw_inputs, var_names, dict_param):
    varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der = inputs
    sm_handler.add_smooth(varName, [x], ord=order, knots=[knots],
                              is_cyclic=[is_cyclic], lam=50,
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx, time_bin=0.001,
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len,
                              kernel_direction=direction)

link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,counts,poissFam,fisher_scoring=False)
full_fit,reduced_fit = gam_model.fit_full_and_reduced(sm_handler.smooths_var,th_pval=0.001,
                                              smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
                                              conv_criteria='deviance',
                                              initial_smooths_guess=False,
                                              method='L-BFGS-B',
                                              gcv_sel_tol=10 ** (-13),
                                              use_dgcv=True,
                                              fit_initial_beta=True,
                                              trial_num_vec=trial_idx)

session = 'unknown'
trial_type = 'all'
neuNum = table['neuron_id'][job_id]
dtype_dict = {'names': (
                    'session', 'trial_type', 'neuron', 'full_pseudo_r2_train', 'full_pseudo_r2_eval',
                    'reduced_pseudo_r2_train', 'reduced_pseudo_r2_eval', 'variable', 'pval', 'mutual_info', 'x',
                    'model_rate_Hz', 'raw_rate_Hz', 'kernel_strength', 'signed_kernel_strength', 'kernel_x',
                    'kernel', 'kernel_mCI', 'kernel_pCI'),
              'formats': (
              'U30', 'U30', int, float, float, float, float, 'U30', float, float, object, object, object, float, float,
              object,
              object, object, object)
              }
results = np.zeros(len((full_fit.var_list)), dtype=dtype_dict)
cs_table = full_fit.covariate_significance
for cc in range(len(full_fit.var_list)):
    var = full_fit.var_list[cc]
    cs_var = cs_table[cs_table['covariate'] == var]
    results['session'][cc] = session
    results['neuron'][cc] = neuNum
    results['variable'][cc] = var
    results['trial_type'][cc] = trial_type
    results['full_pseudo_r2_train'][cc] = full_fit.pseudo_r2
    results['full_pseudo_r2_eval'][cc] = pseudo_r2_comp(counts, full_fit, sm_handler,
                                                        use_tp=~(filter_trials))
    results['reduced_pseudo_r2_train'][cc] = reduced_fit.pseudo_r2
    results['reduced_pseudo_r2_eval'][cc] = pseudo_r2_comp(counts, reduced_fit, sm_handler,
                                                           use_tp=~(filter_trials))
    results['pval'][cc] = cs_var['p-val']

    results['pval'][cc] = cs_var['p-val']
    if var in full_fit.mutual_info.keys():
        results['mutual_info'][cc] = full_fit.mutual_info[var]
    else:
        results['mutual_info'][cc] = np.nan
    if var in full_fit.tuning_Hz.__dict__.keys():
        results['x'][cc] = full_fit.tuning_Hz.__dict__[var].x
        results['model_rate_Hz'][cc] = full_fit.tuning_Hz.__dict__[var].y_model
        results['raw_rate_Hz'][cc] = full_fit.tuning_Hz.__dict__[var].y_raw

    # compute kernel strength
    if full_fit.smooth_info[var]['is_temporal_kernel']:
        dim_kern = full_fit.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = full_fit.smooth_info[var]['knots'][0].shape[0]
        x = np.zeros(dim_kern)
        x[(dim_kern - 1) // 2] = 1
        xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
        fX, fminus, fplus = full_fit.smooth_compute([x], var, 0.99)
        if (var == 'spike_hist') or ('neu_') in var:
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
    results['kernel_x'][cc] = xx2
