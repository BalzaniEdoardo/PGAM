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
from processing_tools import pseudo_r2_comp, postprocess_results
from scipy.integrate import simps
from scipy.io import savemat, loadmat
from scipy.interpolate import interp1d
from plotting_tools import plot_kernels,plot_rateHz

# load fit kernel and input
res = loadmat('spatial_firstRes_NYU-28_2020-10-21_001_u378.mat')['results']
gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat('gam_preproc_neu378_ACAd_NYU-28_2020-10-21_001.mat')

logmu = np.zeros(counts.shape[0])
sm_handler = smooths_handler()
save_kerns = {}

cnt = 1
for k in range(res.shape[1]):
    if res[0,k]['variable'] == 'spike_hist':
        continue
    cnt += res[0,k]['beta_full'][0].shape[0]
beta_tmp = np.zeros(cnt)
beta_tmp[0] = res[0,0]['intercept_full']

coeff_ind = 1
for inputs in construct_knots(gam_raw_inputs,counts, var_names, dict_param):
    varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der = inputs


    if varName == 'c0000' or varName == 'prior50' or 'neuron' in varName or varName=='spike_hist':
        continue
    found = False
    for row in res[0]:
        if row['variable'] == varName:
            found = True
            break
    print(varName)
    if not found:
        raise ValueError('could not find variable %s'%varName)

    sm_handler.add_smooth(varName, [x], ord=order, knots=[knots],
                          is_cyclic=[is_cyclic], lam=50,
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx, time_bin=0.005,
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len,
                          kernel_direction=direction)

    pval = row['pval']
    if pval < 0.001:
        beta = row['beta_full'][0]
        beta_tmp[coeff_ind:coeff_ind+beta.shape[0]] = beta

        # kernel = row['kernel'][0]
    else:
        # betasize = row['beta_full'][0].shape[0]
        # beta = np.random.normal(size=betasize)
        beta = row['beta_full'][0]
        beta_tmp[coeff_ind:coeff_ind + beta.shape[0]] = beta
        # kernel = 0.1*res[0,10]['kernel'][0]# + 0.1*res[0,10]['kernel'][0].mean() * np.random.normal(size=res[0,10]['kernel'][0].shape[0])


    if is_temporal_kernel:
        save_kerns[varName] = {'x':row['kernel_x'][0],'y':np.dot(sm_handler[varName].basis_kernel.toarray()[:,:-1],beta)}

    else:
        xx = np.linspace(knots[0], knots[-1], 100)
        sm_handler_tmp = smooths_handler()
        sm_handler_tmp.add_smooth(varName, [xx], ord=order, knots=[knots],
                          is_cyclic=[is_cyclic], lam=50,
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx, time_bin=0.005,
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len,
                          kernel_direction=direction)
        X,iidx = sm_handler_tmp.get_exog_mat([varName])
        fun = interp1d(np.linspace(knots[0], knots[-1], xx.shape[0]), np.dot(X[:, iidx[varName]],beta))
        save_kerns[varName] = {'x':xx,'y':fun(xx)}

    coeff_ind += beta.shape[0]
    # if any(np.isnan(logmu)):
    #     break
X = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)[0]
trueMean = counts.mean()
logmu = np.dot(X,beta_tmp)
spk = np.random.poisson(np.exp(logmu))


link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)


filter_trials = np.ones(trial_idx.shape[0], dtype=bool)


X, index = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,spk, poissFam,fisher_scoring=False)
full_fit,reduced_fit = gam_model.fit_full_and_reduced(sm_handler.smooths_var,th_pval=0.001,
                                              smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
                                              conv_criteria='deviance',
                                              initial_smooths_guess=False,
                                              method='L-BFGS-B',
                                              gcv_sel_tol=10 ** (-10),
                                              use_dgcv=True,
                                              fit_initial_beta=True,
                                              trial_num_vec=trial_idx,
                                              filter_trials=filter_trials)

results = postprocess_results(spk, full_fit,reduced_fit, 'syntetic', 0, filter_trials,
                        sm_handler, family)


dict_ax = plot_kernels(results, dict_ax={}, ncols=6)
for key in dict_ax.keys():
    ax = dict_ax[key]
    ker = results[results['variable']==key]['kernel'][0]
    yy = save_kerns[key]['y'] + np.median(ker) - np.median(save_kerns[key]['y'])
    ax.plot(save_kerns[key]['x'],yy)

