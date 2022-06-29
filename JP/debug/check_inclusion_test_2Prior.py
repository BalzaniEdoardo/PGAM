import numpy as np
import os,sys
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/JP')
from processing_tools import *
from create_basis import construct_knots, dict_param
from parsing_tools import parse_mat, parse_fit_list
import sys, inspect, os
import matplotlib.pylab as plt
import traceback
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir, 'GAM_library'))

from GAM_library import GAM_result, general_additive_model
from gam_data_handlers import smooths_handler
from der_wrt_smoothing import deriv3_link, d2variance_family
import statsmodels as sm
from processing_tools import pseudo_r2_comp, postprocess_results
from scipy.integrate import simps
from scipy.io import savemat,loadmat
import re


basefld = '/Volumes/Balsip HD/ASD-MOUSE/CA3/'
session = 'CA3_CSP003_2019-11-20_002.mat'
splits = session.split('_')
area = splits[0]
subject = splits[1]
date = splits[2]
sess_num = splits[3]

table = parse_fit_list('/Users/edoardo/Work/Code/GAM_code/JP/list_to_fit_GAM.mat')
paths = table['path_file']
func = lambda string : string.split('\\')[-1]
vec_func = np.vectorize(func)
session_list = vec_func(paths)
sel = (session_list == session) * (table['use_coupling']==False)
use_coupling = False
table = table[sel]
use_subjectivePrior = True
var_zscore_par = {}

neu_fit = []
pval_prior = []
firing_hz = []
cov_sign = []
beta_vec = []
idx_vec = []
beta_cov = []
for row in table:
    neu_id = row['neuron_id']
    fhname = '%sgam_preproc_neu%d_%s'%(basefld,neu_id,session)
    gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(fhname)
    sm_handler = smooths_handler()
    for inputs in construct_knots(gam_raw_inputs, counts, var_names, dict_param,trialCathegory_spatial=True,use50Prior=False):
        varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale = inputs
        var_zscore_par[varName] = {'loc': loc, 'scale': scale}
        # if varName != 'spike_hist':
        #     continue
        if (not use_coupling) and (varName.startswith('neuron_')):
            continue
        if (not use_subjectivePrior) and (varName == 'subjective_prior'):
            continue
        if x.sum() == 0:
            print('\n\n', varName, ' is empty! all values are 0s.')
            continue
        if all(np.isnan(x)):
            print('\n\n', varName, ' is all nans!.')
            continue
        if varName == 'prior50':
            continue

        print('adding', varName)
        sm_handler.add_smooth(varName, [x], ord=order, knots=[knots],
                              is_cyclic=[is_cyclic], lam=50,
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx, time_bin=0.005,
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len,
                              kernel_direction=direction)

        # if not is_temporal_kernel:
        #     xx = np.unique(x[~np.isnan(x)])
        #     bX = sm_handler[varName].eval_basis([xx]).toarray()
        #     plt.figure()
        #     plt.title(varName)
        #     for k in range(bX.shape[1]):
        #         plt.plot(xx, bX[:, k])

    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)

    # sel_num = int(np.unique(trial_idx).shape[0]*0.9)
    unchosen = np.arange(0, np.unique(trial_idx).shape[0])[::10]
    choose_trials = np.array(list(set(np.arange(0, np.unique(trial_idx).shape[0])).difference(set(unchosen))),
                             dtype=int)
    choose_trials = np.unique(trial_idx)[np.sort(choose_trials)]
    filter_trials = np.zeros(trial_idx.shape[0], dtype=bool)
    for tr in choose_trials:
        filter_trials[trial_idx == tr] = True

    # X, index = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)

    gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, counts, poissFam, fisher_scoring=False)

    full_fit, reduced_fit = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001,
                                                           smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
                                                           conv_criteria='deviance',
                                                           initial_smooths_guess=False,
                                                           method='L-BFGS-B',
                                                           gcv_sel_tol=10 ** (-10),
                                                           use_dgcv=True,
                                                           fit_initial_beta=True,
                                                           trial_num_vec=trial_idx,
                                                           filter_trials=filter_trials)

    X, _ii = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
    mu = np.dot(X, full_fit.beta)
    sigma2 = np.einsum('ij,jk,ik->i', X, full_fit.cov_beta, X,
                       optimize=True)
    lam_s = np.exp(mu + sigma2 * 0.5)/0.005
    rate_prior = np.zeros(3)
    prval = [20, 80]
    for k in range(2):
        rate_prior[k] = np.average(lam_s, weights=sm_handler['prior20']._x[0] == prval[k])

    neu_fit.append(neu_id)
    sel = full_fit.covariate_significance['covariate'] == 'prior20'
    pval_prior.append(full_fit.covariate_significance['p-val'][sel])
    firing_hz.append(rate_prior)
    cov_sign.append(full_fit.covariate_significance)
    beta_vec.append(full_fit.beta)
    beta_cov.append(full_fit.cov_beta)
    idx_vec.append(full_fit.index_dict)
    np.savez('firing_hz_noPrior50.npz',pval_prior=pval_prior, neu_fit=neu_fit,firing_hz=firing_hz,cov_sign=cov_sign,
             beta=beta_vec,idx_dict=idx_vec,beta_cov=beta_cov)
