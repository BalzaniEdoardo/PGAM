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
import statsmodels.api as sm
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
sm_handler = smooths_handler()


use_row = 12
row = table[use_row]
neu_id = row['neuron_id']
fhname = '%sgam_preproc_neu%d_%s'%(basefld,neu_id,session)
gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(fhname)

for inputs in construct_knots(gam_raw_inputs, counts, var_names, dict_param,trialCathegory_spatial=True,use50Prior=False):

    varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale = inputs
    if varName in sm_handler.smooths_var and (not varName=='spike_hist'):
        continue
    elif varName in sm_handler.smooths_var and ( varName=='spike_hist'):
        sm_handler.smooths_var.remove('spike_hist')
        sm_handler.smooths_dict.pop('spike_hist')

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

fr2 = np.load('/Users/edoardo/Work/Code/GAM_code/JP/debug/firing_hz_noPrior50.npz',allow_pickle=True)
cov_beta_this = fr2['beta_cov'][use_row]
beta = fr2['beta'][use_row]
X,_ii = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
mu = np.dot(X, beta)
sigma2 = np.einsum('ij,jk,ik->i', X, cov_beta_this, X,
                       optimize=True)
lam_s = np.exp(mu + sigma2 * 0.5)/0.005

xx = np.linspace(0,1,21)
meanRate = np.zeros(20)
meanRateRaw = np.zeros(20)
for k in range(20):
    x0 = xx[k]
    x1 = xx[k + 1]
    meanRate[k] = np.average(lam_s, weights=(sm_handler['subjective_prior']._x[0] >= x0) * (
                sm_handler['subjective_prior']._x[0] < x1))
    meanRateRaw[k] = np.average(counts / 0.005, weights=(sm_handler['subjective_prior']._x[0] >= x0) * (
                sm_handler['subjective_prior']._x[0] < x1))
plt.plot(xx[:-1],meanRateRaw)
plt.plot(xx[:-1],meanRate)

xx = np.linspace(0,1,100)
fX = sm_handler['subjective_prior'].eval_basis([xx]).toarray()
fX = fX[:, :-1] - sm_handler['subjective_prior'].colMean_X
mean_y = np.dot(fX, beta[_ii['subjective_prior']])
old_shape = fX.shape[1]
fX = sm.add_constant(fX)
if old_shape == fX.shape[1]:
    fX = np.hstack((np.ones((fX.shape[0],1)),fX))
index = np.hstack(([0],_ii['subjective_prior']))
se_y = np.sqrt(np.sum(np.dot(fX, cov_beta_this[index, :][:, index]) * fX, axis=1))
plt.figure()
plt.plot(xx*var_zscore_par['subjective_prior']['scale'] + var_zscore_par['subjective_prior']['loc'] , mean_y-1.96*se_y)
plt.plot(xx*var_zscore_par['subjective_prior']['scale'] + var_zscore_par['subjective_prior']['loc'], mean_y+1.96*se_y)
plt.plot(xx*var_zscore_par['subjective_prior']['scale'] + var_zscore_par['subjective_prior']['loc'], mean_y)


unchosen = np.arange(0, np.unique(trial_idx).shape[0])[::10]
choose_trials = np.array(list(set(np.arange(0, np.unique(trial_idx).shape[0])).difference(set(unchosen))),
                         dtype=int)
choose_trials = np.unique(trial_idx)[np.sort(choose_trials)]
filter_trials = np.zeros(trial_idx.shape[0], dtype=bool)
for tr in choose_trials:
    filter_trials[trial_idx == tr] = True

link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)
pr2_eval = pseudo_r2_comp_noFit(counts, sm_handler.smooths_var, beta, sm_handler, family, use_tp=~filter_trials, exog=X)[0]
pr2_train = pseudo_r2_comp_noFit(counts, sm_handler.smooths_var, beta, sm_handler, family, use_tp=filter_trials, exog=X)[0]






