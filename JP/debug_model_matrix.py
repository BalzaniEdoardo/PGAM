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
from scipy.io import savemat
from copy import deepcopy
import matplotlib.pylab as plt

table = np.zeros(1, dtype={'names': ('neuron_id', 'path_file'), 'formats': (int, 'U400')})
table['neuron_id'] = 1
table['path_file'] = 'gam_preproc_neu378_ACAd_NYU-28_2020-10-21_001.mat'
gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(table['path_file'][0])

modelX_stim = np.ones((counts.shape[0],1))
# modelX_stim = np.ones((20001,1))
use_var_stim = ['const']
cc = 0
var_zscore_par = {}
for inputs in construct_knots(gam_raw_inputs,counts, var_names, dict_param):
    varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, mn, std = inputs
    var_zscore_par[varName] = {'mn':mn,'std':std}
    if 'neuron' in varName:
        continue
    if x.sum() == 0:
        print(varName)
        continue


    if ('cL' in varName) or ('cR' in varName) or (varName == 'c0000'):

        use_var_stim += [varName]
        # x = np.zeros(20001)
        # x[np.random.choice(np.arange(1000,19000))] = 1
        # trial_idx = np.ones(20001)
        modelX_stim = np.hstack((modelX_stim, x.reshape(-1, 1)))


        cc += 1000
print('minEig stim: ',np.linalg.eigh(np.dot(modelX_stim.T,modelX_stim))[0].min())


list_add = ['cL100', 'cL025', 'cL012', 'cL006', 'c0000', 'cR006', 'cR012',
       'cR025', 'cR100','choiceL', 'choiceR', 'choice0',
            'prev_choiceR', 'prev_choice0', 'prev_choiceL',
            'prior20' ,'prior80','feedback_correct',
            'feedback_incorrect','moveL', 'moveR',
            'prev_feedback_incorrect','prev_feedback_correct','subjective_prior',
            'movement_PC1', 'movement_PC2',
            'movement_PC3', 'movement_PC4', 'movement_PC5', 'movement_PC6',
            'movement_PC7', 'movement_PC8', 'movement_PC9', 'movement_PC10']

lst_choice = []
for k in range(1,len(list_add)+1):
    modelX_choice_and_prior = np.ones((counts.shape[0], 1))
    use_var_choice_and_prior = ['const']
    var_use = deepcopy(use_var_stim)

    for inputs in construct_knots(gam_raw_inputs,counts, var_names, dict_param):
        varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der,_,_ = inputs
        # print(varName)
        if varName.startswith('choice'):
            lst_choice.append(x)
        if 'neuron' in varName or varName == 'choice0':
            continue

        if x.sum() == 0:
            # print(varName)
            continue
        # if k >10 and varName=='prev_choiceR':
        #     print('ciaooooo',any(list_cond[:k]))

        if varName in list_add[:k]:

            modelX_choice_and_prior = np.hstack((modelX_choice_and_prior, x.reshape(-1, 1)))
            var_use += [varName]

    print('add var', list_add[:k][-1])
    # modelX_all = np.hstack((modelX_stim[:,:-1], modelX_choice_and_prior[:,1:]))
    # modelX_allcntr = np.hstack((modelX_stim[:,:], modelX_choice_and_prior[:,1:]))
    modelX_choice_and_prior[np.isnan(modelX_choice_and_prior)] = 0
    print(k,'eigs',np.linalg.eigh(np.dot(modelX_choice_and_prior[:,:].T,modelX_choice_and_prior[:,:]))[0].min())
    # print(k,'all contrasts',np.linalg.eigh(np.dot(modelX_allcntr[:,:].T,modelX_allcntr[:,:]))[0].min())
    print('\n')



# sm_handler = smooths_handler()
# for inputs in construct_knots(gam_raw_inputs,counts, var_names, dict_param):
#     varName, knots, x, is_cyclic, order, kzernel_len, direction, is_temporal_kernel, penalty_type, der = inputs
#     if varName == 'choice0':
#         continue
#     sm_handler.add_smooth(varName, [x], ord=order, knots=[knots],
#                           is_cyclic=[is_cyclic], lam=50,
#                           penalty_type=penalty_type,
#                           der=der,
#                           trial_idx=trial_idx, time_bin=0.001,
#                           is_temporal_kernel=is_temporal_kernel,
#                           kernel_length=kernel_len,
#                           kernel_direction=direction)
#     # mdl = np.hs tack((mdl, x.reshape(-1,1)))
#
# Xbasis, idx = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
