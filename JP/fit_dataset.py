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
full,reduced = gam_model.fit_full_and_reduced(sm_handler.smooths_var,th_pval=0.001,
                                              smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
                                              conv_criteria='deviance',
                                              initial_smooths_guess=False,
                                              method='L-BFGS-B',
                                              gcv_sel_tol=10 ** (-13),
                                              use_dgcv=True,
                                              fit_initial_beta=True,
                                              trial_num_vec=trial_idx)



