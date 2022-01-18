from create_basis import construct_knots, dict_param
from parsing_tools import parse_mat
import sys, inspect, os

basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
os.path.append(os.path.join(basedir, 'GAM_library'))

from GAM_library import GAM_result, general_additive_model
from gam_data_handlers import smooths_handler
import numpy as np


try:
    job_id = int(sys.argv[1]) - 1
    table = np.load('fit_list.npy')

except:
    table = np.zeros(1, dtype={'names':('neuron_id', 'path_file'), 'formats':(int, 'U400')})
    table['neuron_id'] = 1
    table['path_file'] = 'gam_preproc_ACAd_NYU-28_2020-10-21_001.mat'
    job_id = 0

gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(table['path_file'][job_id])
sm_handler = smooths_handler()
for inputs in construct_knots(gam_raw_inputs, var_names, dict_param):
    varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der = inputs
    sm_handler.add_smooth()






