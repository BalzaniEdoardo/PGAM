import inspect
import os
import sys

thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))

# if scratch exists, assume you are in a singularity container
# with the pgam installed, and the folder with firefly-utils also added
# to the file system.
# else use local
if not os.path.exists("/scratch/jpn5/GAM_Repo/GAM_library/"):
    sys.path.append(os.path.join(os.path.dirname(thisPath), "GAM_library"))
    sys.path.append(
        os.path.join(os.path.dirname(thisPath), "preprocessing_pipeline/util_preproc")
    )
    sys.path.append(os.path.join(os.path.dirname(thisPath), "firefly_utils"))
    is_cluster = False
    job_id = 0
    table_path = "fit_list_table.npy"

else:
    is_cluster = True
    # check where is the firefly_utils folder in the container
    sys.path.append("/<path to firefly_utils>")
    job_id = int(sys.argv[1]) - 1
    table_path = "fit_list_table.npy"

from copy import deepcopy

import dill
import numpy as np
import scipy.stats as sts
from data_handler import *
import gam_data_handlers as gdh
from knots_constructor import knots_cerate
from path_class import get_paths_class
from scipy.io import loadmat
from utils_loading import unpack_preproc_data

from GAM_library import *
import processing_utils as preproc


def pseudo_r2_compute(spk, family, modelX, params):
    lin_pred = np.dot(modelX, params)
    mu = family.fitted(lin_pred)
    res_dev_t = family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t**2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t**2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2


folder_name = deepcopy(thisPath)

tot_fits = 50
plot_res = False
fit_fully_coupled = False
use_k_fold = True
reducedcoupling = False
num_folds = 5
use_fisher_scoring = False

print("folder name")
print(folder_name)
print(" ")
main_dir = os.path.dirname(folder_name)

user_paths = get_paths_class()
if not is_cluster:
    path_to_concat = user_paths.get_path("local_concat")
else:
    path_to_concat = user_paths.get_path("data_hpc")

# select the job to be run
table_fit = np.load(table_path, allow_pickle=True)
table_fit = table_fit[job_id*tot_fits: (job_id+1) * tot_fits]
# parameters to extract
par_list = [
    "Xt",
    "Yt",
    "lfp_beta",
    "lfp_alpha",
    "lfp_theta",
    "var_names",
    "info_trial",
    "trial_idx",
    "brain_area",
    "pre_trial_dur",
    "post_trial_dur",
    "time_bin",
    "cR",
    "presence_rate",
    "isiV",
    "unit_type",
    "channel_id",
    "electrode_id",
    "cluster_id",
]

for row in table_fit:
    path_fit = row['path_fit']
    session = row['session']
    neuron = row['neuron']
    cond_type = row['cond_type']
    cond_value = row['cond_value']

    # open file
    fhName = os.path.join(path_to_concat, session + '.npz')
    dat = np.load(fhName, allow_pickle=True)

    # extract model inputs
    (
        Xt,
        yt,
        lfp_beta,
        lfp_alpha,
        lfp_theta,
        var_names,
        trial_type,
        trial_idx,
        brain_area,
        pre_trial_dur,
        pre_trial_dur,
        time_bin,
        cont_rate_filter,
        presence_rate_filter,
        isi_v_filter,
        unit_type,
        channel_id,
        electrode_id,
        cluster_id,
    ) = unpack_preproc_data(fhName, par_list)
    all_vars = np.hstack((var_names, ['lfp_beta', 'lfp_alpha', 'lfp_theta', 'spike_hist']))
    cond_knots = preproc.get_cond_knots(trial_type)

    # retrieve trials
    train_trials, test_trials = preproc.split_trials(trial_type, cond_type, cond_value)
    keep_train, keep_test, trial_idx_train, trial_idx_test = preproc.get_trial_bool(trial_idx, train_trials, test_trials)

    # initialize vars
    dict_xlims = {}
    sm_handler = gdh.smooths_handler()
    sm_handler_test = gdh.smooths_handler()
    for var in all_vars:
        x_train, x_test, is_cyclic = preproc.process_variable(
            neuron,
            var,
            Xt,
            yt,
            lfp_theta,
            lfp_beta,
            lfp_alpha,
            var_names,
            keep_train,
            keep_test
        )

        knots, x_trans, include_var, is_cyclic, order, \
            kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
            knots_cerate(x_train, var, session, hist_filt_dur='short',
                         exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                         condition=cond_knots)

        x_test = knots_cerate(x_test, var, session,
                              exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'])[1]

        dict_xlims = preproc.get_xlim_dict(var, x_trans, knots, dict_xlims)

        # include variable in handler
        if include_var:
            sm_handler, sm_handler_test = preproc.include_variable(var,
                          x_trans,
                          x_test,
                          sm_handler,
                          sm_handler_test,
                          trial_idx_test,
                          is_temporal_kernel,
                          order,
                          knots,
                          is_cyclic,
                          penalty_type,
                          der,
                          time_bin,
                          trial_idx_train,
                          kernel_len,
                          kernel_direction,
                          )

    neuron_keep = preproc.filter_neurons(cont_rate_filter, unit_type, presence_rate_filter, isi_v_filter, yt)

    for other in neuron_keep:
        if other == neuron:
            continue
        print('adding unit: %d' % other)
        if brain_area[neuron - 1] == brain_area[other - 1]:
            filt_len = 'short'
        else:
            filt_len = 'long'

        tmpy = yt[keep_train, other - 1]
        x = tmpy
        x_test = yt[keep_test, other - 1]

        knots, x_trans, include_var, is_cyclic, order, \
            kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
            knots_cerate(x, 'spike_hist', session, hist_filt_dur=filt_len,
                         exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'])

        x_test = \
            knots_cerate(x_test, 'spike_hist', session, hist_filt_dur=filt_len,
                         exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'])[1]

        var = 'neu_%d' % other
        if include_var:
            sm_handler, sm_handler_test = preproc.include_variable(var,
                                                                   x_trans,
                                                                   x_test,
                                                                   sm_handler,
                                                                   sm_handler_test,
                                                                   trial_idx_test,
                                                                   is_temporal_kernel,
                                                                   order,
                                                                   knots,
                                                                   is_cyclic,
                                                                   penalty_type,
                                                                   der,
                                                                   time_bin,
                                                                   trial_idx_train,
                                                                   kernel_len,
                                                                   kernel_direction,
                                                                   )
