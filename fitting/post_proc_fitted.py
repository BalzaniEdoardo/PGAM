import inspect
import os
import sys
import statsmodels.api as sm
from pathlib import Path
from scipy.io import savemat

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
    table_path = "/Users/ebalzani/Code/Demo_PGAM/fit_list.npy"
    save_path = Path("/Users/ebalzani/Code/Demo_PGAM/results/m44s187/")
    from path_class import get_paths_class

else:
    is_cluster = True
    # check where is the firefly_utils folder in the container
    # sys.path.append("/<path to f:refly_utils>")
    job_id = int(sys.argv[1]) - 1
    table_path = "fit_list.npy"
    save_path = Path("/scratch/eb162/postproc_gam_to_matlab/matlab_processed/")
save_path.mkdir(parents=True, exist_ok=True)
from copy import deepcopy

import dill
import numpy as np
import scipy.stats as sts
from data_handler import *
import gam_data_handlers as gdh
from knots_constructor import knots_cerate
#
from scipy.io import loadmat
from utils_loading import unpack_preproc_data

from GAM_library import *
import processing_utility as preproc
import post_processing as postproc



folder_name = deepcopy(thisPath)

tot_fits = 27
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

if not is_cluster:
    user_paths = get_paths_class()
    path_to_concat = user_paths.get_path("local_concat")
else:
    path_to_concat = "/scratch/eb162/dataset_firefly/" 

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
    path_fit = row['path']
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
    
    # load fits
    with open(path_fit, 'rb') as fh:
        fit_dict = dill.load(fh)
    full = fit_dict['full']
    reduced = fit_dict['reduced']
    
    all_vars = full.var_list

    cond_knots = preproc.get_cond_knots(trial_type)

    # retrieve trials
    train_trials, test_trials = preproc.split_trials(trial_type, cond_type, cond_value)
    keep_train, keep_test, trial_idx_train, trial_idx_test = preproc.get_trial_bool(trial_idx, train_trials, test_trials)
    bool_train = np.zeros((Xt.shape[0], ), dtype=bool)
    bool_train[keep_train] = True

    # initialize vars
    dict_xlims = {}
    sm_handler = gdh.smooths_handler()
    sm_handler_test = gdh.smooths_handler()
    for var in all_vars:

#         if var == 'hand_vel1' or var == 'hand_vel2':
#             continue
#         if (cond_type != 'ptb') and (var == 't_ptb'):
#             continue
#         if var == 'rad_path_from_xy':
#             continue
        if var.startswith("neu_"):
            continue

        print(f"adding {var}...")
        x_var, is_cyclic = preproc.process_variable_without_split(neuron,
            var,
            Xt,
            yt,
            lfp_theta,
            lfp_beta,
            lfp_alpha,
            var_names
        )

        knots, x_trans, include_var, is_cyclic, order, \
            kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
            knots_cerate(x_var, var, session, hist_filt_dur='short',
                         exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                         condition=cond_knots)

        dict_xlims = preproc.get_xlim_dict(var, x_trans, knots, dict_xlims)

        # include variable in handler
        if include_var:
            sm_handler = preproc.include_variable_without_split(
                var,
                x_trans,
                sm_handler,
                is_temporal_kernel,
                order,
                knots,
                is_cyclic,
                penalty_type,
                der,
                time_bin,
                kernel_len,
                kernel_direction,
                trial_idx
            )

    neuron_keep = [int(var.split('_')[1]) for var in all_vars if var.startswith("neu_")]
    #neuron_keep = preproc.filter_neurons(cont_rate_filter, unit_type, presence_rate_filter, isi_v_filter, yt)

    for other in neuron_keep:
        if other == neuron:
            continue
        print('adding unit: %d' % other)
        if brain_area[neuron - 1] == brain_area[other - 1]:
            filt_len = 'short'
        else:
            filt_len = 'long'

        tmpy = yt[:, other - 1]
        x = tmpy

        knots, x_trans, include_var, is_cyclic, order, \
            kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
            knots_cerate(x, 'spike_hist', session, hist_filt_dur=filt_len,
                         exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'])

        #knots = full.smooth_info[f"neu_{other}"]['knots'].flatten()
        var = 'neu_%d' % other
        if include_var:
            sm_handler = preproc.include_variable_without_split(
                var,
                x_trans,
                sm_handler,
                is_temporal_kernel,
                order,
                knots,
                is_cyclic,
                penalty_type,
                der,
                time_bin,
                kernel_len,
                kernel_direction,
                trial_idx
            )


    if set(full.var_list) != set(sm_handler.smooths_var):
        var_diff = set(sm_handler.smooths_var).union(full.var_list) -\
                   set(full.var_list).intersection(sm_handler.smooths_var)
        var_diff = np.array(var_diff)
        np.save(save_path / 'error_ff_postproc_%s_%s_%.4f_c%d.npy'%(session, cond_type, cond_value, neuron), var_diff)
        raise ValueError("Inconsistent predictors!")

    poissFam = sm.genmod.families.family.Poisson(link=sm.genmod.families.links.log())
    info_save = {
        "session": session,
        "neuron": neuron,
        "cond_type": cond_type,
        "cond_value": cond_value,
        "unit_type": unit_type[neuron-1],
        "channel_id": channel_id[neuron-1],
        "electrode_id": electrode_id[neuron-1],
        "cluster_id": cluster_id[neuron-1],
    }

    results = postproc.postprocess_results(
        neuron,
        yt[:, neuron-1],
        full, reduced,
        bool_train,
        sm_handler,
        poissFam,
        trial_idx,
        var_zscore_par=None,
        info_save=info_save,
        bins=15
    )

    fhName = '%s/ff_postproc_%s_%s_%.4f_c%d.mat'%(session, session, cond_type, cond_value, neuron)
    savemat(save_path / fhName, mdict={"pgam_results": results})
    
