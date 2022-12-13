#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:51:08 2022

@author: edoardo
"""


# import libs
import numpy as np
import sys,os
# apped path to GAM_library if not in the envs (not needed if working within a Docker container or 
# if GAM_library is in the PATH or PYTHONPATH environment variables)
sys.path.append('../')

import GAM_library as gamlib
import gam_data_handlers as gdh
from post_processing import postprocess_results
import yaml
import statsmodels.api as sm
from scipy.io import savemat


np.random.seed(4)


def fit_from_config(fit_num, path_fit_list, frac_eval=0.2, save_as_mat=False):
    # load fit info
    with open(path_fit_list, 'r') as stream:
        fit_dict = yaml.safe_load(stream)

    # unpack the info and load the data
    experiment_ID = fit_dict['experiment_ID'][fit_num]
    session_ID = fit_dict['session_ID'][fit_num]
    neuron_num = fit_dict['neuron_num'][fit_num]
    path_to_input = fit_dict['path_to_input'][fit_num]
    path_to_config = fit_dict['path_to_config'][fit_num]
    path_out = fit_dict['path_to_output'][fit_num]


    print('FIT INFO:\nEXP ID: %s\nSESSION ID: %s\nNEURON NUM: %d\nINPUT DATA PATH: %s\nCONFIG PATH: %s\n\n'%(
        experiment_ID,session_ID,neuron_num+1,path_to_input,path_to_config))

    # load & unpack data and config
    data = np.load(path_to_input, allow_pickle=True)
    counts = data['counts']
    variables = data['variables']
    variable_names = data['variable_names']
    neu_names = data['neu_names']
    trial_ids = data['trial_ids']
    if 'neu_info' in data.keys():
        neu_info = data['neu_info'].all()
    else:
        neu_info = {}

    with open(path_to_config, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # create a train and eval set (approximately with the right frac of trials)
    train_trials = trial_ids % (np.round(1/frac_eval)) != 0
    eval_trials = ~train_trials

    # create and populate the smooth handler object
    sm_handler = gdh.smooths_handler()
    for var in config_dict.keys():
        print('adding %s...'%var)
        # check if var is a neuron or a variable
        if var in variable_names:
            x_var = np.squeeze(variables[:, np.array(variable_names) == var])
        elif var in neu_names:
            x_var = np.squeeze(counts[:, np.array(neu_names) == var])
        else:
            raise ValueError('Variable "%s" not found in the input data!'%var)

        knots = config_dict[var]['knots']

        if np.isscalar(knots):
            knots = None
        else:
            knots = [np.array(knots)]

        lam = config_dict[var]['lam']
        penalty_type = config_dict[var]['penalty_type']
        der = config_dict[var]['der']
        order = config_dict[var]['order']
        is_temporal_kernel = config_dict[var]['is_temporal_kernel']
        is_cyclic =  config_dict[var]['is_cyclic']
        knots_num = config_dict[var]['knots_num']
        kernel_length = config_dict[var]['kernel_length']
        kernel_direction = config_dict[var]['kernel_direction']
        samp_period = config_dict[var]['samp_period']

        # rename the variable as spike hist if the input is the spike counts of the neuron we are fitting
        if var == neu_names[neuron_num]:
            label = 'spike_hist'
        else:
            label = var

        sm_handler.add_smooth(label, [x_var], knots=knots, ord=order, is_temporal_kernel=is_temporal_kernel,
                         trial_idx=trial_ids, is_cyclic=is_cyclic, penalty_type=penalty_type, der=der, lam=lam,
                             knots_num=knots_num, kernel_length=kernel_length,kernel_direction=kernel_direction,
                             time_bin=samp_period)

    link = sm.genmod.families.links.log()
    poissFam = sm.genmod.families.family.Poisson(link=link)

    spk_counts = np.squeeze(counts[:, neuron_num])

    # create the pgam model
    pgam = gamlib.general_additive_model(sm_handler,
                                  sm_handler.smooths_var, # list of coovarate we want to include in the model
                                  spk_counts, # vector of spike counts
                                  poissFam # poisson family with exponential link from statsmodels.api
                                 )

    print('\nfitting neuron %s...\n'%neu_names[neuron_num])
    full, reduced = pgam.fit_full_and_reduced(sm_handler.smooths_var,
                                              th_pval=0.001,# pval for significance of covariate icluseioon
                                              max_iter=10 ** 2, # max number of iteration
                                              use_dgcv=True, # learn the smoothing penalties by dgcv
                                              trial_num_vec=trial_ids,
                                              filter_trials=train_trials)

    print('post-process fit results...')
    res = postprocess_results(neu_names[neuron_num], spk_counts, full, reduced, train_trials,
                            sm_handler, poissFam, trial_ids, var_zscore_par=None, info_save=neu_info, bins=100)


    # saving the file: save_name will be expID_sessionID_neuID_configName

    config_basename = os.path.basename(path_to_config).split('.')[0]
    save_name = '%s_%s_%s_%s'%(experiment_ID, session_ID, neu_names[neuron_num], config_basename)

    if save_as_mat:
        savemat(os.path.join(path_out, save_name+'.mat'), mdict={'results':res})
    else:
        np.savez(os.path.join(path_out, save_name+'.npz'), results=res)
    return res
if __name__ == '__main__':

    #################################################
    # User defined input
    #################################################

    # frac of the trials used for fit eval
    frac_eval = 0.2

    # PATH to fit list
    if len(sys.argv) < 3:
        path_fit_list = '../../../demo/fit_list_example_data.yml'
    else:
        path_fit_list = sys.argv[2]

    # save as mat
    save_as_mat = False
    #################################################

    # load the job id (either as an input form the command line, or as a default value if not passed)
    argv = sys.argv
    if len(argv) == 2:  # assumes the script is run the command "python fit_from_config.py fit_num"
        fit_num = int(sys.argv[1]) - 1  # HPC job-array indices starts from 1.
    else:
        fit_num = 0  # set a default value

    fit_from_config(fit_num, path_fit_list, frac_eval=frac_eval, save_as_mat=save_as_mat)