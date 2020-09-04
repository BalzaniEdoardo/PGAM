import numpy as np
import sys,os,dill
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
from gam_data_handlers import *
from basis_set_param_per_session import *

def unpack_preproc_data(path,par_list):
    """
    extract the variables in the order specified by the list
    :param path: 
    :param par_list: list containing any of the following
        Xt input concatenated
        yt spike count matrix
        lfp_alpha
        lfp_beta
        lfp_theta
        info_trial array containing all trial information
        trial_idx: array containing the information  about the trial id
        var_names: name of the variable in Xt, is of the same length of the column of Xt
        time_bin: float, binning in sec
        brain_area: vector containing the brain area doe each neuron (same length as the col of yt)
        cluster_id: vector containing the cluster id doe each neuron (same length as the col of yt)
        electrode_id: vector containing the electrode id doe each neuron (same length as the col of yt)
        channel_id: vector containing the channel id doe each neuron (same length as the col of yt)

    :return:
    """

    dat = np.load(path, allow_pickle=True)
    unit_inf = None
    concat = None
    list_out = []
    for par in par_list:
        if par in ['unit_type', 'spike_width', 'waveform', 'amplitude_wf', 'cluster_id', 'electrode_id',
                   'channel_id', 'brain_area', 'uQ', 'isiV', 'cR', 'date_exp', 'presence_rate']:
            if unit_inf is None:
                unit_inf = dat['unit_info'].all()
            list_out += [unit_inf[par]]
        elif par in ['Yt', 'Xt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'trial_idx']:
            if concat is None:
                concat = dat['data_concat'].all()
            list_out += [concat[par]]
        elif par in [ 'var_names', 'time_bin','post_trial_dur', 'pre_trial_dur']:
            list_out += [dat[par]]
        elif par == 'info_trial':
            list_out += [dat['info_trial'].all().trial_type]
    return list_out


def add_smooth(sm_handler, x, var, knots, session,trial_idx,time_bin=0.006,lam=10):
    if var.startswith('neu_'):
        use_info = 'spike_hist'
    else:
        use_info = var

    order = basis_info[session][use_info]['order']
    penalty_type = basis_info[session][use_info]['penalty_type']
    der = basis_info[session][use_info]['der']
    is_temporal_kernel = basis_info[session][use_info]['knots_type'] == 'temporal'
    kernel_length = basis_info[session][use_info]['kernel_length']
    kernel_direction = basis_info[session][use_info]['kernel_direction']
    is_cyclic = basis_info[session][use_info]['is_cyclic']

    if var in sm_handler.smooths_dict.keys():
        sm_handler.smooths_dict.pop(var)
        sm_handler.smooths_var.remove(var)

    sm_handler.add_smooth(var, [x], ord=order, knots=[knots], knots_num=None, perc_out_range=None,
                          is_cyclic=[is_cyclic], lam=lam, penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx, time_bin=time_bin, is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_length, kernel_direction=kernel_direction)
    return sm_handler


    