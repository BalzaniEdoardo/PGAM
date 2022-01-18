import numpy as np
from parsing_tools import parse_mat

gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(
    '/Users/edoardo/Work/Code/GAM_code/JP/gam_preproc_ACAd_NYU-28_2020-10-21_001.mat')
"""
Parameter definition for the basis set. 
kernel_len: size in time points of the kenrel (kern_len = 500 are 500ms for 1ms binning)
knots_num: number of equispaced knots that defines the family of polynomial interpolators
ditection: either 0,1,-1, acausal, anticausal, causal
"""
basis_input = {
    'cL': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'choice': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'prev_choice': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'feedback': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'prev_feedback': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'move': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'prior': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'subjective_prior':{
        'kernel_len': 501,
        'knots_num': 10,
        'direction': 0
    },
    'neuron':{
        'kernel_len': 501,
        'knots_num': 10,
        'direction': 1
    }
}

def create_knots():
    return

