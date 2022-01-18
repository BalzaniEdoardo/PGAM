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
dict_param = {
    'cL': {
        'kernel_len': 501,
        'knots_num': 10,
        'direction':0
    },
    'cR': {
            'kernel_len': 501,
            'knots_num': 10,
            'direction':0
        },
    'c0': {
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
        'direction': -1
    }
}


def construct_knots(gam_raw_inputs, counts, var_names, dict_param):
    # Standard params for the B-splines
    is_cyclic = False  # no angles or period variables
    order = 4  # cubic spline
    penalty_type = 'der'  # derivative based penalization
    der = 2  # degrees of the derivative
    is_temporal_kernel = True

    cc = 0
    for varName in np.hstack((var_names,['spike_hist'])):
        if varName != 'spike_hist':
            found = False
            for key in dict_param.keys():
                if key in varName:
                    found = True
                    break

            if not found:
                raise ValueError('Variable %s not present in parameters keys!'%varName)
        else:
            key = 'neuron'
        pars = dict_param[key]
        kernel_len = pars['kernel_len']
        knots_num = pars['knots_num']
        direction = pars['direction']
        knots = np.linspace(-kernel_len, kernel_len, knots_num)
        knots = np.hstack(([knots[0]] * 3, knots, [knots[-1]] * 3))
        if varName != 'spike_hist':
            x = gam_raw_inputs[cc]
        else:
            x = counts

        ## eventual additional pre-processing
        cc+= 1
        yield varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der



if __name__ == '__main__':
    gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat('/Users/edoardo/Work/Code/GAM_code/JP/gam_preproc_ACAd_NYU-28_2020-10-21_001.mat')

    for inputs in construct_knots(gam_raw_inputs, var_names, dict_param):
        varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der = inputs
        print(varName)