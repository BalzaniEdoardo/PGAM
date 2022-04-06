import numpy as np
from parsing_tools import parse_mat
from scipy.stats import zscore
from copy import deepcopy
# gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat(
#     '/Users/edoardo/Work/Code/GAM_code/JP/gam_preproc_ACAd_NYU-28_2020-10-21_001.mat')
"""
Parameter definition for the basis set. 
kernel_len: size in time points of the kenrel (kern_len = 500 are 500ms for 1ms binning)
knots_num: number of equispaced knots that defines the family of polynomial interpolators
ditection: either 0,1,-1, acausal, anticausal, causal
"""

dict_param = {
    'cL': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'cR': {
            'kernel_len': 101,
            'knots_num': 10,
            'direction':0
        },
    'c0': {
            'kernel_len': 101,
            'knots_num': 10,
            'direction':0
        },
    'choice': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'prev_choice': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'feedback': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'prev_feedback': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'move': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'prior': {
        'kernel_len': 101,
        'knots_num': 10,
        'direction':0
    },
    'subjective_prior':{
        'kernel_len': 101,
        'knots_num': 10,
        'direction': 0
    },
    'neuron':{
        'kernel_len': 101,
        'knots_num': 10,
        'direction': 1
    }
}


def construct_knots(gam_raw_inputs, counts, var_names, dict_param, trialCathegory_spatial=False, use50Prior=True,
                    expPrior='all'):
    # Standard params for the B-splines
    

    base_chategory = ['prev_choiceL','choiceL','feedback_correct','prev_feedback_correct','prior20']
    # 'choiceL': ['choiceL', 'choice0', 'choiceR'],
    #
    # feedback_correct':['feedback_correct', 'feedback_incorrect'],
    #                  'prev_feedback_correct':['prev_feedback_correct','prev_feedback_incorrect'],
    cathegory_vars = {'prior20':['prior20','prior50','prior80'],
                      'prev_feedback_correct': ['prev_feedback_correct', 'prev_feedback_incorrect'],
                      'prev_choiceL': ['prev_choiceL', 'prev_choice0', 'prev_choiceR']
                      }

    cathegory_vals = {'prev_choiceL':[-1, 0, 1],
                      'prev_feedback_correct':[1,0],
                      'prior20':[20,50,80]}
    if not use50Prior:
        cathegory_vars['prior20'] = ['prior20','prior80']
        cathegory_vals['prior20'] = [20, 80]
    
    if expPrior != 'all':
        idx_prior = np.where(np.array(var_names) == expPrior)[0]
        bl = np.array(gam_raw_inputs[idx_prior], dtype=bool).reshape(-1,)
        gam_raw_inputs = gam_raw_inputs[:, bl]
        counts = counts[bl]
        
    # cc = 0
    all_vars = np.hstack((var_names,['spike_hist']))
    for varName in all_vars:

        is_cyclic = False  # no angles or period variables
        order = 4  # cubic spline
        penalty_type = 'der'  # derivative based penalization
        der = 2  # degrees of the derivative

        is_cathegorical = varName in list(np.hstack(list(cathegory_vars.values())))
        cc = np.where(all_vars==varName)[0]
        if len(cc) != 1:
            continue
        cc = cc[0]
        is_temporal_kernel = True
        if varName == 'subjective_prior' or ('movement_PC' in varName):
            is_temporal_kernel = False

        if is_cathegorical and trialCathegory_spatial:
            is_temporal_kernel = False


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
        if varName != 'spike_hist':
            x = deepcopy(gam_raw_inputs[cc])
        else:
            x = deepcopy(counts)

        if (not is_temporal_kernel) and (not is_cathegorical):
            ## x should be already z-scored otherwise do this
            minx = np.nanpercentile(x,1)
            maxx = np.nanpercentile(x,99)
            valid = (x <= maxx) * (x >= minx)
            x[~valid] = np.nan
            # scale to (0,1)
            x = (x - minx) / (maxx - minx)
            loc = minx
            scale = (maxx - minx)
            knots = np.linspace(0,1,8)
            knots = np.hstack(([knots[0]] * 3, knots, [knots[-1]] * 3))

        elif (not is_temporal_kernel) and is_cathegorical:
            order = 1
            penalty_type = 'EqSpaced'
            loc = 0
            scale = 1
            if not varName in base_chategory:

                x = np.nan * np.ones(x.shape[0])
                is_cyclic = False
                yield varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale


            else:
                var_list = cathegory_vars[varName]
                x_out = np.nan * np.ones(x.shape[0])
                for cat_num in range(len(var_list)):
                    var = var_list[cat_num]
                    val = cathegory_vals[varName][cat_num]
                    cc = np.where(all_vars == var)[0]
                    assert( len(cc) == 1)
                    cc = cc[0]
                    xcat = gam_raw_inputs[cc]
                    if np.sum(xcat==1) < 10:
                        continue
                    x_out[xcat == 1] = val
                x = x_out
                unq_x = np.sort(np.unique(x[~np.isnan(x)]))
                knots = np.hstack((unq_x - 0.01, [unq_x[-1] + 0.01]))


        elif direction == 1:
            knots = np.hstack(([(10) ** -6] * 3, np.linspace((knots_num) ** -6, kernel_len // 2, 10), [kernel_len // 2] * 3))
            loc = np.nan
            scale = np.nan
        else:
            knots = np.linspace(-kernel_len, kernel_len, knots_num)
            knots = np.hstack(([knots[0]] * 3, knots, [knots[-1]] * 3))
            loc = np.nan
            scale = np.nan


        ## eventual additional pre-processing
        # cc+= 1
        yield varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale



if __name__ == '__main__':
    gam_raw_inputs, counts, trial_idx, info_dict, var_names = parse_mat('D:\\MOUSE-ASD-NEURONS\\data\\3step\\data\\C\\ACAd\\ACAd_CSP017_2020-11-20_001.mat')

    for inputs in construct_knots(gam_raw_inputs, var_names, dict_param):
        varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der = inputs
        print(varName)