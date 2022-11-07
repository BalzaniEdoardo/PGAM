#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:26:52 2022

@author: edoardo
"""
from copy import deepcopy
import warnings
import numpy as np
import scipy.stats as sts

from copy import deepcopy

# create function that returns bspline pars for this experiment
def knots_create(covs, counts, var_name_list, cov_info,
                min_obs_between_knots=1000):
    """
    This function gets as an input the covarate time course, and the cov_info for all covarates and returns 
    all the parameters needed to set a 'GAM_library/gam_data_handler.covariate_smooth' object.
    
    input:
    ======
        - covs: M x T matrix of the covariaets
            covs[i,:] : time course of the i-th covariate 
            
        - counts: T dim array of the spike counts
        
        - var_name_list: M+1 array of the covariate names
            if i < M
                var_name_list[i] corresponds to covs[i,:]
            if  i == M
                var_name_list[i] must be 'spike_history'
        
        - cov_info: dictionary containing the parameters for creating the knots and the input to 
            GAM_library/gam_data_handler.covariate_smooth
        
            
    """
    
    
    
    assert(set(cov_info.keys()) == set(var_name_list).intersection(set(cov_info.keys())))
    assert('spike_hist' == var_name_list[-1])
    cov_smooth_input = {}
    percentiles_info = {'levels':[5,10,25,50,75,90,95], 'vals':{}}
    i = 0
    
    for var in var_name_list:
        print(var)
        flag_coupling = False
        
        if not (var in cov_info.keys()):
            flag_coupling = True
            true_var = deepcopy(var)
            var = 'spike_hist'
        info = cov_info[var]
        order = info['order']
        is_cyclic = info['is_cyclic']
        lam = info['initial_penalty']
        is_temporal_kernel = info['is_temporal_kernel']
        is_categorical = info['is_categorical']
        kernel_direction = info['kernel_direction']
        penalty_type = info['penalty_type']
        der = info['der_order']
        is_zscore = info['zscore']
        repeat_extreme_knots = False
        knots = info['knots']
        kernel_len = info['kernel_len']
        
        if penalty_type == 'der':
            assert (order > 2), '%s: der penalty requires order > 2!'%var
        # check that knots are not repeated by the user
        if (not type(knots) is int) and (not knots is None):
            knots = np.array(knots)
            assert (set(knots) == set(np.unique(knots))) , '%s: knots should not repeat!'%var
            
        mu,sd = None,None
        if var == 'spike_hist':
            x = deepcopy(counts)
        else:
            x = deepcopy(covs[i])
            
        if is_zscore:
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            x = (x - mu)/sd
        
        percentiles_info['vals'][var] = np.nanpercentile(x,percentiles_info['levels'])
        
        if (not is_temporal_kernel) and (not is_categorical):
            # check knots occupancy
            median_val = np.nanmedian(x)
            # if the median of the session distribution is outside the knots range,
            # flag a warning but use the mid knot interval as the center of the range
            if (median_val < knots[0]) or (median_val > knots[-1]):
                warnings.warn('Median of %s outside the knots range!'%var, UserWarning, stacklevel=2)
                median_val = (knots[0] + knots[-1])*0.5
            
            flag_range = np.zeros(knots.shape[0]-1,dtype=bool)
            center_bin = np.where(knots <= median_val)[0][-1]
            for j in range(center_bin, len(knots)-1):
                if not ((x>=knots[j]) & (x<knots[j+1])).sum() > min_obs_between_knots:
                    break
                flag_range[j] = True
                
            for j in range(center_bin,-1,-1):
                if not ((x>=knots[j]) & (x<knots[j+1])).sum() > min_obs_between_knots:
                    break
                flag_range[j] = True
            selknots = np.where(flag_range)[0]
            if selknots.shape[0] == 0:
                warnings.warn('Not enough values available for var %s!'%var, UserWarning, stacklevel=2)
                knots = None
            else:
                knots = np.hstack((knots[:-1][selknots],[knots[selknots[-1]+1]]))
                # repeat knots
                knots = np.hstack(([knots[0]]*(order-1), knots, [knots[-1]]*(order-1)))
        
        elif is_categorical:
            # create a step function basis, each step contains a single cathegory
            assert(order == 1)
            assert(penalty_type == 'EqSpaced')
            category_list = []
            tmp = np.unique(x[~np.isnan(x)])
            for cc in tmp:
                if (x == cc).sum() < min_obs_between_knots:
                    continue
                category_list.append(cc)
            assert (len(category_list) > 1), '%s: <= 1 category meet the inclusion criteria'
            category_list = np.array(category_list)
            knots = list((category_list[1:] + category_list[:-1])*0.5)
            knots = [category_list[0] - (knots[0]-category_list[0])] + knots +\
                [category_list[-1] + (knots[-1]-category_list[-2])]
            knots = np.array(knots)
            
        else:
            # temporal kernel
            assert (type(knots) is int), '%s: Specify the number of knots for temporal kernels'%var
            knots_num = deepcopy(knots)
            if kernel_direction == 1:
                knots = np.hstack(([(10) ** -6] * (order-1), np.linspace(10 ** -6, 
                                    kernel_len // 2, 10), [kernel_len // 2] * (order-1)))
            elif kernel_direction == 0:
                knots = np.linspace(-kernel_len, kernel_len, knots_num)
                knots = np.hstack(([knots[0]] * (order-1), knots, [knots[-1]] * (order-1)))
            else:
                tmp = - np.linspace((10) ** -6, kernel_len // 2, 10) 
                knots = np.hstack(([-(10) ** -6] * (order-1), tmp, [tmp[-1]] * (order-1)))
                knots = knots[::-1]
        
        #varName, knots, x, is_cyclic, order, kernel_len, direction, is_temporal_kernel, penalty_type, der, loc, scale
        if flag_coupling:
            var = true_var
        
        cov_smooth_input[var] = {'knots':[knots], 
                                 'x':[x],
                                 'is_cyclic':is_cyclic,
                                 'order':order,
                                 'kernel_len':kernel_len,
                                 'kernel_direction':kernel_direction,
                                 'is_temporal_kernel':is_temporal_kernel,
                                 'penalty_type':penalty_type,
                                 'der':der,
                                 'loc':mu,
                                 'scale':sd,
                                 'lam':lam,
                                 'repeat_extreme_knots':repeat_extreme_knots
                                 }
        i += 1
        
    return cov_smooth_input,percentiles_info


if __name__ == '__main__':
    from scipy.io import loadmat
    import sys
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    from gam_data_handlers import smooths_handler
    from GAM_library import general_additive_model
    import statsmodels.api as sm
    from der_wrt_smoothing import deriv3_link, d2variance_family

    dat = loadmat('/Users/edoardo/Work/Code/GAM_code/JP/MOUSE_FF/data/ON_VISp_CI033_20210421.mat')
    # create the cov info dictionary
    knots_info = np.load('knots_for_bspline.npy',allow_pickle=True).all()
    cov_info = {
        'rad target': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': 2,
                'zscore' :False,
                'knots':knots_info['rad target'],
                'kernel_len':None,
                },
        'ymv': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': 2,
                'zscore' :False,
                'knots':knots_info['ymv'],
                'kernel_len':None,
                },
        'yma': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': 2,
                'zscore' :False,
                'knots':knots_info['ymv'],
                'kernel_len':None,
                },
        'ymp': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': 2,
                'zscore' :False,
                'knots':knots_info['ymp'],
                'kernel_len':None,
                },
        'yfp': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': 2,
                'zscore' :False,
                'knots':knots_info['yfp'],
                'kernel_len':None,
                },
        'yfv': {'order': 1,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': True,
                'kernel_direction': None,
                'penalty_type': 'EqSpaced',
                'der_order': None,
                'zscore' :False,
                'knots':None,
                'kernel_len':None,
                },
        'pd': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': None,
                'zscore' :True,
                'knots':knots_info['pd'],
                'kernel_len':None,
                },
        'eyeV': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': None,
                'zscore' :True,
                'knots':knots_info['eyeV'],
                'kernel_len':None,
                },
        'eyeH': {'order': 2,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'diff',
                'der_order': None,
                'zscore' :True,
                'knots':knots_info['eyeH'],
                'kernel_len':None,
                },
        'moveOn': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 0,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'MoveOff': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 0,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'FFon': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 0,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'FFoff': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 0,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'reward': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 0,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'theta': {'order': 4,
                'is_cyclic': [True],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': knots_info['theta'],
                'kernel_len':None,
                },
         'beta': {'order': 4,
                'is_cyclic': [True],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': knots_info['beta'],
                'kernel_len':None,
                },
        'alpha': {'order': 4,
                'is_cyclic': [True],
                'initial_penalty': 10,
                'is_temporal_kernel': False,
                'is_categorical': False,
                'kernel_direction': None,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': knots_info['alpha'],
                'kernel_len':None,
                },
        'spike_hist': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 1,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'PTB1': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 1,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'PTB2': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 1,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'PTB3': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 1,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'PTB4': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 1,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                },
        'PTB5': {'order': 4,
                'is_cyclic': [False],
                'initial_penalty': 10,
                'is_temporal_kernel': True,
                'is_categorical': False,
                'kernel_direction': 1,
                'penalty_type': 'der',
                'der_order': 2,
                'zscore' :False,
                'knots': 8,
                'kernel_len':201,
                }
        
        
        }

    # neu_num = 13

    # counts = np.squeeze(dat['N'][neu_num,:])
    # F_var_list = ['rad target', 'ymp', 'ymv', 'yma', 'yfp', 'yfv', 'pd',
    #            'eyeH', 'eyeV', 'FFon', 'FFoff', 'moveOn', 'MoveOff', 'reward',
    #            'theta', 'alpha', 'beta']
    # neu_list = []
    # for var in dat['variable_names'][0]:
    #     if (not var[0] in F_var_list) and (var[0]!='yfa'):
    #         neu_list.append(var[0])
    # neu_name = neu_list[neu_num]
    # neu_list.remove(neu_name)
    # neu_list = []
    # var_list = F_var_list + neu_list + ['spike_hist']
    
    # orig_var_list = []
    # for k in range(len(dat['variable_names'][0])):
    #     orig_var_list.append(dat['variable_names'][0][k][0])
    # orig_var_list = np.array(orig_var_list)
    
    # # create list of covariates with the same order as var_list
    # covs = np.zeros((len(var_list),dat['F'].shape[1]))
    # cc = 0
    # for var in var_list:
    #     if var in F_var_list:
    #         idx = np.where(orig_var_list == var)[0][0]
    #         covs[cc] = dat['F'][idx]
        
    #     elif var == 'spike_hist':
    #         idx = np.where(orig_var_list == neu_name)[0][0] - len(F_var_list) - 1
    #         covs[cc] = dat['N'][idx,:]
    # #         print(neu_name,var,idx)
            
    #     else:
    #         idx = np.where(orig_var_list == var)[0][0] - len(F_var_list) - 1
    # #         print(neu_name,var,idx)
    #         covs[cc] = dat['N'][idx,:]
        
    #     cc += 1
    
    # input_dict, perc_dict = knots_create(covs, counts, var_list, cov_info, min_obs_between_knots=1000)

    # # create the input
    # sm_handler = smooths_handler()
    # for var in var_list:
    #     var_dict = input_dict[var]
    #     print(var, var_dict['knots'])
    #     sm_handler.add_smooth(var, var_dict['x'], ord=var_dict['order'], knots=var_dict['knots'],
    #                       penalty_type=var_dict['penalty_type'], der=var_dict['der'], 
    #                       kernel_length=var_dict['kernel_len'],
    #                       kernel_direction=var_dict['kernel_direction'],trial_idx=np.squeeze(dat['T']),
    #                       is_temporal_kernel=var_dict['is_temporal_kernel'], time_bin=0.005,
    #                       lam=var_dict['lam'],is_cyclic=var_dict['is_cyclic'])
    
    # link = deriv3_link(sm.genmod.families.links.log())
    # poissFam = sm.genmod.families.family.Poisson(link=link)
    # family = d2variance_family(poissFam)
    
    # # train trail (90% dataset)
    # trial_idx = np.squeeze(dat['T'])
    # unchosen = np.arange(0, np.unique(trial_idx).shape[0])[::10]
    # choose_trials = np.array(list(set(np.arange(0, np.unique(trial_idx).shape[0])).difference(set(unchosen))),dtype=int)
    # choose_trials = np.unique(trial_idx)[np.sort(choose_trials)]
    # filter_trials = np.zeros(trial_idx.shape[0], dtype=bool)
    # for tr in choose_trials:
    #     filter_trials[trial_idx==tr] = True
    
    # filter_trials =  np.ones(trial_idx.shape[0], dtype=bool)
    # gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,counts,poissFam,fisher_scoring=False)
    
    # full_fit, reduced_fit = gam_model.fit_full_and_reduced(sm_handler.smooths_var,th_pval=0.001,
    #                                                   smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
    #                                                   conv_criteria='deviance',
    #                                                   initial_smooths_guess=False,
    #                                                   method='L-BFGS-B',
    #                                                   gcv_sel_tol=10 ** (-10),
    #                                                   use_dgcv=True,
    #                                                   fit_initial_beta=True,
    #                                                   trial_num_vec=trial_idx,
                                                      # filter_trials=filter_trials)
            
            