#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:42:32 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import scipy.stats as sts
from copy import deepcopy


# exclude_eye_position = ['m44s213','m53s133','m53s134','m53s105','m53s182']

def knots_cerate(x,var,session, hand_vel_temp=False,hist_filt_dur='short',
                 exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'],
                 condition=None):
    """
        this function returns the knots, a transformed version of the data
        and a boolean indicating if the variable should be included in the GAM.
        Input are z-scored if var is eye position, otherwise no transformation
        is applied

    Parameters
    ----------
    x : numpy array dim (N,)
        input raw concatenated data.
    var : string
        name of the variable.
    session : string
        sessioin id, format is m\d+s\d+ as a regexp pattern
    knots_num : int, optional
        number of knots for the basis. The default is 8.
    exclude_eye_position : TYPE, optional
        list of session with bad eye tracking. The default is ['m44s213','m53s133','m53s134','m53s105','m53s182'].

    Returns
    -------
    knots : numpy array dim (knots_num,)
        knots for the basis set.
    x_trans : numpy array dim (N,)
        transformed input.
    include_var : bool
        do you have to include the variable or not?

    """
    is_cyclic = False
    include_var = True
    kernel_len = None
    order = 4
    
    penalty_type = 'der'
    der = 2
    is_temporal_kernel = False
    kernel_direction = 0



    if 'eye' in var:
        if session in exclude_eye_position:
            knots = None
            x_trans = None
            include_var = False
            return knots, x_trans, include_var, is_cyclic, order,kernel_len,None,None,None,None
        # remove giant outliers
        x_trans = deepcopy(x)
        x_trans[np.abs(x) > np.nanpercentile(np.abs(x), 99)] = np.nan
        
        # zscore
        x_trans = (x_trans - np.nanmean(x_trans)) / np.nanstd(x_trans)
    else:
        x_trans = x
    
    if var == 'rad_vel':
        if condition != 'controlgain':
            knots = np.hstack((np.linspace(0,150,5)[:-1],np.linspace(150,200,3)))
            knots = np.hstack((
                [knots[0]] * 3,
                knots,
                [knots[-1]] * 3
            ))
            x_trans[(x_trans > 200) | (x_trans < 0)] = np.nan
        else:
            knots = np.linspace(0, 400, 11)
            order = 1
            penalty_type = 'EqSpaced'
            # conditioin that holds when there is no gain
            if (x_trans > 200).sum() / np.sum(~np.isnan(x_trans)) < 0.1:
                idx_maxVel = np.where(knots == 200)[0][0]
                knots = knots[:idx_maxVel+1]
                x_trans[(x_trans > 200) | (x_trans < 0)] = np.nan
            else:
                x_trans[(x_trans > 400) | (x_trans < 0)] = np.nan

    
    elif var == 'ang_vel':
        if condition != 'controlgain':
            knots = np.linspace(-65,65,6)
            knots = np.hstack((
                [knots[0]] * 3,
                knots,
                [knots[-1]] * 3
            ))
        else:
            knots = np.linspace(-91, 91, 8)
            order = 1
            penalty_type = 'EqSpaced'
            # knots = np.hstack((
            #     np.linspace(-100,-65,3),knots[1:-1],np.linspace(65,100,3)))
        
            # condition for gain == 2
            if np.nanmax(x_trans) > 120:
                x_trans[np.abs(x_trans) > 91] = np.nan
            else:

                 knots = np.linspace(-65,65,6)

                 x_trans[np.abs(x_trans) > 65] = np.nan
    
    elif var == 'rad_path':
        knots = np.linspace(0,350,6)
        
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        x_trans[(x_trans > 350) | (x_trans < 0)] = np.nan
        
    elif var == 'rad_path_from_xy':
        knots = np.linspace(0,350,6)
        
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        x_trans[(x_trans > 350) | (x_trans < 0)] = np.nan
    
    elif var == 'ang_path':
        knots = np.linspace(-60,60,6)
        
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        x_trans[(x_trans > 60) | (x_trans < -60)] = np.nan
        
    elif var == 'rad_target':
        knots = np.linspace(0,400,6)
        
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        
        x_trans[(x_trans > 400) | (x_trans < 0)] = np.nan
        
        
    elif var == 'ang_target':
        knots = np.linspace(-50,50,6)
        
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        
        x_trans[(x_trans > 50) | (x_trans < -50)] = np.nan
    
    elif var == 'eye_vert' or var == 'eye_hori':
        knots = np.linspace(-2,2,8)
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        
        x_trans[(x_trans > 2) | (x_trans < -2)] = np.nan
    
    elif var in ['lfp_beta', 'lfp_alpha','lfp_theta']:
        knots = np.linspace(-np.pi,np.pi,8)
        # knots = np.hstack(([knots[0]]*3,
        #                         knots,
        #                         [knots[-1]]*3
        #                        ))
        is_cyclic = True
        
    elif var.startswith('t_') and var != 't_flyOFF' and var!='t_ptb':
        kernel_len = 165
        knots = np.linspace(-165,165,10)
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        is_temporal_kernel = True
    
    elif var =='t_ptb':
        kernel_len = 401
        knots = np.linspace(10**-6,200,10)
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        is_temporal_kernel = True
        kernel_direction = 1
        
    elif var == 't_flyOFF':
        kernel_len = 322
        knots = np.linspace(-327,327,11)
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        
        is_temporal_kernel = True
        kernel_direction = 0
        
    elif var == 'T_ang_move_init':
        kernel_len = 160 # about half a second 80*6ms
        knots = np.linspace(0,kernel_len,6)
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        
        is_temporal_kernel = True
        kernel_direction = 1
    
    elif var == 'ang_acc':
        knots = np.linspace(-250,250,6)
        
        knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
        
        x_trans[(x_trans > 250) | (x_trans < -250)] = np.nan
        
    elif var == 'rad_acc':
       knots = np.linspace(-1000,1000,8)
       
       knots = np.hstack(([knots[0]]*3,
                               knots,
                               [knots[-1]]*3
                              ))
       
       x_trans[(x_trans > 1000) | (x_trans < -1000)] = np.nan
      
    elif var == 'hand_vel1':
        if not hand_vel_temp:
        
           knots = np.linspace(-100,100,6)
           
           knots = np.hstack(([knots[0]]*3,
                                   knots,
                               [knots[-1]]*3
                              ))
       
           x_trans[(x_trans > 100) | (x_trans < -100)] = np.nan
           
        else:
             kernel_len = 165
             knots = np.linspace(-165,165,10)
             knots = np.hstack(([knots[0]]*3,
                                knots,
                                [knots[-1]]*3
                               ))
             is_temporal_kernel = True
       
    elif var == 'hand_vel2':
      if not hand_vel_temp:
          knots = np.linspace(-70,70,6)
      
          knots = np.hstack(([knots[0]]*3,
                              knots,
                              [knots[-1]]*3
                             ))
      
          x_trans[(x_trans > 70) | (x_trans < -70)] = np.nan
      else:
          kernel_len = 165
          knots = np.linspace(-165,165,10)
          knots = np.hstack(([knots[0]]*3,
                           knots,
                           [knots[-1]]*3
                          ))
          is_temporal_kernel = True
     
    elif var == 'spike_hist':
        if hist_filt_dur == 'short':
            kernel_direction = 1
            kernel_len = 11
            knots = np.linspace((10)**-6,5,6)
            penalty_type = 'EqSpaced'
            order = 1
            
            # kernel_len = 199
            # knots = np.linspace((10)**-6,5,100)
            # penalty_type = 'adaptive'
            # order = 4
            
            der = 2
            is_temporal_kernel=True
        elif hist_filt_dur == 'long_many':
            kernel_direction = 1
            #kernel_len = 99
            kernel_len = 199
            # order = 1
            # knots = np.linspace((10)**-6,5,8)
            # penalty_type = 'EqSpaced'
            
            order = 4
            knots = np.hstack(([(10)**-6]*3,np.linspace((10)**-6,5,12),[5]*3))
            penalty_type = 'der'
            der = 2
            is_temporal_kernel=True
            
        else:
            kernel_direction = 1
            #kernel_len = 99
            kernel_len = 199
            # order = 1
            # knots = np.linspace((10)**-6,5,8)
            # penalty_type = 'EqSpaced'
            
            order = 4
            knots = np.hstack(([(10)**-6]*3,np.linspace((10)**-6,5,6),[5]*3))
            penalty_type = 'der'
            der = 2
            is_temporal_kernel=True

        
        
    else:
        raise ValueError('Knots for variable %s not implemented!'%var)
    
    
    
    return knots, x_trans, include_var, is_cyclic, order, \
        kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der


if __name__ == '__main__':
# if False:
    from utils_loading import unpack_preproc_data, add_smooth
    # dat = np.load('input_hist.npz')
    # hist_matrix = dat['hist']
    # edge_matrix = dat['edge']
    # info = dat['info']
    # use_var = 'rad_path_from_xy'
    use_var = 'rad_path'
    reload = True
    perform_PQL = False
    fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s113.npz'
    session = os.path.basename(fhName).split('.')[0]
    # dat = np.load(fhName,allow_pickle=True)
    neuron = 76
    # neuron = 2
    cond_type = 'all'
    cond_value = 1
    cond_knots = 'all'
    par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
            'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
            'unit_type']
    if reload:
        (Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
         trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
         cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type) = unpack_preproc_data(fhName, par_list)

    if cond_type != 'odd':
        idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
    else:
        idx_subselect = np.where(trial_type['all'])[0]
        if cond_value == 1:
            idx_subselect = idx_subselect[1::3]
        else:
            idx_subselect = idx_subselect[::3]
    keep = []
    for ii in idx_subselect:
        keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

    print(' condition', cond_type, cond_value)

    keep = np.array(keep, dtype=int)
    trial_idx = trial_idx[keep]

    sm_handler = smooths_handler()

    for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
        # for now skip
        if var != use_var:
            continue
        if 'hand' in var:
            continue
        if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
            is_cyclic = True
        else:
            is_cyclic = False

        if var == 'lfp_theta':
            x = lfp_theta[keep, neuron - 1]

        elif var == 'lfp_beta':
            x = lfp_beta[keep, neuron - 1]

        elif var == 'lfp_alpha':
            x = lfp_alpha[keep, neuron - 1]

        elif var == 'spike_hist':
            tmpy = yt[keep, neuron - 1]
            x = tmpy
            # x = np.hstack(([0], tmpy[:-1]))

        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep, cc]



        knots, x_trans, include_var, is_cyclic, order,\
            kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,session,hand_vel_temp=True,hist_filt_dur='short',
                              exclude_eye_position=['m44s213','m53s133','m53s134','m53s105','m53s182'],
                             condition=cond_knots)

        if include_var:
            if var in sm_handler.smooths_dict.keys():
                sm_handler.smooths_dict.pop(var)
                sm_handler.smooths_var.remove(var)

            sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                                  knots_num=None, perc_out_range=None,
                          is_cyclic=[is_cyclic], lam=50.,
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx, time_bin=0.006,
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len, kernel_direction=kernel_direction,
                          ord_AD=3,ad_knots=8,repeat_extreme_knots=False)



    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)

    gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                        fisher_scoring=True)

    t0 = perf_counter()
    full, reduced = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001, method='L-BFGS-B', tol=1e-8,
                                                    conv_criteria='gcv',
                                                    max_iter=100, gcv_sel_tol=10 ** -13, random_init=False,
                                                    use_dgcv=True, initial_smooths_guess=False,
                                                    fit_initial_beta=True, pseudoR2_per_variable=True,
                                                    trial_num_vec=trial_idx, k_fold=False, fold_num=5,
                                                    reducedAdaptive=False,compute_MI=True,
                                                    perform_PQL=perform_PQL)

# # =============================================================================
# #     PLOT STUFF
# # =============================================================================
    gam_res = full
    FLOAT_EPS = np.finfo(float).eps
    import matplotlib.pylab as plt

    var_list = gam_res.var_list

    pvals = np.clip(gam_res.covariate_significance['p-val'], FLOAT_EPS, np.inf)
    dropvar = np.log(pvals) > np.mean(np.log(pvals)) + 1.5 * np.std(np.log(pvals))
    dropvar = pvals > 0.001
    drop_names = gam_res.covariate_significance['covariate'][dropvar]
    fig = plt.figure(figsize=(14, 8))
    plt.suptitle('%s - neuron %d  - PQL %d - %s %f' % (session, neuron,perform_PQL,cond_type,cond_value))
    cc = 0
    cc_plot = 1
    for var in np.hstack((var_names, ['lfp_beta','spike_hist'])):
        # if var != use_var:
        #     continue

        if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
            cc += 1
            continue
        print('plotting var', var)

        ax = plt.subplot(5, 4, cc_plot)
        # ax = plt.subplot(1, 1, cc_plot)

        if var == 'spike_hist':
            continue
        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep, cc]
            # max_x, min_x = X[var].max(), X[var].min()
            min_x = gam_res.smooth_info[var]['knots'][0][0]
            max_x = gam_res.smooth_info[var]['knots'][0][-1]
            min_x = np.max([min_x,np.nanpercentile(x, 1)])
            max_x = np.min([max_x,np.nanpercentile(x, 99)])



        if gam_res.smooth_info[var]['is_temporal_kernel']:

            dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
            knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
            ord_ = gam_res.smooth_info[var]['ord']
            idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

            impulse = np.zeros(dim_kern)
            impulse[(dim_kern - 1) // 2] = 1
            xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
            fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99, trial_idx=None)
            if var != 'spike_hist':

                xx = xx[idx_select][1:-1]
                fX = fX[idx_select][1:-1]
                fX_p_ci = fX_p_ci[idx_select][1:-1]
                fX_m_ci = fX_m_ci[idx_select][1:-1]
            else:
                if xx.shape[0] > 20:
                    xx = xx[:(-ord_ - 1)]
                    fX = fX[:(-ord_ - 1)]
                    fX_p_ci = fX_p_ci[:(-ord_ - 1)]
                    fX_m_ci = fX_m_ci[:(-ord_ - 1)]
                else:
                    xx = xx[:(-ord_ )]
                    fX = fX[:(-ord_ )]
                    fX_p_ci = fX_p_ci[:(-ord_ )]
                    fX_m_ci = fX_m_ci[:(-ord_ )]


        else:
            knots = gam_res.smooth_info[var]['knots']
            knots_sort = np.unique(knots[0])
            knots_sort.sort()

            xx = np.linspace(min_x,max_x,100)#(knots_sort[1:] + knots_sort[:-1]) * 0.5

            fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)
        # if np.sum(drop_names == var):
        #     label = var
        # else:
        #     label = var
        label = var
        if var == 'spike_hist':
            iend = xx.shape[0] // 2

            print('set spike_hist')

            iidx = np.where(impulse==1)[0][0]
            if impulse.shape[0] < 20:
                iidx = iidx
            fX = fX[iidx + 1:][::-1]
            fX_p_ci = fX_p_ci[iidx+1:][::-1]
            fX_m_ci = fX_m_ci[iidx+1:][::-1]
            plt.plot(xx[:fX.shape[0]], fX, ls='-',marker='o', color='k', label=label)
            plt.fill_between(xx[:fX_m_ci.shape[0]], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
        else:
            plt.plot(xx, fX, ls='-', color='k', label=label)
            plt.fill_between(xx, fX_m_ci, fX_p_ci, color='k', alpha=0.4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend()

        cc += 1
        cc_plot += 1
        
    # plt.figure()
    # full_tmp = deepcopy(gam_res)
    # numbasis = len(full_tmp.beta[full_tmp.index_dict[use_var]])
    # plt.figure()
    # xvar = use_var
    # if use_var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
    #     is_cyclic = True
    # else:
    #     is_cyclic = False

    # if var == 'lfp_theta':
    #     x = lfp_theta[keep, neuron - 1]

    # elif var == 'lfp_beta':
    #     x = lfp_beta[keep, neuron - 1]

    # elif var == 'lfp_alpha':
    #     x = lfp_alpha[keep, neuron - 1]

    # elif var == 'spike_hist':
    #     tmpy = yt[keep, neuron - 1]
    #     x = tmpy
    #     # x = np.hstack(([0], tmpy[:-1]))

    # else:
    #     cc = np.where(var_names == use_var)[0][0]
    #     x = Xt[keep, cc]

    # x0 = np.nanpercentile(x,1)
    # x1 = np.nanpercentile(x,99)


    # x0 = np.max([min_x,x0])
    # max_x = np.min([max_x,x1])
    # x = np.linspace(x0,x1,100)
    # for k in range(numbasis):
    #     beta = np.zeros(numbasis)
    #     beta[k] = 1
    #     # full_tmp.beta[full_tmp.index_dict[var]] =
    #     full_tmp.beta[full_tmp.index_dict[use_var]] = beta

    #     fX,_,_ = full_tmp.smooth_compute([x],use_var)
    #     plt.plot(x,fX)