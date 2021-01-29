#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:31:59 2021

@author: edoardo
GAM fit for an example session

"""
import numpy as np
import sys, os, dill
import matplotlib.pylab as plt
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)

# GAM_library contains all the main function and classes related to the GAM model
sys.path.append(os.path.join(main_dir,os.path.join('GAM_library')))
from GAM_library import *
# Some function that i find useful specifically for the firefly data
# (like the function for creating knots)
sys.path.append(os.path.join(main_dir,os.path.join('firefly_utils')))
from knots_constructor import knots_cerate

# this object contains spikes from all trials concatenated, the LFP istantaneous
# phase for several frequencies, and the input variables
dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s113.npz',
              allow_pickle=True)

fit_unit = 1


# Unpack all the variables

# GAM model inputs
concat = dat['data_concat'].all()
X = concat['Xt']
spikes = concat['Yt']
var_names = dat['var_names']
trial_idx = concat['trial_idx']
lfp_beta = concat['lfp_beta']
lfp_alpha = concat['lfp_alpha']
lfp_theta = concat['lfp_theta']
del concat



# =============================================================================
# GAM input variable description
# =============================================================================
# X : (time point, num variables) np.array of float
#     trial concatenated input variables

# spikes : (time point, num units) np.array of int
#     trial concatenated spike counts

# var_names: (num variables,) np.array  of str
#     X[:, k] is the vector with the time series of variable var_names[k]

# trial_idx :  (time point,) np.array  of int
#     trial ID for each time point

# lfp_beta/alpha/theta : (time point, num units) np.array of float
#     lfp phase in the frequencies 
#     beta: [12 Hz - 30 Hz]
#     alpha: [8 Hz - 12 Hz]
#     theta: [4 Hz - 8 Hz]
# =============================================================================


## Info regarding the units:
# quality metric for the units
unit_info = dat['unit_info'].all()
unit_type = unit_info['unit_type']
isiV = unit_info['isiV'] # % of isi violations 
cR =  unit_info['cR'] # contamination rate
presence_rate = unit_info['presence_rate'] # measure of stability of the firing in time

# std filters for unit quality
cont_rate_filter = (cR < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate > 0.9
isi_v_filter = isiV < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

brain_area = unit_info['brain_area'][combine_filter]
cluster_id = unit_info['cluster_id'][combine_filter]
electrode_id = unit_info['electrode_id'][combine_filter]
channel_id = unit_info['channel_id'][combine_filter]
unit_type = unit_type[combine_filter]

spikes =spikes[:,combine_filter]
lfp_alpha = lfp_alpha[:,combine_filter]
lfp_beta = lfp_alpha[:,combine_filter]
lfp_theta = lfp_alpha[:,combine_filter]
keep_unit = np.arange(1,1+combine_filter.shape[0])[combine_filter]
fit_unit = keep_unit[fit_unit]

##  truncate ang dist (this variable become noisy when the monkey is close to the target)
ang_idx = np.where(np.array(var_names) == 'ang_target')[0][0]
X[np.abs(X[:, ang_idx]) > 50, ang_idx] = np.nan


# extract trial info

# trial_type = np.array of dimension (num trials,) with column the different conditions 
trial_type = dat['info_trial'].all().trial_type 
idx_subselect = np.where(trial_type['all'] == True)[0]
keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

keep = np.array(keep,dtype=int)


## get the paramters defining the spline-basis
# create knots and check that the location is covering the input distribution
# create the class smooth_handler that will contain the variablewith the 
# input transformed in the spline basis set and method useful for fitting
sm_handler = smooths_handler()

plt.figure(figsize=(10,8))
plt.suptitle('knots placement')
cnt = 1

for var in np.hstack((var_names,['lfp_beta','spike_hist'])):
    
    if 'hand' in var:
        # hand vel not tracked
        continue
    
    

    if var == 'lfp_theta':
        x = lfp_theta[keep, fit_unit]

    elif var == 'lfp_beta':
        x = lfp_beta[keep, fit_unit]

    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, fit_unit]

    elif var == 'spike_hist':
        tmpy = spikes[keep, fit_unit]
        x = tmpy
        # x = np.hstack(([0], tmpy[:-1]))

    else:
        cc = np.where(var_names == var)[0][0]
        x = X[keep, cc]

    knots, x_trans, include_var, is_cyclic, order,\
        kernel_len,kernel_direction,is_temporal_kernel,penalty_type,der =\
                knots_cerate(x,var,'m53s113')
         
    # Output description
    # knots: location of the interp knots
    
    # x_trans: transformation of the data (set nan outside the knots range,
    # z-score eye position)
    
    # is_cyclic: bool, if the variable is periodic, like angles
    
    # order: order of the spline, 4 is cubic
    
    # is_temporal_kernel: bool, if the varibleis temporal 
    # (compute convolution of the basis function x*b(x) as a tranformation)
    
    # kernel_direction: 0 acausal, 1 causal
    
    # penalty_type: derivative based regularization or difference based (works well for equispaced knots)
    
    # der: order of the derivative used in the penalization (der=2)
    
   
    if't_' in var or var == 'spike_hist':
        # skip events
        continue
    else:
        plt.subplot(3,4,cnt)
        plt.title(var)
        plt.hist(x_trans,bins=30,density=True)
        plt.plot(knots,[0]*knots.shape[0],'or')
        cnt+=1
    
    if include_var:

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                      is_cyclic=[is_cyclic], lam=50.,
                      penalty_type=penalty_type,
                      der=der,
                      trial_idx=trial_idx, time_bin=0.006,
                      is_temporal_kernel=is_temporal_kernel,
                      kernel_length=kernel_len, kernel_direction=kernel_direction,
                      repeat_extreme_knots=False)
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# create the poisson gamily with log link function
link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

# def class gam model passing the smooth_handler object and the spikes
gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, spikes[keep, fit_unit], poissFam,
                                    fisher_scoring=True)

# fit the full model and reduced (with only significant variables)
# standard settings
full, reduced = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001,
                                               method='L-BFGS-B', tol=1e-8,
                                               conv_criteria='gcv',
                                               max_iter=100, gcv_sel_tol=10 ** -13, 
                                               random_init=False,
                                               use_dgcv=True, initial_smooths_guess=False,
                                               fit_initial_beta=True, 
                                               pseudoR2_per_variable=True,
                                               trial_num_vec=trial_idx, 
                                               k_fold=False, fold_num=5,
                                               reducedAdaptive=False,compute_MI=False,
                                               perform_PQL=True)




# plot fit res (sorry for the code, i copy pasted it)
gam_res = full
var_list = gam_res.var_list

pvals = gam_res.covariate_significance['p-val']
covs = gam_res.covariate_significance['covariate']
fig = plt.figure(figsize=(14, 8))
plt.suptitle('%s - neuron %d  - PQL %d - %s %f' % ('m53s113', fit_unit ,True, 'all',1))
cc = 0
cc_plot = 1
for var in gam_res.var_list:
    

    if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
        cc += 1
        continue
    

    ax = plt.subplot(5, 4, cc_plot)
    # ax = plt.subplot(1, 1, cc_plot)

    if var == 'spike_hist':
        continue
    elif 'lfp' in var:
        min_x = -np.pi
        max_x = np.pi
        
    else:
        cc = np.where(var_names == var)[0][0]
        x = X[keep, cc]
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

        xx = np.linspace(min_x,max_x,100)

        fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)

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
    # plt.legend()
   

    plt.title(var + ' '+'sign: %s'%(pvals[covs==var]<0.001))
    cc += 1
    cc_plot += 1
        
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# save reults
# sv_folder = ''
# gam_res = {}
# gam_res['full'] = full
# gam_res['reduced'] = reduced

# with open(os.path.join(sv_folder, 'gam_fit_%s_c%d_%s_%.4f.dill' % (session, neuron, cond_type, cond_value)),
#               "wb") as dill_file:
#         dill_file.write(dill.dumps(gam_res))

