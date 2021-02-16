#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:26:05 2021

@author: edoardo

The preocessing necessary in order to obtain a tuning function is:
    1) create the model matrix
    2) (*not mandatory, the kernel is defined up to a translation constant), impose sum(f(x))) = 0
        by mean shifting the model matrix
    3) remove the last column of the model matrix
    4) f(x) = <X, beta> as usual
    

"""

import numpy as np
import dill,os,sys
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)

# GAM_library contains all the main function and classes related to the GAM model
sys.path.append(os.path.join(main_dir,os.path.join('GAM_library')))
from GAM_library import *
# Some function that i find useful specifically for the firefly data
# (like the function for creating knots)
sys.path.append(os.path.join(main_dir,os.path.join('firefly_utils')))
from knots_constructor import knots_cerate
import scipy.sparse as sparse
import scipy.stats as sts
import matplotlib.pylab as plt

def model_matrix(X, var_name, smooth_info, trial_idx, pre_trial_dur=0,post_trial_dur=0,
                       time_bin=0.006):
    
    is_temporal = smooth_info[var_name]['is_temporal_kernel']
    ord_spline = smooth_info[var_name]['ord']
    is_cyclic = smooth_info[var_name]['is_cyclic']
    knots = smooth_info[var_name]['knots']
    basis_kernel = smooth_info[var_name]['basis_kernel']
    try:
        penalty_type = smooth_info[var_name]['penalty_type']
        xmin = smooth_info[var_name]['xmin']
        xmax = smooth_info[var_name]['xmax']
        der = smooth_info[var_name]['der']
    except KeyError:# old fits do ont have the key
        penalty_type = 'EqSpaced'
        xmin = None
        xmax = None
        der = None

    if not is_temporal:
        # this evaluates the basis spline at the variable X
        fX = basisAndPenalty(X, knots, is_cyclic=is_cyclic, ord=ord_spline,
                             penalty_type=penalty_type, xmin=xmin, xmax=xmax, 
                             der=der,compute_pen=False,domain_fun=None,
                             sparseX=False)[0]
    else:
        if type(basis_kernel) is sparse.csr.csr_matrix or \
             type(basis_kernel) is sparse.csr.csr_matrix:
            basis_kernel = basis_kernel.toarray()

        if trial_idx is None:
            pass
        # this convolves the basis function containined in basis_kernel with X
        fX = basis_temporal(X, basis_kernel,trial_idx,
                            pre_trial_dur,post_trial_dur, time_bin, sparseX=False)
        
    # preprocess by translating so that sum(f(x)) = 0, and remove one column
    nan_filter = np.array(np.sum(np.isnan(np.array(X)), axis=0), dtype=bool)
    # mean center and remove col if more than 1 smooth in the AM
    
    fX = np.array(fX[:, :-1] - np.mean(fX[~nan_filter, :-1], axis=0))
    fX[nan_filter,:] = 0
    
                
    return fX




    
    

session = 'm53s113'
unit = 22

with open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_all_1.0000.dill'%(session,session,unit),'rb') as fh:
    res_dict = dill.load(fh)
    
full = res_dict['full']
beta = full.beta
cov_beta = full.cov_beta
index_dict = full.index_dict

# for variables that we are not convolving, a vector over which evaluating the 
# basis
var = 'rad_vel'


x = np.linspace(0,180,1000)
indices = index_dict[var]
fX = model_matrix([x], var, full.smooth_info, trial_idx=None, pre_trial_dur=0,post_trial_dur=0,
                      time_bin=0.006)

# compute the dot prod X*beta and sqrt(diag(X * cov * X^T))
mean_kernel = np.dot(fX, beta[indices])
sd_kernel = np.sqrt(np.sum(np.dot(fX, cov_beta[indices, :][:, indices]) * fX, axis=1))

# confidence intervals
perc = 0.99
norm = sts.norm()
ci_kernel = sd_kernel * norm.ppf(1-(1-perc)*0.5)


plt.figure()
plt.subplot(321)
# plot the basis
# the translations are a result of mean centering,
# whhich is an operation X[:,k] - v[k], with v a vector. the dot product is
# will shift the kernel of  sum_k beta[k]  v[k], but in an additive model
# shift on the vertical axis are absorbed by a single intercept term


plt.plot(x, fX)
plt.subplot(322)
plt.plot(x,mean_kernel)
plt.fill_between(x, mean_kernel-ci_kernel, mean_kernel+ci_kernel, color='b',alpha=0.5)


# for variables that we want to convolve we need to pass an inpulse
# and the model matrix will be basis convolved with the impulse
var = 't_stop'

# length fof the kernel
kern_len = full.smooth_info['t_stop']['time_pt_for_kernel'].shape[0]
x = np.zeros((kern_len,))
x[kern_len//2] = 1

fX = model_matrix([x], var, full.smooth_info, trial_idx=None, pre_trial_dur=0,post_trial_dur=0,
                      time_bin=0.006)

plt.subplot(323)
# the true x axis is time
time = np.arange(kern_len)*0.006
time = time - time[kern_len//2]
plt.plot(time, fX)


indices = index_dict[var]
mean_kernel = np.dot(fX, beta[indices])
sd_kernel = np.sqrt(np.sum(np.dot(fX, cov_beta[indices, :][:, indices]) * fX, axis=1))
ci_kernel = sd_kernel * norm.ppf(1-(1-perc)*0.5)


plt.subplot(324)
# the true x axis is time
time = np.arange(kern_len)*0.006
time = time - time[kern_len//2]
plt.plot(time,  mean_kernel)
plt.fill_between(time, mean_kernel-ci_kernel, mean_kernel+ci_kernel, color='b',alpha=0.5)



# causal temporal filters have create a causal convolution kenel

var = 'spike_hist'
x = np.zeros((11,))
x[5] = 1

fX = model_matrix([x], var, full.smooth_info, trial_idx=None, pre_trial_dur=0,post_trial_dur=0,
                      time_bin=0.006)

plt.subplot(325)
# convolution flips the kernel so that only things that happen
# before the impulse contributes to the rate
time = np.arange(11)*0.006
time = time - time[5]
plt.plot(time, fX)





    
