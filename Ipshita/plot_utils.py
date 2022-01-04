#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:02:18 2021

@author: edoardo
"""
from scipy.io import loadmat
import numpy as np
import sys
sys.path.append('../GAM_library')
sys.path.append('../firefly_utils')
from GAM_library import *
from data_handler import *
import scipy.stats as sts
from copy import deepcopy
import matplotlib.pylab as plt
from fit_utils import construct_knots_medSpatialDensity,\
    construct_knots_highSpatialDensity,compute_tuning,\
        construct_knots_lowSpatialDensity


    


def plot_results_Hz(spk, fit, sm_handler , var_list, filter_trials, ax_dict=None):
    
    exog, _ = sm_handler.get_exog_mat(fit.var_list)
    numRows = len(var_list)//5 + 1
    numCols = 5

    cc = 0
    if ax_dict is None:
        fig = plt.figure(figsize=(13.5,5))
        ax_dict = {}
        
    else:
        fig = None

    for var in var_list:
        if var == 'spike_hist':
            continue

        if not var in ax_dict.keys():
            ax = plt.subplot(numRows,numCols,cc+1)
        else:
            ax = ax_dict[var]
        x, fX, fXraw = compute_tuning(spk, fit, exog, var, sm_handler, filter_trials)
        # fX = full_fit.tuning_Hz.__dict__[var].__dict__['y_model']
        # fXraw = full_fit.tuning_Hz.__dict__[var].__dict__['y_raw']

        # x = fit.tuning_Hz.__dict__[var].__dict__['x']

        p, = ax.plot(x,fX,'r')
        p, = ax.plot(x,fXraw,'k')
        # ax.fill_between(xx2,fminus,fplus,color = p.get_color(),alpha=0.5)
        ax_dict[var] = ax
        ax.set_title(var)
        cc += 1

    plt.tight_layout()
    return ax_dict,fig

def plot_results(full_fit , var_list, ax_dict=None):
    numRows = len(var_list)//5 + 1
    numCols = 5

    cc = 0
    if ax_dict is None:
        fig = plt.figure(figsize=(13.5,5))
        ax_dict = {}
    else:
        fig = None

    for var in var_list:
        # compute kernel strength
        if full_fit.smooth_info[var]['is_temporal_kernel']:
            dim_kern = full_fit.smooth_info[var]['basis_kernel'].shape[0]
            x = np.zeros(dim_kern)
            x[(dim_kern - 1) // 2] = 1
            xx2 = np.arange(x.shape[0]) * 6 - np.where(x)[0][0] * 6
            fX, fminus, fplus = full_fit.smooth_compute([x], var, 0.99)
            if (var == 'spike_hist') or ('neu_') in var:
                fminus = fminus[(dim_kern - 1) // 2:] - fX[0]
                fplus = fplus[(dim_kern - 1) // 2:] - fX[0]
                fX = fX[(dim_kern - 1) // 2:] - fX[0]
                xx2 = xx2[(dim_kern - 1) // 2:]
            else:
                fplus = fplus - fX[-1]
                fminus = fminus - fX[-1]
                fX = fX - fX[-1]
        else:
            knots = full_fit.smooth_info[var]['knots']
            xmin = knots[0].min()
            xmax = knots[0].max()
            xx2 = np.linspace(xmin, xmax, 100)
            fX, fminus, fplus = full_fit.smooth_compute([xx2], var, 0.99)
        if not var in ax_dict.keys():
            ax = plt.subplot(numRows,numCols,cc+1)
        else:
            ax = ax_dict[var]
        p, = ax.plot(xx2,fX)
        ax.fill_between(xx2,fminus,fplus,color = p.get_color(),alpha=0.5)
        ax_dict[var] = ax
        ax.set_title(var)
        cc += 1

    plt.tight_layout()
    return ax_dict,fig
        
def plot_basis(varName, trial_idx, var_dict, dat, ax1=None,knotsCons='med'):

    found = False
    for varType in var_dict.keys():
        if varName in var_dict[varType]:
            found = True
            break
    assert(found)
    if knotsCons == 'med':
        func = construct_knots_medSpatialDensity
    elif knotsCons == 'low':
        func = construct_knots_lowSpatialDensity
    else:
        func = construct_knots_highSpatialDensity
    knots, x, is_cyclic, order, \
    kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
        func(dat, varType, varName, neuNum=0, portNum=0, history_filt_len=199)

    xx = np.linspace(knots[0],knots[-1],100)
    sm_handler = smooths_handler()
    sm_handler.add_smooth(varName, [xx], ord=order, knots=[knots],
                          is_cyclic=[is_cyclic], lam=50,
                          penalty_type=penalty_type,
                          der=der,
                          trial_idx=trial_idx, time_bin=0.006,
                          is_temporal_kernel=is_temporal_kernel,
                          kernel_length=kernel_len,
                          kernel_direction=kernel_direction)
    if ax1 is None:
        ax1 = plt.subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.hist(x,bins=40,density=True,alpha=0.5)
    ax1.set_ylabel('density')
    ax2.plot(xx, sm_handler[varName].X.toarray())
    ax2.plot(knots, [0]*len(knots),'or')
    ax2.set_ylim(0,1.5)
    ax2.set_ylabel('log-rate')
    ax1.set_xlabel(varName)
