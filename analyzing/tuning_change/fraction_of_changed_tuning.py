#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:56:20 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import dill
import pandas as pd
import scipy.stats as sts
import scipy.linalg as linalg
from time import perf_counter
from seaborn import heatmap
from path_class import get_paths_class
from knots_constructor import knots_cerate
from copy import deepcopy
from time import sleep
from spline_basis_toolbox import *
from bisect import bisect_left
from statsmodels.distributions import ECDF
import venn
plt.close('all')


npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'
tuning_change_fld = '/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_change/significance_tuning_function_change/'
condition = 'controlgain'
ba = 'PFC'
session = 'm53s42'

dtype_dict = {'names':('session','condition','brain_area','variable','p-val'),
                                 'formats':('U30','U30','U30','U30',float)}
result_table = np.zeros(0,dtype=dtype_dict)


lst_files = os.listdir(tuning_change_fld)
pattern = '^newTest_m\d+s\d+_[a-z]+_tuningChange.npz$'
for fh in lst_files:
    if not re.match(pattern,fh):
        continue

    splt = fh.split('_')
    session = splt[1]
    condition = splt[2]
    print(session,condition)
    dat = np.load(os.path.join(tuning_change_fld,fh),
        allow_pickle=True)

    npz_dat = np.load(os.path.join(npz_folder,'%s.npz'%(session)),
        allow_pickle=True)

    unit_info = npz_dat['unit_info'].all()
    brain_area = unit_info['brain_area']

    # tensor_A = dat['tensor_A']
    # tensor_B = dat['tensor_B']
    # index_dict_A = dat['index_dict_A'].all()
    # index_dict_B = dat['index_dict_B'].all()


    var_sign = dat['var_sign']
    sele_non_coupling = np.zeros(var_sign.shape,dtype=bool)

    cc = 0

    var_vector = []
    for var in var_sign['variable']:
        if var.startswith('neu') or var.startswith('spike_hist') :#or var.startswith('lfp') :
            cc+=1
            continue
        sele_non_coupling[cc] = True

        if var in ['rad_vel','ang_vel']:
            var_vector += ['sensory']

        elif var in ['rad_acc','ang_acc']:
            var_vector += ['acceleration']

        elif var in ['t_stop','t_move']:
            var_vector += ['move ON/OFF']

        elif var in ['eye_hori','eye_vert']:
            var_vector += ['eye']

        elif var in ['ang_path','rad_path']:
            var_vector += ['dist orig']

        elif var in ['rad_target','ang_target']:
            var_vector += ['dist targ']
        else:
            var_vector += [var]
        cc+=1

    var_sign = var_sign[sele_non_coupling]



    # unit_list = dat['unit_list']

    tmp = np.zeros(var_sign.shape[0], dtype=dtype_dict)
    tmp['variable'] = var_vector#var_sign['variable']
    tmp['session'] = session
    tmp['condition'] = condition
    tmp['brain_area'] = brain_area[var_sign['unit']-1]
    tmp['p-val'] = var_sign['p-val']

    result_table = np.hstack((result_table,tmp))


condition_list = ['controlgain','ptb','density','odd']
var_list = ['sensory','acceleration', 'move ON/OFF','dist orig','dist targ','t_flyOFF', 't_reward','eye']
for ba in np.unique(result_table['brain_area']):
    res_ba = result_table[(result_table['brain_area'] == ba)]

    plt.figure(figsize=(10,6))
    plt.suptitle(ba)
    sbplt_cnt = 1
    for var in var_list:
        ax = plt.subplot(4,2,sbplt_cnt)
        sbplt_cnt+=1

        vec_mean_changed = []
        vec_cond = []
        for cond in condition_list:

            res_cond = res_ba[(res_ba['condition']==cond) *(res_ba['variable']==var)]
            if res_cond.shape[0] == 0:
                continue

            vec_mean_changed += [(res_cond['p-val']<0.001).mean()]
            vec_cond += [cond]

        plt.barh(range(len(vec_cond)),vec_mean_changed[::-1])
        plt.yticks(range(len(vec_cond)),vec_cond[::-1])
        plt.xlim(0,1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(var)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig('%s_tuningChange.pdf'%(ba))

        # df = pd.DataFrame(res_cond)
        # df['p-val'] = df['p-val'] < 0.001
        # data = df.groupby(["variable"])["p-val"].mean()
        # plt.figure()
        # plt.title(ba + ' ' + cond)
        # data.plot.pie()
# get the npz