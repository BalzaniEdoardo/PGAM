#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:10:19 2021

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
path_gen = get_paths_class()
from numba import njit

fold = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_m53s113/'
pattern = '^fit_results_m53s113_c\d+_all_1.0000.dill$'


# full model params



cnt_fits = 0
for fh in os.listdir(fold):
    if not re.match(pattern, fh):
        continue
    cnt_fits += 1

fit_type = 'full'
mean_beta = np.zeros(cnt_fits,dtype=object)
cov_beta = np.zeros(cnt_fits,dtype=object)
cov_sigificance = {}
smooth_info = {}
unit_id = np.zeros(68,dtype=int)

cc = 0
for fh in os.listdir(fold):
    if not re.match(pattern, fh):
        continue
    
    with open(os.path.join(fold, fh), 'rb') as fit:
        gam_res = dill.load(fit)
    
    neu = int(fh.split('_c')[1].split('_')[0])-1
    gam_res = gam_res[fit_type]
    
    if gam_res is None:
        cc += 1
        continue
    
    mean_beta[cc] = gam_res.beta
    cov_beta[cc] = gam_res.cov_beta
    cov_sigificance[cc] = gam_res.covariate_significance
    smooth_info[cc] = gam_res.smooth_info
    unit_id[cc] = neu
    
    cc+=1
    

np.savez('fit_results_m53s113.npz',mean_beta=mean_beta,cov_beta=cov_beta,
         cov_sigificance=cov_sigificance,smooth_info=smooth_info,
         unit_id=unit_id)
         
    