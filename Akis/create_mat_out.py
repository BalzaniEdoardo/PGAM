import numpy as np
import matplotlib.pylab as plt
import dill,os,re,sys
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import scipy.stats as sts
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import seaborn as sbs
from scipy.io import savemat
from scipy.integrate import simps

from seaborn.algorithms import bootstrap
import statsmodels.api as stats
import statsmodels.formula.api as smf
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from GAM_library import *
fh_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/fit_longFilters/gam_m53s91/fit_results_m53s91_c2_all_1.0000.dill'
with open(fh_path,'rb') as fh:
    res = dill.load(fh)



print('LINEAR REGRESSION START')
dtype_dict = {'names':('monkey','session','condition_type','condition_value','unit','brain_area','pseudo_r2','variable','pval','mutual_info', 'x',
                       'model_rate_Hz','raw_rate_Hz','kernel_strength','signed_kernel_strength'),
              'formats':('U30','U30','U30',float,int,'U30',float,'U30',float,float,object,object,object)}

full = res['full']
regr_res = np.zeros(len((full.var_list)),dtype=dtype_dict)
cs_table = full.covariate_significance
for cc in range(len(full.var_list)):

    var = full.var_list[cc]
    cs_var = cs_table[cs_table['covariate'] == var]
    regr_res['brain_area'][cc] = res['brain_area']
    regr_res['monkey'][cc] = 'Schro'
    regr_res['session'][cc] = 'm53s51'
    regr_res['unit'][cc] = 1
    regr_res['variable'][cc] = var
    regr_res['condition_type'][cc] = 'all'
    regr_res['condition_value'][cc] = 1
    regr_res['pseudo_r2'][cc] = res['p_r2_coupling_full']
    regr_res['pval'][cc] = cs_var['p-val']
    if var in full.mutual_info.keys():
        regr_res['mutual_info'][cc] = full.mutual_info[var]
    else:
        regr_res['mutual_info'][cc] = np.nan
    if var in full.tuning_Hz.__dict__.keys():
        regr_res['x'][cc] = full.tuning_Hz.__dict__[var].x
        regr_res['model_rate_Hz'][cc] = full.tuning_Hz.__dict__[var].y_model
        regr_res['raw_rate_Hz'][cc] = full.tuning_Hz.__dict__[var].y_raw

    # compute kernel strength
    if full.smooth_info[var]['is_temporal_kernel']:
        dim_kern = full.smooth_info[var]['basis_kernel'].shape[0]
        knots_num = full.smooth_info[var]['knots'][0].shape[0]
        x = np.zeros(dim_kern)
        x[(dim_kern - 1) // 2] = 1
        fX = full.smooth_compute([x], var, 0.95)[0]
        if (var == 'spike_hist') or ('neu_') in var:
            fX = fX[(dim_kern - 1) // 2:] - fX[0]
        else:
            fX = fX - fX[-1]
        regr_res['kernel_strength'][cc] = simps(fX**2, dx=0.006)
        regr_res['signed_kernel_strength'][cc] = simps(fX,dx=0.006)

    else:
        knots = full.smooth_info[var]['knots']
        xmin = knots[0].min()
        xmax = knots[0].max()
        func = lambda x: (full.smooth_compute([x],var,0.95)[0] - full.smooth_compute([x],var,0.95)[0].mean() )**2
        xx = np.linspace(xmin,xmax,500)
        dx = xx[1] - xx[0]
        regr_res['kernel_strength'][cc] = simps(func(xx),dx=dx)


savemat('tuning_regression_res.mat',mdict={'regression':regr_res})