#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:05:38 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(thisPath)),'GAM_Library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF,monotone_fn_inverter
import matplotlib.pylab as plt
from basis_set_param_per_session import *


# plot input distribution for all session and save the histograms
bs_fold = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET/'

first = False
pattern = '^m\d+s\d+.npz$'

hist_size = 400
hist_matrix = np.zeros((0,hist_size),dtype=int)
edge_matrix = np.zeros((0,hist_size+1),dtype=np.float32)
info = np.zeros(0,dtype={'names':('session','variable'),'formats':('U20','U20')})


for root, dirs, names in os.walk(bs_fold):
    for name in names:
        if not re.match('^m44s207.npz$', name):
            continue
        dat = np.load(os.path.join(root,name), allow_pickle=True)
        session = name.split('.npz')[0]
        
        var_names = dat['var_names']
        concat = dat['data_concat'].all()
        Xt = concat['Xt']
        
        cnt_var = 0
        for var in var_names:
            if var.startswith('t_') or var.startswith('lfp_'):
                continue
            cnt_var += 1
        
        tmp_info = np.zeros(cnt_var, dtype={'names':('session','variable'),'formats':('U20','U20')})
        tmp_hist = np.zeros((cnt_var, hist_size))
        tmp_edge = np.zeros((cnt_var, hist_size+1))

        cc = 0
        for var in var_names:
            if var.startswith('t_') or var.startswith('lfp_'):
                continue
            if not 'eye_vert' in var:
                continue
            idx = np.where(var_names == var)[0][0]
            xx = Xt[:,idx]
            
            xx = xx[~np.isnan(xx)]
            if 'eye' in var:
                xx = xx[np.abs(xx)<np.nanpercentile(np.abs(xx),99)]
            else:
                continue
            if var == 'eye_vert':
                ecdf_eyevert = ECDF(xx)
                inv_ecdf_eyvert = monotone_fn_inverter(ecdf_eyevert,xx)


found = False    
for root, dirs, names in os.walk(bs_fold):
    for name in names:
        if not re.match(pattern, name):
            continue
        print(name)
        dat = np.load(os.path.join(root,name), allow_pickle=True)
        session = name.split('.npz')[0]
        
        var_names = dat['var_names']
        concat = dat['data_concat'].all()
        Xt_other = concat['Xt']
        
        cnt_var = 0
        for var in var_names:
            if var.startswith('t_') or var.startswith('lfp_'):
                continue
            cnt_var += 1
        
        tmp_info = np.zeros(cnt_var, dtype={'names':('session','variable'),'formats':('U20','U20')})
        tmp_hist = np.zeros((cnt_var, hist_size))
        tmp_edge = np.zeros((cnt_var, hist_size+1))

        cc = 0
        for var in var_names:
            if var.startswith('t_') or var.startswith('lfp_'):
                continue
            if var != 'eye_vert':
                continue
            idx = np.where(var_names == var)[0][0]
            xx_other = Xt_other[:,idx]
            filt_nan = ~np.isnan(xx_other)
            xx_other = xx_other[~np.isnan(xx_other)]
            
            if 'eye' in var:
                xx_other_perc = xx_other[np.abs(xx_other)<np.nanpercentile(np.abs(xx_other),99)]
            else:
                continue
            tmp_hist[cc,:],tmp_edge[cc,:] = np.histogram(xx_other_perc,bins=hist_size)
            tmp_info[cc]['variable'] = var
            tmp_info[cc]['session'] = session
            if var =='eye_vert':
                 ecdf = ECDF(xx_other_perc)  
                 inv_ecdf = monotone_fn_inverter(ecdf,xx_other)
                 
                 # plt.plot(ecdf.x,ecdf.y)
            cc+=1
            
        hist_matrix = np.vstack((hist_matrix, tmp_hist))
        edge_matrix = np.vstack((edge_matrix, tmp_edge))
        info = np.hstack((info,tmp_info))
        found = True
        break
    if found:
        break
    

# transform input 2 and plot some GAM fit
# match the distr histogram
xx_trans = inv_ecdf_eyvert(ecdf(xx_other))
xx_trans[np.abs(xx_other)>np.nanpercentile(np.abs(xx_other),99)] = np.nan
sm_handler = smooths_handler()

var = 'spike_hist'
order = basis_info[session][var]['order']
penalty_type = basis_info[session][var]['penalty_type']
der = basis_info[session][var]['der']
is_temporal_kernel = basis_info[session][var]['knots_type'] == 'temporal'
kernel_length = basis_info[session][var]['kernel_length']
kernel_direction = basis_info[session][var]['kernel_direction']


all_index = concat['trial_idx'][filt_nan]
trial_type = dat['info_trial'].all().trial_type
keep = []
idx_subselect = np.where(trial_type['all'] == 1)[0]

for ii in idx_subselect:
    keep = np.hstack((keep, np.where(all_index == ii)[0]))
    if np.sum(all_index == ii) == 0:
        raise ValueError
keep = np.array(keep,dtype=int)
yt = concat['Yt']
tmpy = yt[keep,1]
x = np.hstack(([0],tmpy[:-1]))

knots = knots_by_session(x, session, var, basis_info)


sm_handler.add_smooth(var, [x], ord=order, knots=[knots], knots_num=None, perc_out_range=None,
                      is_cyclic=[False], lam=None, penalty_type=penalty_type, der=der,
                      knots_percentiles=None)


            
knots = np.hstack(([-15]*3,np.linspace(-15,10,7),[10]*3))
xx_trans[(xx_trans > 10) | (xx_trans < -15)] = np.nan

sm_handler.add_smooth('cdf transf', [xx_trans], ord=4, knots=[knots], knots_num=None, perc_out_range=None,
                      is_cyclic=[False], lam=None, penalty_type='der', der=2,
                      knots_percentiles=None)

link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler,['cdf transf','spike_hist'],yt[keep,1], poissFam,fisher_scoring=False)

trial_IDX = all_index[keep]

t0 = perf_counter()
full,reduced = gam_model.fit_full_and_reduced(['cdf transf','spike_hist'], th_pval=0.001,method = 'L-BFGS-B',tol=1e-8,conv_criteria='gcv',
                     max_iter=10000,gcv_sel_tol=10**-13,random_init=False,
                     use_dgcv=True,initial_smooths_guess=False,smooth_pen=[50.]*2+[50.]*1,
                     fit_initial_beta=True,pseudoR2_per_variable=True,trial_num_vec=trial_IDX,k_fold = True,fold_num=5)
t1 = perf_counter()
print(t1-t0)


xx = np.linspace(knots[0],knots[-1],100)
fX, fX_p_ci, fX_m_ci = full.smooth_compute([xx], 'cdf transf', perc=0.99)
xx_tr = inv_ecdf(ecdf_eyevert(xx))
  





knots_oth = inv_ecdf(ecdf_eyevert(knots))#np.hstack(([xx_tr.min()]*3,np.linspace(xx_tr.min(),xx_tr.max(),7),[xx_tr.max()]*3))
xx_oth = deepcopy(xx_other)
xx_oth[(xx_other > xx_tr.max()) | (xx_other < xx_tr.min())] = np.nan

sm_handler.add_smooth('no transf', [xx_oth], ord=4, knots=[knots_oth], knots_num=None, perc_out_range=None,
                      is_cyclic=[False], lam=None, penalty_type='der', der=2,
                      knots_percentiles=None)

link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler,['no transf','spike_hist'],yt[keep,1], poissFam,fisher_scoring=False)

trial_IDX = all_index[keep]

t0 = perf_counter()
full_oth,reduced_oth = gam_model.fit_full_and_reduced(['cdf transf','spike_hist'], th_pval=0.001,method = 'L-BFGS-B',tol=1e-8,conv_criteria='gcv',
                     max_iter=10000,gcv_sel_tol=10**-13,random_init=False,
                     use_dgcv=True,initial_smooths_guess=False,smooth_pen=[50.]*2+[50.]*1,
                     fit_initial_beta=True,pseudoR2_per_variable=True,trial_num_vec=trial_IDX,k_fold = True,fold_num=5)
t1 = perf_counter()
print(t1-t0)








xx_oth = np.linspace(knots_oth[0],knots_oth[-1],100)
fX_oth, fX_p_ci_oth, fX_m_ci_oth = full_oth.smooth_compute([xx_oth], 'cdf transf', perc=0.99)

plt.figure()
p, = plt.plot(xx_tr,fX)
plt.plot(xx_tr,fX_m_ci,'--',color=p.get_color())
plt.plot(xx_tr,fX_p_ci,'--',color=p.get_color())


p, = plt.plot(xx,fX_oth)
plt.plot(xx,fX_m_ci_oth,'--',color=p.get_color())
plt.plot(xx,fX_p_ci_oth,'--',color=p.get_color())
# xx_tr = inv_ecdf(ecdf_eyevert(xx))
  