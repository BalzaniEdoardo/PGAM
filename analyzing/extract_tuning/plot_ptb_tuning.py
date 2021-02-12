#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:20:54 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import matplotlib.pylab as plt
import dill,sys,os
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library')
from seaborn import *
from GAM_library import *
from scipy.integrate import simps
from spectral_clustering import *
from basis_set_param_per_session import *
from spline_basis_toolbox import *
from scipy.cluster.hierarchy import linkage,dendrogram
import scipy.stats as sts
import statsmodels.api as sm

info_matrix = np.load('response_strength_info_ptb_regr.npy',allow_pickle=True)


info_ppc = info_matrix[info_matrix['brain_area'] == 'PPC']
info_pfc = info_matrix[info_matrix['brain_area'] == 'PFC']

print('PFC % ptb tuning:',info_pfc['t_ptb'].mean() )
print('PPC % ptb tuning:',info_ppc['t_ptb'].mean() )

plt.bar([0],[info_ppc['t_ptb'].mean()],width=0.4,color='b')

plt.bar([1],[info_pfc['t_ptb'].mean()],width=0.4,color='r')
plt.xticks([0,1],['PPC','PFC'],fontsize=15)
plt.title('tuning to perturbation',fontsize=20)
plt.ylim(0,0.8)
plt.ylabel('fraction tuned',fontsize=15)
plt.xlim(-0.5,1.5)
plt.tight_layout()
plt.savefig('ptb_frac_tuned.png')


sign_ptb_ppc = info_ppc[info_ppc['t_ptb']]

idx_sort = np.argsort(sign_ptb_ppc['t_ptb_resp_magn_full'])

var = 't_ptb'

cnt_plt = 0
plt.figure(figsize=(10,6.5))
plt.suptitle('PPC examples')
for k in range(0,20):
    
    if cnt_plt ==8:
        break
   
    inf_neu = sign_ptb_ppc[idx_sort[-k]]
    neu_1 = inf_neu['unit']
    if neu_1 in [81,119]:
        continue
    session_i = inf_neu['session']
    folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/gam_ptb_regressor/gam_%s/' % ( session_i)
    fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
    with open(folder + fhName, "rb") as dill_file:
        gam_res_dict = dill.load(dill_file)
    
    gam_res = gam_res_dict['full']
    dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
    ord_ = gam_res.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    
    idx_cut = np.where(impulse==1)[0][0]+1
    
    # xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99, trial_idx=None)
    
    fX = fX[idx_cut:]
    fX_p_ci = fX_p_ci[idx_cut:]
    fX_m_ci = fX_m_ci[idx_cut:]
    
    fX = fX[::-1]
    fX_p_ci = fX_p_ci[::-1]
    fX_m_ci = fX_m_ci[::-1]
    
   
    plt.subplot(2,4,cnt_plt+1)
    # plt.title('%d'%neu_1)
    xx = np.arange(0,fX.shape[0])*0.006
    plt.plot(xx, fX, ls='-', color='b', label='t_ptb')
    plt.fill_between(xx,fX_m_ci, fX_p_ci, color='b', alpha=0.4)
    if cnt_plt %4 == 0:
        plt.ylabel('kernel gain')
    if cnt_plt >= 4:
        plt.xlabel('time [sec]')
    cnt_plt += 1
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('PPC_ptb_tuning_example.png')
 

sign_ptb_pfc = info_pfc[info_pfc['t_ptb']]

idx_sort = np.argsort(sign_ptb_pfc['t_ptb_resp_magn_full'])

var = 't_ptb'

cnt_plt = 0
plt.figure(figsize=(10,6.5))
plt.suptitle('PFC examples')
for k in range(0,20):
    
    if cnt_plt ==8:
        break
   
    inf_neu = sign_ptb_pfc[idx_sort[-k]]
    neu_1 = inf_neu['unit']
    if neu_1 in [81,119]:
        continue
    session_i = inf_neu['session']
    folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/gam_ptb_regressor/gam_%s/' % ( session_i)
    fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
    with open(folder + fhName, "rb") as dill_file:
        gam_res_dict = dill.load(dill_file)
    
    gam_res = gam_res_dict['full']
    dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
    ord_ = gam_res.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    
    idx_cut = np.where(impulse==1)[0][0]+1
    
    # xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99, trial_idx=None)
    
    fX = fX[idx_cut:]
    fX_p_ci = fX_p_ci[idx_cut:]
    fX_m_ci = fX_m_ci[idx_cut:]
    
    fX = fX[::-1]
    fX_p_ci = fX_p_ci[::-1]
    fX_m_ci = fX_m_ci[::-1]
    
   
    plt.subplot(2,4,cnt_plt+1)
    # plt.title('%d'%neu_1)
    xx = np.arange(0,fX.shape[0])*0.006
    plt.plot(xx, fX, ls='-', color='r', label='t_ptb')
    plt.fill_between(xx,fX_m_ci, fX_p_ci, color='r', alpha=0.4)
    if cnt_plt %4 == 0:
        plt.ylabel('kernel gain')
    if cnt_plt >= 4:
        plt.xlabel('time [sec]')
    cnt_plt += 1
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('PFC_ptb_tuning_example.png')
 