#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:16:59 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from statsmodels.distributions.empirical_distribution import ECDF
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
# from spline_basis_toolbox import *
from GAM_library import *
import dill


dat = np.load('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/fullFit_coupling_results.npy',
        allow_pickle=True)


monkey = 'Schro'

ele_dist = dat['electrode distance']


plt.figure()
plt.suptitle('Fraction coupled x distance')
color_dict = {'MST':'g','PPC':'b','PFC':'r'}
for (ba_sender,ba_receiver) in [('MST', 'MST'),('PPC', 'PPC'),('PFC', 'PFC')]:
    # ba_sender = 'MST'
    # ba_receiver = 'MST'
    
    filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey)
            )
    distance_vec = dat['electrode distance'][filt]
    is_sign = dat['is significant'][filt]
    
    
    dist_level = np.unique(distance_vec)
    freq = np.zeros(dist_level.shape)
    cc = 0
    for dd in dist_level:
        sel = distance_vec == dd
        freq[cc] = is_sign[sel].sum() / sel.sum()
        cc+=1
    
    plt.plot(dist_level,freq,marker='o',color=color_dict[ba_sender],label=ba_sender)
plt.xlabel('distamce (um)')
plt.ylabel('coupling probability')
plt.legend()


plt.figure()
plt.suptitle('Coupling strength x distance')
color_dict = {'MST':'g','PPC':'b','PFC':'r'}
for (ba_sender,ba_receiver) in [('MST', 'MST'),('PPC', 'PPC'),('PFC', 'PFC')]:
    # ba_sender = 'MST'
    # ba_receiver = 'MST'
    
    filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey)
            )
    distance_vec = dat['electrode distance'][filt]
    is_sign = dat['coupling strength'][filt]
    
    
    dist_level = np.unique(distance_vec)
    freq = np.zeros(dist_level.shape)
    cc = 0
    for dd in dist_level:
        sel = distance_vec == dd
        freq[cc] = is_sign[sel].mean()
        cc+=1
    
    plt.plot(dist_level,freq,marker='o',color=color_dict[ba_sender],label=ba_sender)
plt.xlabel('distamce (um)')
plt.ylabel('coupling strength')
plt.legend()

# select some significant coupling cross-area


plt.figure()

# MST -> PFC
ba_sender = 'MST'
ba_receiver = 'PFC'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]

coup_str = np.sort(betw_dat['coupling strength'])
cs_cdf = ECDF(coup_str)


xx = np.linspace(coup_str[0], coup_str[-1],100)
plt.plot(xx,cs_cdf(xx),label='MST->PFC')

# MST -> PPC
ba_sender = 'MST'
ba_receiver = 'PPC'

filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]

coup_str = np.sort(betw_dat['coupling strength'])
cs_cdf = ECDF(coup_str)
plt.plot(xx,cs_cdf(xx),label='MST->PPC')
plt.legend()


# PPC -> MST
ba_sender = 'PPC'
ba_receiver = 'MST'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]

coup_str = np.sort(betw_dat['coupling strength'])
cs_cdf = ECDF(coup_str)


xx = np.linspace(coup_str[0], coup_str[-1],100)
plt.plot(xx,cs_cdf(xx),label='PPC->MST')

# PPC -> PFC
ba_sender = 'PPC'
ba_receiver = 'PFC'
filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]

coup_str = np.sort(betw_dat['coupling strength'])
cs_cdf = ECDF(coup_str)
plt.plot(xx,cs_cdf(xx),label='PPC->PFC')

# PFC - >MST
ba_sender = 'PFC'
ba_receiver = 'MST'
filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]

coup_str = np.sort(betw_dat['coupling strength'])
cs_cdf = ECDF(coup_str)
plt.plot(xx,cs_cdf(xx),label='PFC->MST')

# pfc -> ppc
ba_sender = 'PFC'
ba_receiver = 'PPC'
filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
# xx = np.linspace(coup_str[0], coup_str[-1],100)

coup_str = np.sort(betw_dat['coupling strength'])
cs_cdf = ECDF(coup_str)
plt.plot(xx,cs_cdf(xx),label='PFC->PPC')




# # within area
# ba_sender = 'MST'
# ba_receiver = 'MST'
# # xx = np.linspace(coup_str[0], coup_str[-1],100)

# filt = ((dat['brain area receiver'] == ba_receiver ) & 
#             (dat['brain area sender'] == ba_sender) &
#             (dat['monkey'] == monkey) &
#             (dat['is significant'])
#             )
# betw_dat = dat[filt]

# coup_str = np.sort(betw_dat['coupling strength'])
# xx = np.linspace(coup_str[0], coup_str[-1],100)

# cs_cdf = ECDF(coup_str)
# plt.plot(xx,cs_cdf(xx),label='MST->MST')

# ba_sender = 'PPC'
# ba_receiver = 'PPC'
# filt = ((dat['brain area receiver'] == ba_receiver ) & 
#             (dat['brain area sender'] == ba_sender) &
#             (dat['monkey'] == monkey) &
#             (dat['is significant'])
#             )
# betw_dat = dat[filt]

# coup_str = np.sort(betw_dat['coupling strength'])
# xx = np.linspace(coup_str[0], coup_str[-1],100)

# cs_cdf = ECDF(coup_str)
# plt.plot(xx,cs_cdf(xx),label='PPC->PPC')

# ba_sender = 'PFC'
# ba_receiver = 'PFC'
# filt = ((dat['brain area receiver'] == ba_receiver ) & 
#             (dat['brain area sender'] == ba_sender) &
#             (dat['monkey'] == monkey) &
#             (dat['is significant'])
#             )
# betw_dat = dat[filt]

# coup_str = np.sort(betw_dat['coupling strength'])
# xx = np.linspace(coup_str[0], coup_str[-1],100)

# cs_cdf = ECDF(coup_str)
# plt.plot(xx,cs_cdf(xx),label='PFC->PFC')





plt.legend()
plt.title('between area coupling strength')
plt.xlabel('coupling strength')
plt.ylabel('cdf')





# =============================================================================
# MST -> PFC
# =============================================================================
ba_sender = 'MST'
ba_receiver = 'PFC'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
idx = np.argsort(betw_dat['coupling strength'])

plt.figure(figsize=(12,8))
for kk in range(1,26):
    
    unit_rec = betw_dat[idx[-kk]]['unit receiver']
    unit_sen = betw_dat[idx[-kk]]['unit sender']
    session = betw_dat[idx[-kk]]['session']
    cond = 'all'
    value = 1
    
    
    plt.subplot(5,5,kk)
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    with open(os.path.join(fld_dill%session,'fit_results_%s_c%d_%s_%.4f.dill'%(session,unit_rec,cond,value)),'rb') as fh:
        res_dict = dill.load(fh)
    
    full_gam = res_dict['full']
    
    
    var = 'neu_%d'%unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99, trial_idx=None)# full_gam.['neu_%d'%unit_sen]
    
    idx  = np.where(xx > 0)[0]
    pp, = plt.plot(-xx[idx][::-1],fX[idx][::-1])
    plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1],color=pp.get_color(),alpha=0.4)
    
    if kk >20:
        plt.xlabel('time [sec]')
    
    if kk % 5 == 1:
        plt.ylabel('gain')
plt.suptitle('MST->PFC coupling filter examples')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_filt_example/mst_to_pfc_couplingFilt.png')





# =============================================================================
# MST -> PPC
# =============================================================================
ba_sender = 'MST'
ba_receiver = 'PPC'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
idx = np.argsort(betw_dat['coupling strength'])

plt.figure(figsize=(12,8))
for kk in range(1,26):
    
    unit_rec = betw_dat[idx[-kk]]['unit receiver']
    unit_sen = betw_dat[idx[-kk]]['unit sender']
    session = betw_dat[idx[-kk]]['session']
    cond = 'all'
    value = 1
    
    
    plt.subplot(5,5,kk)
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    with open(os.path.join(fld_dill%session,'fit_results_%s_c%d_%s_%.4f.dill'%(session,unit_rec,cond,value)),'rb') as fh:
        res_dict = dill.load(fh)
    
    full_gam = res_dict['full']
    
    
    var = 'neu_%d'%unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99, trial_idx=None)# full_gam.['neu_%d'%unit_sen]
    
    idx  = np.where(xx > 0)[0]
    pp, = plt.plot(-xx[idx][::-1],fX[idx][::-1])
    plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1],color=pp.get_color(),alpha=0.4)
    
    if kk >20:
        plt.xlabel('time [sec]')
    
    if kk % 5 == 1:
        plt.ylabel('gain')
plt.suptitle('MST->PPC coupling filter examples')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_filt_example/mst_to_ppc_couplingFilt.png')

# =============================================================================
# PFC -> MST
# =============================================================================
ba_sender = 'PFC'
ba_receiver = 'MST'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
idx = np.argsort(betw_dat['coupling strength'])

plt.figure(figsize=(12,8))
for kk in range(1,26):
    
    unit_rec = betw_dat[idx[-kk]]['unit receiver']
    unit_sen = betw_dat[idx[-kk]]['unit sender']
    session = betw_dat[idx[-kk]]['session']
    cond = 'all'
    value = 1
    
    
    plt.subplot(5,5,kk)
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    with open(os.path.join(fld_dill%session,'fit_results_%s_c%d_%s_%.4f.dill'%(session,unit_rec,cond,value)),'rb') as fh:
        res_dict = dill.load(fh)
    
    full_gam = res_dict['full']
    
    
    var = 'neu_%d'%unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99, trial_idx=None)# full_gam.['neu_%d'%unit_sen]
    
    idx  = np.where(xx > 0)[0]
    pp, = plt.plot(-xx[idx][::-1],fX[idx][::-1])
    plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1],color=pp.get_color(),alpha=0.4)
    
    if kk >20:
        plt.xlabel('time [sec]')
    
    if kk % 5 == 1:
        plt.ylabel('gain')
plt.suptitle('PFC->MST coupling filter examples')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_filt_example/pfc_to_mst_couplingFilt.png')

# =============================================================================
# PFC -> PPC
# =============================================================================
ba_sender = 'PFC'
ba_receiver = 'PPC'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
idx = np.argsort(betw_dat['coupling strength'])

plt.figure(figsize=(12,8))
for kk in range(1,26):
    
    unit_rec = betw_dat[idx[-kk]]['unit receiver']
    unit_sen = betw_dat[idx[-kk]]['unit sender']
    session = betw_dat[idx[-kk]]['session']
    cond = 'all'
    value = 1
    
    
    plt.subplot(5,5,kk)
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    with open(os.path.join(fld_dill%session,'fit_results_%s_c%d_%s_%.4f.dill'%(session,unit_rec,cond,value)),'rb') as fh:
        res_dict = dill.load(fh)
    
    full_gam = res_dict['full']
    
    
    var = 'neu_%d'%unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99, trial_idx=None)# full_gam.['neu_%d'%unit_sen]
    
    idx  = np.where(xx > 0)[0]
    pp, = plt.plot(-xx[idx][::-1],fX[idx][::-1])
    plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1],color=pp.get_color(),alpha=0.4)
    
    if kk >20:
        plt.xlabel('time [sec]')
    
    if kk % 5 == 1:
        plt.ylabel('gain')
plt.suptitle('PFC->PPC coupling filter examples')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_filt_example/pfc_to_ppc_couplingFilt.png')

# =============================================================================
# PPC -> MST
# =============================================================================
ba_sender = 'PPC'
ba_receiver = 'MST'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
idx = np.argsort(betw_dat['coupling strength'])

plt.figure(figsize=(12,8))
for kk in range(1,26):
    
    unit_rec = betw_dat[idx[-kk]]['unit receiver']
    unit_sen = betw_dat[idx[-kk]]['unit sender']
    session = betw_dat[idx[-kk]]['session']
    cond = 'all'
    value = 1
    
    
    plt.subplot(5,5,kk)
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    with open(os.path.join(fld_dill%session,'fit_results_%s_c%d_%s_%.4f.dill'%(session,unit_rec,cond,value)),'rb') as fh:
        res_dict = dill.load(fh)
    
    full_gam = res_dict['full']
    
    
    var = 'neu_%d'%unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99, trial_idx=None)# full_gam.['neu_%d'%unit_sen]
    
    idx  = np.where(xx > 0)[0]
    pp, = plt.plot(-xx[idx][::-1],fX[idx][::-1])
    plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1],color=pp.get_color(),alpha=0.4)
    
    if kk >20:
        plt.xlabel('time [sec]')
    
    if kk % 5 == 1:
        plt.ylabel('gain')
plt.suptitle('PPC->MST coupling filter examples')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_filt_example/ppc_to_mst_couplingFilt.png')

# =============================================================================
# PPC -> PFC
# =============================================================================
ba_sender = 'PPC'
ba_receiver = 'PFC'



filt = ((dat['brain area receiver'] == ba_receiver ) & 
            (dat['brain area sender'] == ba_sender) &
            (dat['monkey'] == monkey) &
            (dat['is significant'])
            )
betw_dat = dat[filt]
idx = np.argsort(betw_dat['coupling strength'])

plt.figure(figsize=(12,8))
for kk in range(1,26):
    
    unit_rec = betw_dat[idx[-kk]]['unit receiver']
    unit_sen = betw_dat[idx[-kk]]['unit sender']
    session = betw_dat[idx[-kk]]['session']
    cond = 'all'
    value = 1
    
    
    plt.subplot(5,5,kk)
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    with open(os.path.join(fld_dill%session,'fit_results_%s_c%d_%s_%.4f.dill'%(session,unit_rec,cond,value)),'rb') as fh:
        res_dict = dill.load(fh)
    
    full_gam = res_dict['full']
    
    
    var = 'neu_%d'%unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99, trial_idx=None)# full_gam.['neu_%d'%unit_sen]
    
    idx  = np.where(xx > 0)[0]
    pp, = plt.plot(-xx[idx][::-1],fX[idx][::-1])
    plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1],color=pp.get_color(),alpha=0.4)
    
    if kk >20:
        plt.xlabel('time [sec]')
    
    if kk % 5 == 1:
        plt.ylabel('gain')
plt.suptitle('PPC->PFC coupling filter examples')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_filt_example/ppc_to_pfc_couplingFilt.png')




