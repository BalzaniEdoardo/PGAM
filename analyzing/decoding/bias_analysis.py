#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:48:30 2021

@author: edoardo
"""


import numpy as np
import matplotlib.pylab as plt
from statsmodels.distributions import ECDF
import scipy.stats as sts


var = 'rad_target'

decoding_results_x_var = np.load('predict_density_%s.npy'%var,allow_pickle=True).all()
# plt.figure(figsize=(12,6))
   # plt.suptitle()
means_dict_ba = {}

session_list = {}
for ba in ['PPC','PFC']:
    means_dict_ba[ba] = []
    session_list[ba] = list(decoding_results_x_var[ba].keys())

   
    cc = 1
    plt.figure(figsize=(12,5))
    plt.suptitle(ba+'\ndecoding (same condition - opposite condtion)')
    for session in  session_list[ba]:
        true_hd = decoding_results_x_var[ba][session]['true HD']
        true_ld = decoding_results_x_var[ba][session]['true LD']
        
        
        hd_m_ld = decoding_results_x_var[ba][session]['LD -> LD'] - decoding_results_x_var[ba][session]['HD -> LD']
        ld_m_hd = decoding_results_x_var[ba][session]['HD -> HD'] - decoding_results_x_var[ba][session]['LD -> HD']
        
        means_dict_ba[ba] += [np.nanmean(ld_m_hd)]
        # means_dict_ba[ba] += [np.nanmean(hd_m_ld)]
        
        plt.subplot(2,5,cc)
        hdmld_cdf = ECDF(hd_m_ld)
        ldmhd_cdf = ECDF(ld_m_hd)
        
        
        xx = np.linspace(-100,100,1000)
        
        p, = plt.plot(xx,hdmld_cdf(xx),label='LD test')
        # plt.plot(xx,hdld_cdf(xx),label='HD->LD',color=p.get_color(),ls='--')
        # q, = plt.plot(xx,ldld_cdf(xx),label='LD->LD')
        plt.plot(xx,ldmhd_cdf(xx),label='HD test')
        plt.plot([0,0],[0,1],'--r')
        plt.plot([-100,100],[0.5,0.5],'--k')
        if cc ==1:
            plt.legend(fontsize=8,loc=2,frameon=False)
        cc+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    plt.savefig('%s_bias_compare_cdf_%s.png'%(ba,var))


ba = 'PPC'


dat_beh = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/mean_bias_response.npz')
sess_behav = dat_beh['session_list']
mean_HD = dat_beh['mean_HD']
mean_LD = dat_beh['mean_LD']

kk=1
plt.figure(figsize=[8.21, 4.8 ])
for ba in ['PPC','PFC']:
    bias_behav = []
    bias_neur = []
    cc = 0
    for session in session_list[ba]:
        bl = sess_behav == session
        if sum(bl) == 0:
            continue
        bias_behav+= [ mean_HD[np.where(bl)[0][0]] - mean_LD[np.where(bl)[0][0]]]
        bias_neur += [means_dict_ba[ba][cc]]
        cc+=1
    plt.subplot(1,2,kk)
    corr,pval= sts.pearsonr(bias_behav,bias_neur)
    print(ba,corr,pval)
    plt.title('%s bias - corr : %.3f'%(ba,corr))
    plt.scatter(bias_behav,bias_neur)
    plt.xlabel('bias behavior [cm]')
    plt.ylabel('bias decoding [cm]')
    
    kk+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('%s_bias_compare_%s.png'%(ba,var))
# for session in decoding_results_x_var[ba].keys():
    
#     dat_npz = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,allow_pickle=True)
#     concat = dat_npz['data_concat'].all()
#     trial_ind  = concat['trial_idx']
#     var_names = dat_npz['var_names']
#     X = concat['Xt']
#     r_vel = X[trial_ind==3,var_names=='rad_vel']
#     theta_vel = X[trial_ind==3,var_names=='ang_vel']
#     rad_target = X[trial_ind==3,var_names=='rad_target']
#     dt = 0.006
    
    
#     break
    
    
