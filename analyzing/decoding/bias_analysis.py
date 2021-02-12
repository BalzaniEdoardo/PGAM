#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:48:30 2021

@author: edoardo
"""


import numpy as np
import matplotlib.pylab as plt
from statsmodels.distributions import ECDF

var = 'rad_path'

decoding_results_x_var = np.load('predict_density_%s.npy'%var,allow_pickle=True).all()
# plt.figure(figsize=(12,6))
   # plt.suptitle()
means_dict_hdhd = {}
means_dict_ldld = {}

for ba in ['PPC','PFC']:
    means_dict_hdhd[ba] = []
    means_dict_ldld[ba] = []
    cc = 1
    plt.figure(figsize=(12,5))
    plt.suptitle(ba)
    for session in  decoding_results_x_var[ba].keys():
        true_hd = decoding_results_x_var[ba][session]['true HD']
        true_ld = decoding_results_x_var[ba][session]['true LD']
        
        
        hd_m_ld = decoding_results_x_var[ba][session]['LD -> LD'] - decoding_results_x_var[ba][session]['HD -> LD']
        ld_m_hd = decoding_results_x_var[ba][session]['HD -> HD'] - decoding_results_x_var[ba][session]['LD -> HD']
        plt.subplot(2,5,cc)
        hdmld_cdf = ECDF(hd_m_ld)
        ldmhd_cdf = ECDF(ld_m_hd)
        
        
        xx = np.linspace(-100,100,1000)
        
        p, = plt.plot(xx,hdmld_cdf(xx),label='LD-HD')
        # plt.plot(xx,hdld_cdf(xx),label='HD->LD',color=p.get_color(),ls='--')
        # q, = plt.plot(xx,ldld_cdf(xx),label='LD->LD')
        plt.plot(xx,ldmhd_cdf(xx),label='HD-LD')
        plt.plot([0,0],[0,1],'--r')
        plt.plot([-100,100],[0.5,0.5],'--k')
        if cc ==1:
            plt.legend(fontsize=8,loc=2,frameon=False)
        cc+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

ba = 'PPC'
for session in decoding_results_x_var[ba].keys():
    
    dat_npz = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,allow_pickle=True)
    concat = dat_npz['data_concat'].all()
    trial_ind  = 0
    break
    
    
