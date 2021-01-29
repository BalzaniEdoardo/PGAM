#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:01:48 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
dat = np.load('coupling_info.npy')

# sign_filter = dat['is significant']
cond = 'replay'
value = 0

npz_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'
for monkey in ['Schro','Quigley']:#Bruno
    keep = ((dat['manipulation type'] == cond) * 
            (dat['manipulation value'] == value) *
            (dat['monkey'] == monkey)
            )
    coupling_data = dat[keep]
    
    connection = ((coupling_data['sender brain area'] == 'MST') *
        (coupling_data['receiver brain area'] == 'MST'))
    cs_whithin_MST = coupling_data['coupling strength'][connection]
    
    connection = ((coupling_data['sender brain area'] == 'PPC') *
        (coupling_data['receiver brain area'] == 'PPC'))
    cs_whithin_PPC = coupling_data['coupling strength'][connection]
    
    connection = ((coupling_data['sender brain area'] == 'PFC') *
        (coupling_data['receiver brain area'] == 'PFC'))
    cs_whithin_PFC = coupling_data['coupling strength'][connection]
    
    max_val = -np.inf
    if cs_whithin_MST.shape[0]:
        max_val = np.max([max_val,np.nanpercentile(cs_whithin_MST, 99)])
    
    if cs_whithin_PPC.shape[0]:
        max_val = np.max([max_val,np.nanpercentile(cs_whithin_PPC, 99)])
    
    if cs_whithin_PFC.shape[0]:
        max_val = np.max([max_val,np.nanpercentile(cs_whithin_PFC, 99)])
    
    x = np.linspace(0, max_val,1000)
    
    plt.figure()
    plt.title('Within area coupling strength: %s %s=%.4f'%(monkey,cond,value))
    if cs_whithin_MST.shape[0]:
        ecdf_mst = ECDF(cs_whithin_MST)
        plt.plot(x,ecdf_mst(x),'g',label='MST')
    
    if cs_whithin_PPC.shape[0]:
        ecdf_ppc = ECDF(cs_whithin_PPC)
        plt.plot(x,ecdf_ppc(x),'b',label='PPC')
    if cs_whithin_PFC.shape[0]:
        ecdf_pfc = ECDF(cs_whithin_PFC)
        plt.plot(x,ecdf_pfc(x),'r',label='PFC')
    
    plt.legend()
    plt.xlabel('coupling strength')
    plt.ylabel('ECDF')
    plt.savefig('%s_coupling_strength_cdf_%s_%.4f.png'%(monkey,cond,value))
    
    
    
    connection = ((coupling_data['sender brain area'] == 'MST') *
        (coupling_data['receiver brain area'] == 'PPC'))
    cs_whithin_MST_to_PPC = coupling_data['coupling strength'][connection]
    
    
    
    connection = ((coupling_data['sender brain area'] == 'MST') *
        (coupling_data['receiver brain area'] == 'PFC'))
    cs_whithin_MST_to_PFC = coupling_data['coupling strength'][connection]
    
    
    connection = ((coupling_data['sender brain area'] == 'PPC') *
        (coupling_data['receiver brain area'] == 'MST'))
    cs_whithin_PPC_to_MST = coupling_data['coupling strength'][connection]
    
    
    
    connection = ((coupling_data['sender brain area'] == 'PPC') *
        (coupling_data['receiver brain area'] == 'PFC'))
    cs_whithin_PPC_to_PFC = coupling_data['coupling strength'][connection]
    
    
    connection = ((coupling_data['sender brain area'] == 'PFC') *
        (coupling_data['receiver brain area'] == 'MST'))
    cs_whithin_PFC_to_MST = coupling_data['coupling strength'][connection]
    
    
    
    connection = ((coupling_data['sender brain area'] == 'PFC') *
        (coupling_data['receiver brain area'] == 'PPC'))
    cs_whithin_PFC_to_PPC = coupling_data['coupling strength'][connection]
    
    
    
    
    
    plt.figure()
    ax1 = plt.subplot(111)

    plt.title('Between area coupling strength: %s %s=%.4f'%(monkey,cond,value))
    if cs_whithin_MST_to_PPC.shape[0]:
        ecdf = ECDF(cs_whithin_MST_to_PPC)
        plt.plot(x,ecdf(x),label='MST->PPC')
        
    if cs_whithin_MST_to_PFC.shape[0]:
        ecdf = ECDF(cs_whithin_MST_to_PFC)
        plt.plot(x,ecdf(x),label='MST->PFC')
        
    if cs_whithin_PPC_to_MST.shape[0]:
        ecdf = ECDF(cs_whithin_PPC_to_MST)
        plt.plot(x,ecdf(x),label='PPC->MST')
        
    if cs_whithin_PPC_to_PFC.shape[0]:
        ecdf = ECDF(cs_whithin_PPC_to_PFC)
        plt.plot(x,ecdf(x),label='PPC->PFC')
        
    if cs_whithin_PFC_to_MST.shape[0]:
        ecdf = ECDF(cs_whithin_PFC_to_MST)
        plt.plot(x,ecdf(x),label='PFC->MST')
    
    if cs_whithin_PFC_to_PPC.shape[0]:
        ecdf = ECDF(cs_whithin_PFC_to_PPC)
        plt.plot(x,ecdf(x),label='PFC->PPC')
        
    plt.legend()
    plt.xlabel('coupling strength')
    plt.ylabel('ECDF')
    plt.tight_layout()
    # plt.savefig('%s_coupling_strength_cdf_%s_%.4f.png'%(monkey,cond,value))
    
    
    # for session in np.unique(coupling_data['session']):
        
    #     all_dat = np.load(os.path.join(npz_path,'%s.npz'%session),allow_pickle=True)
    #     unit_info = all_dat['unit_info'].all()
    #     cont_rate_filter = (unit_info['cR'] < 0.2) | (unit_info['unit_type']=='multiunit')
    #     presence_rate_filter = unit_info['presence_rate'] > 0.9
    #     isi_v_filter = unit_info['isiV'] < 0.2
    #     pp_size = isi_v_filter.shape[0]
    #     combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
        
    #     use_unit = np.arange(1,len(combine_filter)+1)[combine_filter]
        
    #     session_data = coupling_data[coupling_data['session']==session]
        