#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:20:14 2021

@author: edoardo
"""

import matplotlib.pylab as plt
import numpy as np

import pandas as pd
import seaborn as sns

dtype_dict = {'names':('session','variable','$R^2$','area'),
              'formats':('U30','U30',float,'U30')}
dtype_dict_frac = {'names':('session','variable','fractional increase','area'),
              'formats':('U30','U30',float,'U30')}

table_r2 = np.zeros(0,dtype=dtype_dict)
table_frac = np.zeros(0,dtype=dtype_dict_frac)

for var in ['rad_acc','rad_vel','t_stop','t_flyOFF','t_reward','eye_hori','eye_vert','rad_target','rad_path']:
    
    try:
        score_session = np.load('hist_matched_decoding_%s_with_subsamp.npy'%var,allow_pickle=True).all()
    except:
        score_session = np.load('decoding_linSVM_%s.npy'%var,allow_pickle=True).all()

    
    y_pfc = []
    y_ppc = []
    sess_list = list(score_session.keys())
    for session in sess_list:
        
        y_pfc += [np.mean(score_session[session]['PFC'])]
        y_ppc += [np.mean(score_session[session]['PPC'])]
    
    
    tmp_r2 = np.zeros(len(y_ppc)*2,dtype=dtype_dict)
    tmp_r2['session'] = session
    tmp_r2['area'] = ['PPC']*len(y_ppc)+['PFC']*len(y_pfc)
    tmp_r2['variable'] = var
    tmp_r2['$R^2$'] = y_ppc + y_pfc

    table_r2 = np.hstack((table_r2,tmp_r2))

for ba in ['MST','PFC','VIP']:
    for var in ['rad_acc','rad_vel','t_stop','t_move','t_flyOFF','t_reward','eye_hori','eye_vert','rad_target','rad_path']:
        if ba == 'PFC':
            try:
                score_session = np.load('hist_matched_decoding_%s_with_subsamp.npy'%var,allow_pickle=True).all()
            except:
                score_session = np.load('decoding_linSVM_%s.npy'%var,allow_pickle=True).all()
        if ba == 'MST':
            
            try:
                score_session = np.load('MST_hist_matched_decoding_%s_with_subsamp.npy'%var,allow_pickle=True).all()
            except:
                score_session = np.load('MST_decoding_linSVM_%s.npy'%var,allow_pickle=True).all()
        if ba == 'VIP':
            
            try:
                score_session = np.load('VIP_hist_matched_decoding_%s_with_subsamp.npy'%var,allow_pickle=True).all()
            except:
                score_session = np.load('VIP_decoding_linSVM_%s.npy'%var,allow_pickle=True).all()
        y_pfc = []
        y_ppc = []
        sess_list = list(score_session.keys())
        for session in sess_list:
            if ba == 'MST' and ((session == 'm53s93') | (session == 'm53s128')):
                continue
            print(ba,var,session, np.max(score_session[session]['PPC']), np.min(score_session[session]['PPC']))
            y_pfc += [np.mean(score_session[session][ba])]
            y_ppc += [np.mean(score_session[session]['PPC'])]
            tmp_frac = np.zeros(len(y_ppc),dtype=dtype_dict_frac)
            tmp_frac['session'] = session
            tmp_frac['variable'] = var
            tmp_frac['fractional increase'] = -1*2*(np.array(y_ppc)-np.array(y_pfc)) / (np.array(y_ppc)+np.array(y_pfc)) 
            tmp_frac['area'] = ba
    
        table_frac = np.hstack((table_frac,tmp_frac)) 



fig = plt.figure(figsize=[12.28,  4.83])

ax1 = fig.add_subplot(121)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


ax2 = fig.add_subplot(122)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

palette ={"PFC": "r", "MST": "g",'VIP':'k'}

bl = np.ones(table_frac.shape,dtype=bool)
for var in np.unique(table_frac['variable']):
    if var.startswith('t_'):
        bl [table_frac['variable']==var ] = False
        
        
dat = pd.DataFrame(table_frac[bl])
g= sns.stripplot(data=dat,hue='area', x="variable", y="fractional increase", split=True,
            palette=palette,ax=ax1)

ax1.set_title('Linear regression')
xlim = ax1.get_xlim()
ax1.plot(xlim,[0,0],'--b',lw=2)

dat = pd.DataFrame(table_frac[~bl])
g= sns.stripplot(data=dat,hue='area', x="variable", y="fractional increase", split=True,
            palette=palette,ax=ax2)

ax2.set_title('Linear SVM classifier')
xlim = ax2.get_xlim()
ax2.plot(xlim,[0,0],'--b',lw=2)


plt.suptitle('fractional increase in decoding performace')
plt.tight_layout(rect=[0, 0.05, 0.90, 0.95])
plt.savefig('vip_decoding_pecent_increment_split.png')

# plt.title('Decoding %s'%var)

# k = 0
# xlab = []
# y_pfc = []
# y_ppc = []
# y_ppc = np.array(y_ppc)
# y_pfc = np.array(y_pfc)

    
# frac = 2*(y_ppc-y_pfc) / (y_ppc+y_pfc) 

# xlab += [var]
# plt.ylabel('r^2')
# k += 1



    
# plt.savefig('PFC_vs_PPC_%s_decoding.pdf'%var)
        


