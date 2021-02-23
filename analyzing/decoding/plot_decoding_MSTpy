#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:20:14 2021

@author: edoardo
"""

import matplotlib.pylab as plt
import numpy as np

var= 'rad_acc'
score_session = np.load('hist_matched_decoding_%s_with_subsamp.npy'%var,allow_pickle=True).all()
plt.figure()
plt.title('Decoding  %s\n PFC R^2 - PPC R^2'%var)

k = 0
xlab = []
for session in score_session.keys():
    
    delta_R2 = score_session[session]['PFC'] - score_session[session]['PPC']
    plt.boxplot(delta_R2,positions=[k])
    xlab += [session]
    
    k += 1
plt.plot([0,k],[0,0],'--r')
plt.xticks(range(k),xlab,rotation=90)
plt.tight_layout()

plt.savefig('PFC_vs_PPC_%s_decoding_deltaR2.pdf'%var)

plt.figure()
plt.title('Decoding %s'%var)

k = 0
xlab = []
y_pfc = []
y_ppc = []
for session in score_session.keys():
    
    y_pfc += [np.mean(score_session[session]['PFC'])]
    y_ppc += [np.mean(score_session[session]['PPC'])]
    
    # plt.boxplot(delta_R2,positions=[k])
    xlab += [session]
    plt.ylabel('r^2')
    k += 1


plt.plot(range(k), y_pfc,'-or',label='PFC')
plt.plot(range(k), y_ppc,'-ob',label='PPC')

plt.xticks(range(k),xlab,rotation=90)
plt.tight_layout()
plt.legend()
    
plt.savefig('PFC_vs_PPC_%s_decoding.pdf'%var)
        
fr_dict = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/decoding/firing_rate_pop.npz',allow_pickle=True)['hist_matched_firing_rates'].all()

plt.figure(figsize=(12,6))
cc = 1
for session in fr_dict.keys():
    
    if len(fr_dict[session]['PFC']) == 0:
        continue
    if len(fr_dict[session]['PPC']) == 0:
        continue
    plt.subplot(2,5,cc)
    plt.title(session)
    plt.hist(fr_dict[session]['PPC'][0]/0.006,color='b',alpha=0.4)
    plt.hist(fr_dict[session]['PFC'][0]/0.006,color='r',alpha=0.4)
    ylim = plt.ylim()
    plt.plot([np.median(fr_dict[session]['PPC'][0]/0.006)]*2,ylim,'b')
    plt.plot([np.median(fr_dict[session]['PFC'][0]/0.006)]*2,ylim,'r')
    if cc % 5 == 1:
        plt.ylabel('counts')
    if cc > 5:
        plt.ylabel('firing rate')
    cc+=1
plt.tight_layout()
plt.savefig('rate_matches.png')
