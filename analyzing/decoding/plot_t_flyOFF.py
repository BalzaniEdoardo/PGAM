#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:20:14 2021

@author: edoardo
"""

import matplotlib.pylab as plt
import numpy as np

var= 't_flyOFF'
score_session = np.load('decoding_linSVM_%s.npy'%var,allow_pickle=True).all()
plt.figure()
plt.title('Decoding  %s\n PFC accuracy - PPC accuracy'%var)

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
plt.title('Decoding %s presentation'%var)

k = 0
xlab = []
y_pfc = []
y_ppc = []
for session in score_session.keys():
    
    y_pfc += [np.mean(score_session[session]['PFC'])]
    y_ppc += [np.mean(score_session[session]['PPC'])]
    
    # plt.boxplot(delta_R2,positions=[k])
    xlab += [session]
    plt.ylabel('class-balanced accuracy')
    k += 1


plt.plot(range(k), y_pfc,'-or',label='PFC')
plt.plot(range(k), y_ppc,'-ob',label='PPC')

plt.xticks(range(k),xlab,rotation=90)
plt.tight_layout()
plt.legend()
    
plt.savefig('PFC_vs_PPC_%s_decoding.pdf'%var)
        