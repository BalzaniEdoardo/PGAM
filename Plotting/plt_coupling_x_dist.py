#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:44:01 2020

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.model_selection import GridSearchCV

coupl_res = np.load('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/fullFit_coupling_results.npy')

brain_area_order = ['MST','PPC','PFC','VIP']


color_dict = {'MST':'g','PPC':'b','PFC':'r','VIP':(125/255.,)*3}
for monkey in np.unique(coupl_res['monkey']):
    sele_monk = coupl_res['monkey'] == monkey
    coupl_monk = coupl_res[sele_monk]
    
    
    for session in np.unique(coupl_monk['session']):
        sele_sess = coupl_monk['session'] == session
        coupl_sess = coupl_monk[sele_sess]
    
        # extract brain area
        brain_areas = np.unique(coupl_sess['brain area B'])
        num_brain_area = brain_areas.shape[0]
        plt.figure(figsize=[8.74, 3.55])
        plt.suptitle(session,fontsize=20)
        kk=0
        for ba in brain_area_order:
            if not any(brain_areas == ba):
                continue
            plt.subplot(1,num_brain_area,kk+1)
            plt.title(ba,fontsize=15)
            idx = (coupl_sess['brain area A'] == ba) * (coupl_sess['brain area B'] == ba)
            cpl_str = coupl_sess['coupling strength'][idx]
            distance = coupl_sess['electrode distance'][idx]
            kk+=1
            
            if distance.shape[0]>10:
                print(np.max(distance))
                dst_vec = np.unique(distance)
                cpl_mean = np.zeros(dst_vec.shape[0])
                cc = 0
                for dd in dst_vec:
                    cpl_mean[cc] = np.nanmean(cpl_str[distance==dd])
                    cc+=1
                    
                param_grid = {"alpha": [10,5, 4, 3, 1e0, 1e-1, 1e-2],
                              "kernel": [RBF(l)
                         for l in np.logspace(-4, 4, 10)]
                         }
                
                kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
                kr.fit(distance.reshape(distance.shape[0],1), cpl_str)
                dist_plot = np.linspace(np.nanmin(distance),np.nanmax(distance),100)[:,None]
                
                y_kr = kr.predict(dist_plot)
                
                plt.scatter(distance,cpl_str,alpha=0.5,color='k')
                plt.plot(dist_plot.flatten(),y_kr.flatten(),lw=3,color=color_dict[ba])
                plt.ylabel('coupling strength')
                plt.xlabel('distance $\mu m$')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/coupling_x_distance_%s.png'%session)
                
                
