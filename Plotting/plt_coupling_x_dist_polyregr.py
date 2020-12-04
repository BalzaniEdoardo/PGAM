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

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error


coupl_res = np.load('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/fullFit_coupling_results.npy')

brain_area_order = ['MST','PPC','PFC','VIP']


color_dict = {'MST':'g','PPC':'b','PFC':'r','VIP':(125/255.,)*3}
for monkey in np.unique(coupl_res['monkey']):
    sele_monk = coupl_res['monkey'] == monkey
    coupl_monk = coupl_res[sele_monk]
    
    pooled_dist = {'MST':[],'PPC':[],'PFC':[],'VIP':[]}
    pooled_coupl = {'MST':[],'PPC':[],'PFC':[],'VIP':[]}
    pooled_sign = {'MST':[],'PPC':[],'PFC':[],'VIP':[]}

    for session in np.unique(coupl_monk['session']):
        sele_sess = coupl_monk['session'] == session
        coupl_sess = coupl_monk[sele_sess]
    
        # extract brain area
        brain_areas = np.unique(coupl_sess['brain area B'])
        num_brain_area = brain_areas.shape[0]
        # plt.figure(figsize=[8.74, 3.55])
        # plt.suptitle(session,fontsize=20)
        kk=0
        for ba in brain_area_order:
            if not any(brain_areas == ba):
                continue
            # plt.subplot(1,num_brain_area,kk+1)
            # plt.title(ba,fontsize=15)
            idx = (coupl_sess['brain area A'] == ba) * (coupl_sess['brain area B'] == ba)
            cpl_str = coupl_sess['coupling strength'][idx]
            distance = coupl_sess['electrode distance'][idx]
            issign = coupl_sess['is significant'][idx]
            pooled_dist[ba] = np.hstack((pooled_dist[ba],distance))
            pooled_coupl[ba] = np.hstack((pooled_coupl[ba],cpl_str))
            pooled_sign[ba] = np.hstack((pooled_sign[ba],issign))
            kk+=1
            
            if distance.shape[0]>10:
                print(np.max(distance))
                dst_vec = np.unique(distance)
                cpl_mean = np.zeros(dst_vec.shape[0])
                cc = 0
                for dd in dst_vec:
                    cpl_mean[cc] = np.nanmean(cpl_str[distance==dd])
                    cc+=1
                    
                # param_grid = {"alpha": [10,5, 4, 3, 1e0, 1e-1, 1e-2],
                #               "kernel": [RBF(l)
                #          for l in np.logspace(-4, 4, 10)]
                         # }
                
                # estimator = ('RANSAC', RANSACRegressor(random_state=42))
                # model = make_pipeline(PolynomialFeatures(3), estimator[1])
                # model.fit(distance.reshape(distance.shape[0],1), cpl_str)
                

                # dist_plot = np.linspace(np.nanmin(distance),np.nanmax(distance),100)[:,None]
                
                # y_poly = model.predict(dist_plot)
                
        #         plt.scatter(distance,cpl_str,alpha=0.5,color='k')
        #         plt.plot(dst_vec,cpl_mean,lw=3,color=color_dict[ba])
        #         plt.ylabel('coupling strength')
        #         plt.xlabel('distance $\mu m$')
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.savefig('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/poly/coupling_x_distance_%s.png'%session)
        # plt.close('all')
    
    plt.figure()
    ax = plt.subplot(111)
    plt.title(monkey)
    for ba in pooled_coupl.keys():
        if ba == 'VIP':
            continue
        
        dst_vec = np.unique(pooled_dist[ba])
        dst_vec = dst_vec[dst_vec<2000]
        cpl_mean = np.zeros(dst_vec.shape[0])
        frac_coupled = np.zeros(dst_vec.shape[0])
        cc = 0
        for dd in dst_vec:
            cpl_mean[cc] = np.nanmean(pooled_coupl[ba][pooled_dist[ba]==dd])
            frac_coupled[cc] = np.nanmean(pooled_sign[ba][pooled_dist[ba]==dd])
            cc+=1
    
        plt.plot(dst_vec,cpl_mean,lw=3,color=color_dict[ba],marker='o',label=ba)
                
    plt.xlabel('distance $\mu m$',fontsize=15)
    plt.ylabel('coupling strength',fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.savefig('/Users/edoardo/Dropbox/gam_firefly_pipeline/coupling_x_distance/coupling_x_distance_%s.png'%monkey)
        
    plt.figure()
    ax = plt.subplot(111)
    plt.title(monkey)
    for ba in pooled_coupl.keys():
        if ba == 'VIP':
            continue
        
        dst_vec = np.unique(pooled_dist[ba])
        dst_vec = dst_vec[dst_vec<2000]
        cpl_mean = np.zeros(dst_vec.shape[0])
        frac_coupled = np.zeros(dst_vec.shape[0])
        cc = 0
        for dd in dst_vec:
            cpl_mean[cc] = np.nanmean(pooled_coupl[ba][pooled_dist[ba]==dd])
            frac_coupled[cc] = np.nanmean(pooled_sign[ba][pooled_dist[ba]==dd])
            cc+=1
    
        plt.plot(dst_vec,frac_coupled,lw=3,color=color_dict[ba],marker='o',label=ba)
                
    plt.xlabel('distance $\mu m$',fontsize=15)
    plt.ylabel('coupling probability',fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()