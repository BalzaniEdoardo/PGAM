#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:39:32 2021

@author: edoardo
"""
import numpy as np
import sys,os,dill
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
sys.path.append('/scratch/eb162/GAM_library/')
from spike_times_class import spike_counts
from behav_class import behavior_experiment,load_trial_types
from lfp_class import lfp_class
from copy import deepcopy
from scipy.io import loadmat,savemat
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PIL import ImageColor
from sklearn.decomposition import PCA
from seaborn import heatmap
import umap
import hdbscan


cond = 'density'
val1 = 0.005
val2 = 0.0001
monkey = 'Schro'



monkey_dict = {'Schro':'m53s','Quigley':'m44s','Bruno':'m51'}

sess_list = []
for root, dirs, files in os.walk('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc'):
    for fh_name in files:
        for cond in [cond]:
            if cond in fh_name and monkey_dict[monkey] in fh_name:
                sess_list += [fh_name.split('results_')[1].split('_')[0]]


sess_list = np.unique(sess_list)

first = True

for session in sess_list:
    # if session != 'm53s97':
    #     continue
    # bs_folder_old = '/Volumes/WD_Edo/firefly_analysis/LFP_band/results/processed_dPCA/'
    
    bs_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/multi_event_aligned/'
    
    
    dat = np.load(bs_folder + 'flyON_%s_multiresc_trials.npz'%session, allow_pickle=True)
    dat_info = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,
                       allow_pickle=True)
    
    unit_info = dat_info['unit_info'].all()
    cont_rate_filter = unit_info['cR']
    unit_type = unit_info['unit_type']
    presence_rate_filter = unit_info['presence_rate']
    isi_v_filter = unit_info['isiV']
    
    cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
    presence_rate_filter = presence_rate_filter > 0.9
    isi_v_filter = isi_v_filter < 0.2
    combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
    
    
    
    
    
    rates = dat['rescaled_rate'][combine_filter]
    brain_area = unit_info['brain_area'][combine_filter]
    
    trial_list = dat['trial_list']
    try:
        trial_type = dat['trial_type']
    except:
        trial_type = dat_info['info_trial'].all().trial_type
    time_bounds = dat['time_bounds']
    time_rescale = dat['time_rescale']
    
    
    mu = np.nanmean(np.nanmean(rates,axis=1),axis=1)
    sigma = np.std(np.nanmean(rates,axis=1),axis=1)
    
    
    
    
    cond_A = trial_type[trial_list][cond] == val1
    cond_B = trial_type[trial_list][cond] == val2
    
    
    z_rates = (rates - mu.reshape(mu.shape[0],1,1)) / sigma.reshape(sigma.shape[0],1,1)
    
    if first:
        cond_A_all = deepcopy(cond_A)
        cond_B_all = deepcopy(cond_B)
        brain_area_all = deepcopy(brain_area)
        z_rates_A = np.nanmean(z_rates[:,cond_A],axis=1)
        z_rates_B = np.nanmean(z_rates[:,cond_B],axis=1)
        first = False
        
    else:
        
        cond_A_all = np.hstack((cond_A_all, cond_A))
        cond_B_all = np.hstack((cond_B_all, cond_B))
        brain_area_all = np.hstack((brain_area_all, brain_area))
        z_rates_A = np.vstack((z_rates_A, np.nanmean(z_rates[:,cond_A],axis=1)))
        z_rates_B = np.vstack((z_rates_B, np.nanmean(z_rates[:,cond_B],axis=1)))
    
    # time_rescale = np.arange(z_rate.shape[1]) * 0.006
    # extract position of events


idx_vline = []
for k in time_bounds[1:]:
    idx_vline += [np.where(time_rescale <= k)[0][-1]]
idx_vline = np.hstack(([0], idx_vline))
    
for ba in np.unique(brain_area):
    sel = brain_area_all == ba
    z_A = z_rates_A[sel]
    z_B = z_rates_B[sel]
    
    # z_A =  np.nanmean(z_rates_ba[:,cond_A],axis=1)
    # z_B =  np.nanmean(z_rates_ba[:,cond_B],axis=1)
    idx_B = np.argmax(z_B,axis=1)
    idx_A = np.argmax(z_A,axis=1)

    sort_idx = np.argsort(idx_A)
    plt.figure()
    plt.suptitle(ba,fontsize=20)
    plt.subplot(1,2,1)
    
    plt.title('%s: %.4f'%(cond,val1))
    heatmap(z_A[sort_idx,:],vmin=-3.,vmax=4.5)
    ylim = plt.ylim()
    plt.vlines(idx_vline,ylim[0],ylim[1])
    xaxis = idx_vline
    plt.yticks([])
    plt.xticks(plt.xlim(),[0,1])
    plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)


    
    plt.subplot(1,2,2)
    sort_idx = np.argsort(idx_B)

    plt.title('%s: %.4f'%(cond,val2))
    heatmap(z_B[sort_idx,:],vmin=-3.,vmax=4.5)
    plt.yticks([])
    ylim = plt.ylim()
    plt.vlines(idx_vline,ylim[0],ylim[1])
    plt.xticks(plt.xlim(),[0,1])
    xaxis = idx_vline

    plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('indipSort_%s_%s_%s_aligned_rates.png'%(cond, monkey,ba))
    

plt.figure(figsize=(10,8))
plt.suptitle(cond+' examples')
for k in range(25):
    plt.subplot(5,5,k+1)
    if k == 0:
        plt.plot(z_rates_A[k+50,:],label='%.4f'%val1)
        plt.plot(z_rates_B[k+50,:],label='%.4f'%val2)
        plt.legend()

    else:
        plt.plot(z_rates_A[k,:])
        plt.plot(z_rates_B[k,:])
    
    
    plt.xticks([])
    plt.yticks([])
    ylim = plt.ylim()
    plt.vlines(idx_vline,ylim[0],ylim[1])
    # plt.xticks(plt.xlim(),[0,1])
    plt.title('%s'%brain_area_all[k])
    if k >=20:
        plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)

xaxis = idx_vline

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


model = PCA()
model.fit(z_rates_A)
proj = model.transform(z_rates_A)[:,:6]

n_neighbors = 30
fit = umap.UMAP(n_neighbors=n_neighbors)
umap_res = fit.fit_transform(proj)

plt.close('all')
plt.subplot(1,2,1)
plt.scatter(umap_res[brain_area_all=='PPC',0],umap_res[brain_area_all=='PPC',1],color='b',alpha=0.5)
plt.scatter(umap_res[brain_area_all=='PFC',0],umap_res[brain_area_all=='PFC',1],color='r',alpha=0.5)

plt.scatter(umap_res[brain_area_all=='MST',0],umap_res[brain_area_all=='MST',1],color='g',alpha=0.5)

plt.subplot(1,2,2)
clusterer = hdbscan.HDBSCAN(min_cluster_size=int(80)).fit(umap_res)

for label in np.unique(clusterer.labels_):
   if label == -1:
       color = (0.5,)*3
       plt.scatter(umap_res[clusterer.labels_==label,0],umap_res[clusterer.labels_==label,1],color=color,alpha=0.5)
   else:
       
       plt.scatter(umap_res[clusterer.labels_==label,0],umap_res[clusterer.labels_==label,1],alpha=0.5)


# plt.figure()


# plt.subplot(111,axis)
# plt.savefig('%s_%s_%s_examples.png'%(cond, monkey,ba))
# ba = 'PPC'
# z_mean_rates = np.nanmean(z_rates,axis=1)
# idx = np.argmax(z_mean_rates,axis=1)
# sort_idx = np.argsort(idx)
# heatmap(z_mean_rates[sort_idx,:],vmin=-3.,vmax=4.5)
    
# plt.figure()
# first = True
# for unit in range(z_rates.shape[0]):
#     if first:
#         pa, = plt.plot(np.nanmean(z_rates[:,cond_A,:],axis=1)[unit])
#         pb, = plt.plot(np.nanmean(z_rates[:,cond_B,:],axis=1)[unit])
#         first = False
#     else:
#         plt.plot(np.nanmean(z_rates[:,cond_A,:],axis=1)[unit],color=pa.get_color())
#         plt.plot(np.nanmean(z_rates[:,cond_B,:],axis=1)[unit],color=pb.get_color())

# plt.figure()
# plt.subplot(121)

        


