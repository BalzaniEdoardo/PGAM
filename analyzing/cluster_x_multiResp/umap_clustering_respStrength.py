#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:40:00 2020

@author: edoardo
"""
import os, inspect, sys, re

print(inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')

from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import dill
import pandas as pd
import scipy.stats as sts
import scipy.linalg as linalg
from time import perf_counter
from seaborn import heatmap
from path_class import get_paths_class
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import loadmat
import hdbscan
import umap
from scipy.io import savemat


plt.close('all')

resp_str = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength_info.npy',allow_pickle=True)
variables = np.array(['rad_vel', 'ang_vel','rad_acc','ang_acc', 'rad_path', 'ang_path', 'rad_target', 'ang_target', 'lfp_beta', 'lfp_alpha',
    'lfp_theta', 't_move', 't_flyOFF', 't_stop', 't_reward', 'eye_vert', 'eye_hori', 'spike_hist'])


plt.figure(figsize=(10,4))
bxplt_list = []
for var in variables:
    xx = resp_str['%s_resp_magn_full'%var]
    xx = xx[~np.isnan(xx)]
    xx = xx[xx<=np.nanpercentile(xx,99)]
    bxplt_list += [xx]

plt.boxplot(bxplt_list,labels=variables)
plt.xticks(rotation=90)
plt.tight_layout()

# load and reconstruct table
table_mat = np.load('tuning_info.npy')

variables = np.array(['rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target', 'lfp_beta', 'lfp_alpha',
                      'lfp_theta', 't_move', 't_flyOFF', 't_stop', 't_reward', 'eye_vert', 'eye_hori', 'spike_hist'])

resp_magnitude_full = []
resp_magnitude_reduced = []
for var in variables:
    resp_magnitude_full += ['%s_resp_magn_full' % var]

# for var in variables:
#     resp_magnitude_reduced += ['%s_resp_magn_reduced' % var]

dict_type = {'names': ('session','monkey', 'unit', 'cluster_id', 'electrode_id', 'channel_id', 'brain_area','pseudo_r2') + tuple(variables)
                      + tuple(resp_magnitude_full),
             'formats': ('U30','U30') + (int,) * 4 + ('U3',) +(float,) + (bool,) * variables.shape[0] + (float,) * variables.shape[0]

             }

table_report = np.zeros(0, dtype=dict_type)

# compile table
table_mat = table_mat[table_mat['manipulation type'] == 'all']
for session in np.unique(table_mat['session']):
    tab_sess = table_mat[table_mat['session'] == session]
    tmp = np.zeros(len(np.unique(tab_sess['unit_id'])), dtype=dict_type)
    cc = 0
    for unt in np.unique(tab_sess['unit_id']):
        tab_unit = tab_sess[tab_sess['unit_id']==unt]
        for row in tab_unit:
            if not row['variable'] in variables:
                continue
            tmp[cc]['unit'] = row['unit_id']
            tmp[cc]['monkey'] = row['monkey']
            tmp[cc]['cluster_id'] = row['cluster_id']
            tmp[cc]['channel_id'] = row['channel_id']
            tmp[cc]['electrode_id'] = row['electrode_id']
            tmp[cc]['brain_area'] = row['brain_area']
            tmp[cc]['pseudo_r2'] = row['pseudo-r2']
            tmp[cc][row['variable']] = row['is significant']
            tmp[cc]['%s_resp_magn_full' % row['variable']] = row['response_strength']
            tmp[cc]['session'] = row['session']
        cc+=1

    table_report = np.hstack((table_report,tmp))



# create the matrix for LDA
table_report = table_report[ (table_report['monkey'] != 'Ody') * (table_report['pseudo_r2'] > 0.01)]

X = np.zeros((table_report.shape[0], 16))
cc = 0
for name in table_report.dtype.names:
    if name.endswith('full'):
        X[:, cc] = table_report[name]
        cc += 1

# remove outliers
bool_vec = np.ones(X.shape[0], dtype=bool)
for cc in range(X.shape[1]):
    bool_vec = bool_vec & (X[:, cc] > np.nanpercentile(X[:, cc], 1)) & (X[:, cc] < np.nanpercentile(X[:, cc], 99))

# filter
table_sub = table_report[bool_vec]
data = X[bool_vec]
data = sts.zscore(data)

keep = np.ones(data.shape[0], dtype=bool)  # (table_sub['brain_area'] != 'VIP') # (table_sub['brain_area'] != 'MST')
data = data[keep]

pca_model = PCA(n_components=data.shape[1])
pca_model.fit(data)

plot_comp = data.shape[1]

n_perm = 20
data_perm = np.zeros(data.shape)
expl_var_perm = np.zeros((n_perm, pca_model.explained_variance_ratio_.shape[0]))
for k in range(n_perm):
    for jj in range(data.shape[0]):
        data_perm[jj, :] = data[jj, np.random.permutation(data.shape[1])]

    pca_perm = PCA(n_components=data.shape[1])
    pca_perm.fit(data_perm)
    expl_var_perm[k, :] = pca_perm.explained_variance_ratio_

plt.figure()
plt.plot(np.arange(1, 1 + plot_comp), pca_model.explained_variance_ratio_[:plot_comp], '-og', label='explained by PCA')
plt.errorbar(np.arange(1, 1 + plot_comp), expl_var_perm.mean(axis=0)[:plot_comp],
             yerr=expl_var_perm.std(axis=0)[:plot_comp], color='r', label='explained by chance')
plt.legend()

pval = (expl_var_perm >= pca_model.explained_variance_ratio_).mean(axis=0)
optNum = np.where(pval > 0.05)[0][0]
optPerp = np.round(np.sqrt(data.shape[0]))
pca_dat = pca_model.transform(data)[:, :optNum]

kk = 1
plt.figure()
for n_neighbors in [20, 40, 50, 100]:
    fit = umap.UMAP(n_neighbors=n_neighbors)
    umap_res = fit.fit_transform(pca_dat)

    # tsne_res = TSNE(n_components=2, perplexity=optPerp, early_exaggeration=12.0, learning_rate=200.0, n_iter=maxiter).fit_transform(pca_dat)

    plt.subplot(1,1,kk)

    plt.scatter(umap_res[table_sub['brain_area'][keep]=='PFC',0],
                umap_res[table_sub['brain_area'][keep]=='PFC',1],s=10,c='r',alpha=0.4)

    plt.scatter(umap_res[table_sub['brain_area'][keep]=='MST',0],
                umap_res[table_sub['brain_area'][keep]=='MST',1],s=10,c='g',alpha=0.4)

    plt.scatter(umap_res[table_sub['brain_area'][keep]=='PPC',0],
                umap_res[table_sub['brain_area'][keep]=='PPC',1],s=10,c='b',alpha=0.4)


    kk += 1


cl_size = np.arange(3,51,dtype=int)
score = np.zeros((cl_size.shape[0],50))

for kk in range(50):
    print('KK:',kk)
    cc = 0
    for mn_cl_size in cl_size:
        print('%d/50'%(cc+1))
        tsne_res = TSNE(n_components=2, perplexity=optPerp, early_exaggeration=12.0, learning_rate=200.0, n_iter=500).fit_transform(pca_dat)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(mn_cl_size)).fit(tsne_res)
        score[cc,kk] = (clusterer.probabilities_ < 0.05).mean()
        cc+=1


# tsne_res = TSNE(n_components=2, perplexity=optPerp, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000).fit_transform(pca_dat)
fit = umap.UMAP(n_neighbors=50)
tsne_res = fit.fit_transform(pca_dat)

clusterer = hdbscan.HDBSCAN(min_cluster_size=int(100)).fit(tsne_res)

plt.figure()
plt.suptitle('UMAP results')
plt.subplot(121, aspect='equal')
plt.scatter(tsne_res[table_sub['brain_area'][keep] == 'PPC', 0],
            tsne_res[table_sub['brain_area'][keep] == 'PPC', 1], s=10, c='b', alpha=0.4)
plt.scatter(tsne_res[table_sub['brain_area'][keep] == 'PFC', 0],
            tsne_res[table_sub['brain_area'][keep] == 'PFC', 1], s=10, c='r', alpha=0.4)

plt.scatter(tsne_res[table_sub['brain_area'][keep] == 'MST', 0],
            tsne_res[table_sub['brain_area'][keep] == 'MST', 1], s=10, c='g', alpha=0.4)
plt.scatter(tsne_res[table_sub['brain_area'][keep] == 'VIP', 0],
            tsne_res[table_sub['brain_area'][keep] == 'VIP', 1], s=10, c='k', alpha=0.9)

plt.title('Brain Area')

plt.subplot(122, aspect='equal')
for label in np.unique(clusterer.labels_):
    if label != -1:
        plt.scatter(tsne_res[clusterer.labels_ == label, 0],
                    tsne_res[clusterer.labels_ == label, 1], s=10, alpha=0.4)
    else:
        plt.scatter(tsne_res[clusterer.labels_ == label, 0],
                    tsne_res[clusterer.labels_ == label, 1], s=10, alpha=0.4, c=[0.5, 0.5, 0.5])

plt.title('HDBScan results')
plt.savefig('umap_dim_reduction_respStrength.png')

mdict = {'resp_strength': X[bool_vec], 'info':table_report[bool_vec], 'umap_proj':tsne_res, 'cluster_label':clusterer.labels_,
         'variable_label':variables}
savemat('hdbscan_umap_respStrength.mat',mdict=mdict)

#
# plt.figure()
#
# ppc_bool = table_sub['brain_area'][keep] == 'PPC'
# ppc_tot = (ppc_bool * (clusterer.labels_ != -1)).sum()
#
# pfc_bool = table_sub['brain_area'][keep] == 'PFC'
# pfc_tot = (pfc_bool * (clusterer.labels_ != -1)).sum()
#
# mst_bool = table_sub['brain_area'][keep] == 'MST'
# mst_tot = (mst_bool * (clusterer.labels_ != -1)).sum()
#
# vip_bool = table_sub['brain_area'][keep] == 'VIP'
# vip_tot = (vip_bool * (clusterer.labels_ != -1)).sum()
#
# perc_ppc = []
# perc_pfc = []
# perc_mst = []
# perc_vip = []
# kk = 0
# for label in np.unique(clusterer.labels_):
#     if label == -1:
#         continue
#     perc_ppc += [(ppc_bool * (clusterer.labels_ == label)).sum() / ppc_tot]
#     perc_pfc += [(pfc_bool * (clusterer.labels_ == label)).sum() / pfc_tot]
#     perc_mst += [(mst_bool * (clusterer.labels_ == label)).sum() / mst_tot]
#     perc_vip += [(vip_bool * (clusterer.labels_ == label)).sum() / vip_tot]
#
#     kk += 1
#
# sep = 2
# xaxis_pfc = list(range(kk))
# xaxis_ppc = list(range(2 + kk, 2 + 2 * kk))
# xaxis_mst = list(range(4 + 2 * kk, 4 + 3 * kk))
# xaxis_vip = list(range(6 + 3 * kk, 6 + 4 * kk))
#
# plt.bar(xaxis_pfc, perc_pfc, color='r', label='PFC')
# plt.bar(xaxis_ppc, perc_ppc, color='b', label='PPC')
# plt.bar(xaxis_mst, perc_mst, color='g', label='MST')
# plt.bar(xaxis_vip, perc_vip, color='k', label='VIP')
#
# plt.xticks(np.hstack((xaxis_pfc, xaxis_ppc, xaxis_mst, xaxis_vip)), ['clust A', 'clust B', 'clust C'] * 4, rotation=45)
# plt.ylabel('percentage')
# plt.legend()
# plt.tight_layout()
#

