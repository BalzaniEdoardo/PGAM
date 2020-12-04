#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:43:01 2020

@author: edoardo
"""
import numpy as np
import scipy.stats as sts
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from scipy.io import loadmat

plt.close('all')
table_mat = loadmat('/Users/edoardo/Work/Code/GAM_code/plotting/table_report.mat')['table_report']


labels_ba = np.zeros(table_mat.shape[1],dtype='U10')

# var_names = np.array(['rad_vel','ang_vel','rad_path','ang_path','rad_target',
#              'ang_target', 'lfp_beta','lfp_alpha','lfp_theta','t_move',
#              't_flyOFF','t_stop','t_reward','eye_vert','eye_hori'])

var_names = np.array(['rad_vel','ang_vel','rad_path','ang_path','rad_target',
              'ang_target', 't_move',
              't_flyOFF','t_stop','t_reward','eye_vert','eye_hori'])


boolean_table = np.zeros((table_mat.shape[1],var_names.shape[0]))
magn_table = np.zeros((table_mat.shape[1],var_names.shape[0]))

cc = 0
for var in var_names:
    boolean_table[:,cc] = table_mat[0,:][var]
    magn_table[:,cc] = table_mat[0,:]['%s_resp_magn_full'%var]
    
    cc += 1

magn_table = sts.zscore(magn_table,axis=0)  
db = DBSCAN(eps=5, min_samples=30).fit(magn_table)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

for kk in range(table_mat.shape[1]):
    labels_ba[kk] = table_mat['brain_area'][0,kk][0]

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(magn_table, labels))

pca_model = PCA(n_components=5)
fit_pca = pca_model.fit(magn_table)

proj = fit_pca.transform(magn_table)


plt.figure()
col_ba = {'MST':'g','PPC':'b','PFC':'r','VIP':(125./255,)*3}
for ba in ['PPC','PFC','MST','VIP']:
    plt.scatter(proj[labels_ba==ba,0],proj[labels_ba==ba,1],color=col_ba[ba],alpha=0.5)
    
plt.figure()
cc = 1
col_ba = {'MST':'g','PPC':'b','PFC':'r','VIP':(125./255,)*3}
for ba in ['PPC','PFC','MST','VIP']:
    plt.subplot(2,2,cc)
    plt.scatter(proj[labels_ba==ba,0],proj[labels_ba==ba,1],color=col_ba[ba],alpha=0.5)
    plt.xlim(-5,20)
    plt.ylim(-15,15)
    cc+=1
    

idx_sort = np.arange(var_names.shape[0])#np.argsort(fit_pca.components_[0,:])
plt.figure()
plt.plot(fit_pca.components_[0,idx_sort],'-ob')
plt.plot(fit_pca.components_[1,idx_sort],'-or')
plt.xticks(range(len(var_names)),var_names[idx_sort],rotation=90)
plt.tight_layout()





## non z-score response
boolean_table = np.zeros((table_mat.shape[1],var_names.shape[0]))
magn_table = np.zeros((table_mat.shape[1],var_names.shape[0]))

cc = 0
for var in var_names:
    boolean_table[:,cc] = table_mat[0,:][var]
    magn_table[:,cc] = table_mat[0,:]['%s_resp_magn_full'%var]
    
    cc += 1
    
xaxis = {'MST':[],'PPC':[],'PFC':[]}
yaxis =  {'MST':[],'PPC':[],'PFC':[]}
yerr_p =  {'MST':[],'PPC':[],'PFC':[]}
yerr_m =  {'MST':[],'PPC':[],'PFC':[]}
xticks = []

cc=0
for var in var_names:
    for ba in ['MST','PPC','PFC']:
        xaxis[ba] += [cc]
        yaxis[ba] += [np.nanmedian(magn_table[labels_ba==ba,var_names==var])]
        yerr_p[ba] += [np.abs(np.nanpercentile(magn_table[labels_ba==ba,var_names==var],75)-yaxis[ba][-1])]
        yerr_m[ba] += [np.abs(np.nanpercentile(magn_table[labels_ba==ba,var_names==var],25)-yaxis[ba][-1])]
        
        cc += 1
    xticks += [cc-2]
    cc += 2
        
yerr = {'MST':np.zeros((2,len(yerr_m['MST']))), 
        'PFC':np.zeros((2,len(yerr_m['PFC']))),'PPC':np.zeros((2,len(yerr_m['PPC'])))}
for ba in ['MST','PPC','PFC']:
    yerr[ba][0,:] = yerr_m[ba]
    yerr[ba][1,:] = yerr_p[ba]

plt.figure(figsize=[10.41,  5.57])  
for ba in ['MST','PPC','PFC']:
    plt.bar(xaxis[ba],yaxis[ba],yerr=yerr[ba],width=0.6,color=col_ba[ba],label=ba)
    # plt.bar(xaxis[ba],yaxis[ba],width=0.6,color=col_ba[ba])
plt.xticks(xticks,var_names,rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)

plt.title('response strength median $\pm$ quartiles',fontsize=20)
plt.tight_layout()
if 'lfp_beta' in  var_names:
    
    plt.savefig('/Users/edoardo/Work/Code/GAM_code/plotting/resp_strength/resp_strength_x_variables_lfp.pdf')
else:
    plt.savefig('/Users/edoardo/Work/Code/GAM_code/plotting/resp_strength/resp_strength_x_variables.pdf')

