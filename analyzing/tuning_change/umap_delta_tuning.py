#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:09:27 2021

@author: edoardo
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:18:36 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

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
from knots_constructor import knots_cerate
from copy import deepcopy
from time import sleep
from spline_basis_toolbox import *
from bisect import bisect_left
from statsmodels.distributions import ECDF
import venn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

plt.close('all')
monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}

path_gen = get_paths_class()
info_selectivity = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength_info.npy')
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/eval_matrix_and_info.npz',allow_pickle=True)
        #eval_matrix=eval_matrix,info=info_matrix,index_list=index_list)
info_evals = dat['info']
eval_matrix = dat['eval_matrix']
var_list = dat['index_list']


man_type = 'density'
val1 = 0.005
val2 = 0.0001
variable = 't_reward'


# man_type = 'controlgain'
# val1 = 1
# val2 = 2



filt = ((info_evals['monkey'] == 'Schro') * (info_evals['brain area'] == 'PFC') * 
        (info_evals['manipulation type'] == man_type))

tuning_high_dens = eval_matrix[filt*(info_evals['manipulation value'] == val1),:]
tuning_low_dens = eval_matrix[filt*(info_evals['manipulation value'] == val2),:]

tuning_high_dens = tuning_high_dens[:,var_list == variable]
tuning_low_dens = tuning_low_dens[:,var_list == variable]

info_high_dens = info_evals[filt*(info_evals['manipulation value'] == val1)]
info_low_dens = info_evals[filt*(info_evals['manipulation value'] == val2)]




# Extraxt significant units
sign_units = []
sign_session = []
bool_sele = (info_selectivity['manipulation type'] == man_type) *  (info_selectivity['manipulation value'] == val1) 
            
info_sele = info_selectivity[bool_sele]
for info in info_high_dens:
    filt = ((info_sele['session'] == info['session']) * 
                             (info_sele['unit'] == info['unit id'])
                             )
    assert(np.sum(filt) <= 1)
    if np.sum(filt) == 0:
        continue
    if info_sele['t_flyOFF'][filt]:
        sign_units += [info['unit id']]
        sign_session += [info['session']]
        
bool_sele = (info_selectivity['manipulation type'] == man_type) *  (info_selectivity['manipulation value'] == val2) 
            
info_sele = info_selectivity[bool_sele]

for info in info_low_dens:
    filt = ((info_sele['session'] == info['session']) * 
             (info_sele['unit'] == info['unit id'])
                             )
    assert(np.sum(filt) <= 1)
    if np.sum(filt) == 0:
        continue
    if info['unit id'] in sign_units and info['session'] in sign_session:
        continue
    if info_sele['t_flyOFF'][filt]:
        sign_units += [info['unit id']]
        sign_session += [info['session']]
        


sign_tuning_hd = np.zeros(tuning_high_dens.shape)*np.nan
unit_id_hd = np.zeros(tuning_high_dens.shape[0],dtype='U30')
for k in range(len(sign_units)):
    idx = (info_high_dens['session'] == sign_session[k]) * (info_high_dens['unit id'] == sign_units[k])
    if idx.sum() > 0:
        sign_tuning_hd[k,:] = tuning_high_dens[idx,:]
        unit_id_hd[k] = '%s_%d'%(sign_session[k], sign_units[k])


sign_tuning_ld = np.zeros(tuning_low_dens.shape)*np.nan
unit_id_ld = np.zeros(tuning_low_dens.shape[0],dtype='U30')
for k in range(len(sign_units)):
    idx = (info_low_dens['session'] == sign_session[k]) * (info_low_dens['unit id'] == sign_units[k])
    if idx.sum() > 0:
        sign_tuning_ld[k,:] = tuning_low_dens[idx,:]
        unit_id_ld[k] = '%s_%d'%(sign_session[k], sign_units[k])
        


sign_tuning_ld = sign_tuning_ld[unit_id_ld != '']
unit_id_ld = unit_id_ld[unit_id_ld != '']

sign_tuning_hd = sign_tuning_hd[unit_id_hd != '']
unit_id_hd = unit_id_hd[unit_id_hd != '']





# all resp
response_funcs = np.vstack((sign_tuning_hd,sign_tuning_ld))
id_units = np.hstack((unit_id_hd,unit_id_ld))
condition_label = np.array([0.005]*sign_tuning_hd.shape[0]+[0.0001]*sign_tuning_ld.shape[0])

# cerate tensor contaning all the resp function
resp_tensor = np.zeros((0,2,sign_tuning_ld.shape[1]))
units_id_tensor = []
for ID in np.unique(id_units):
    flt_hd = unit_id_hd == ID
    flt_ld = unit_id_ld == ID
    if not (any(flt_ld) and any(flt_hd)):
        continue
    
    tmp = np.zeros((1,2,sign_tuning_ld.shape[1]))
    tmp[0,0,:] = sign_tuning_hd[flt_hd]
    tmp[0,1,:] = sign_tuning_ld[flt_ld]
    
    resp_tensor = np.vstack((resp_tensor,tmp))
    units_id_tensor += [ID]
    
units_id_tensor = np.array(units_id_tensor)

    

n_perm = 20
plot_comp = sign_tuning_hd.shape[1]
pca_model = PCA(n_components=sign_tuning_hd.shape[1])
diff_tunining = np.squeeze(np.diff(resp_tensor,axis=1))
pca_model.fit(diff_tunining)


data_perm = np.zeros(diff_tunining.shape)
expl_var_perm = np.zeros((n_perm,pca_model.explained_variance_ratio_.shape[0]))
for k in range(n_perm):
    for jj in range(diff_tunining.shape[0]):
        data_perm[jj,:] = diff_tunining[jj,np.random.permutation(diff_tunining.shape[1])]
        
    pca_perm = PCA(n_components=diff_tunining.shape[1])
    pca_perm.fit(data_perm)
    expl_var_perm[k,:] = pca_perm.explained_variance_ratio_

plt.plot(np.arange(1,1+plot_comp),pca_model.explained_variance_ratio_[:plot_comp],'-og',label='explained by PCA')
plt.errorbar(np.arange(1,1+plot_comp),expl_var_perm.mean(axis=0)[:plot_comp],yerr=expl_var_perm.std(axis=0)[:plot_comp],color='r',label='explained by chance')
plt.legend()

pval = (expl_var_perm >= pca_model.explained_variance_ratio_).mean(axis=0)
optNum = np.where(pval > 0.05)[0][0]



pca_dat = pca_model.transform(diff_tunining)[:,:optNum]

# get the pca component number


kk = 1
plt.figure()
for n_neighbors in [5,10,20,40]:
    fit = umap.UMAP(n_neighbors=n_neighbors)
    umap_res = fit.fit_transform(pca_dat)
    # tsne_res = umap(n_components=2, perplexity=optPerp, early_exaggeration=12.0, learning_rate=200.0, n_iter=maxiter).fit_transform(pca_dat)
    
    plt.subplot(2,2,kk)
    plt.scatter(umap_res[:,0],
                 umap_res[:,1],s=10,c='g',alpha=0.4)
    
    # plt.scatter(umap_res[condition_label==0.005,0],
    #             umap_res[condition_label==0.005,1],s=10,c='g',alpha=0.4)
    
    # plt.scatter(umap_res[condition_label==0.0001,0],
    #             umap_res[condition_label==0.0001,1],s=10,c='r',alpha=0.4)
    
    
    
    if n_neighbors == 20:
        final_fit = deepcopy(fit)

    kk += 1 



proj_data  = final_fit.transform(pca_dat)

center_point = np.argmin((proj_data[:,0] - 0.9)**2 + (proj_data[:,1] - 11)**2)
srt_idx = np.argsort((proj_data[center_point,0] - proj_data[:,0] )**2 + (proj_data[center_point,1] - proj_data[:,1] )**2)


plt.figure()
plt.subplot(221)
for k in range(0,5):
    if condition_label[srt_idx[k]] == 0.005:
        color = 'g'
        label='HD'
    else:
        color = 'r'
        label = 'LD'
    plt.plot(response_funcs[srt_idx[k],:],color=color,label =label)

center_point = np.argmin((proj_data[:,0] - 8.8)**2 + (proj_data[:,1] - 7)**2)
srt_idx = np.argsort((proj_data[center_point,0] - proj_data[:,0] )**2 + (proj_data[center_point,1] - proj_data[:,1] )**2)

plt.subplot(222)
for k in range(0,5):
    if condition_label[srt_idx[k]] == 0.005:
        color = 'g'
        label='HD'
    else:
        color = 'r'
        label = 'LD'
    plt.plot(response_funcs[srt_idx[k],:],color=color,label =label)


center_point = np.argmin((proj_data[:,0] - 0.8)**2 + (proj_data[:,1] - 7)**2)
srt_idx = np.argsort((proj_data[center_point,0] - proj_data[:,0] )**2 + (proj_data[center_point,1] - proj_data[:,1] )**2)

plt.subplot(223)
for k in range(0,5):
    if condition_label[srt_idx[k]] == 0.005:
        color = 'g'
        label='HD'
    else:
        color = 'r'
        label = 'LD'
    plt.plot(response_funcs[srt_idx[k],:],color=color,label =label)
    


center_point = np.argmin((proj_data[:,0] - 0.8)**2 + (proj_data[:,1] - 11)**2)
srt_idx = np.argsort((proj_data[center_point,0] - proj_data[:,0] )**2 + (proj_data[center_point,1] - proj_data[:,1] )**2)

plt.subplot(224)
for k in range(0,5):
    if condition_label[srt_idx[k]] == 0.005:
        color = 'g'
        label='HD'
    else:
        color = 'r'
        label = 'LD'
    plt.plot(response_funcs[srt_idx[k],:],color=color,label =label)


# plot example pairs
# response_funcs = np.vstack((sign_tuning_hd,sign_tuning_ld))
# id_units = np.hstack((unit_id_hd,unit_id_ld))
# condition_label = np.array([0.005]*sign_tuning_hd.shape[0]+[0.0001]*sign_tuning_ld.shape[0])


hd_projct = np.zeros((0,2))
ld_projct = np.zeros((0,2))

hd_tun = np.zeros((0,sign_tuning_hd.shape[1]))
ld_tun = np.zeros((0,sign_tuning_hd.shape[1]))

for ID in units_id_tensor:
    # filt = id_units == ID
    # if np.sum(filt) != 2:
    #     continue
    # hd_projct = np.vstack((hd_projct,proj_data[(units_id_tensor == ID) ,:]))
    # ld_projct = np.vstack((ld_projct,proj_data[(units_id_tensor == ID),:]))
    
    hd_tun = np.vstack((hd_tun,resp_tensor[units_id_tensor == ID,0,:]))
    ld_tun = np.vstack((ld_tun,resp_tensor[units_id_tensor == ID,1,: ]))


# srt_indx = np.argsort(np.linalg.norm(hd_projct-ld_projct,axis=1))

# plt.figure()
# cc = 1
# for idx in srt_indx[-8:]:
#     plt.subplot(4,2,cc)
#     plt.plot(hd_tun[idx],label='HD')
#     plt.plot(ld_tun[idx],label='LD')
#     plt.legend()
#     cc+=1

# plt.figure()
# cc = 1
# for idx in srt_indx[:8]:
#     plt.subplot(4,2,cc)
#     plt.plot(hd_tun[idx],label='HD')
#     plt.plot(ld_tun[idx],label='LD')
#     plt.legend()
#     cc+=1

# corr vec
tuning_sim = np.zeros(hd_tun.shape[0])
for k in range(hd_tun.shape[0]):
    tuning_sim[k] = sts.pearsonr(hd_tun[k], ld_tun[k])[0]

srt_indx = np.argsort(np.sum((hd_tun-ld_tun)**2,axis=1))
srt_indx = np.argsort(tuning_sim)

plt.figure(figsize=(8,8))
plt.suptitle('Tuning similarity - most correlated')
cc = 1


for idx in srt_indx[-25:]:
    plt.subplot(5,5,cc)
    plt.plot(hd_tun[idx],label='HD')
    plt.plot(ld_tun[idx],label='LD')
    plt.title('%.3f'%sts.pearsonr(hd_tun[idx], ld_tun[idx])[0])
    
    # plt.legend()
    cc+=1
    
plt.tight_layout()

plt.figure(figsize=(8,8))
plt.suptitle('Tuning similarity - least correlated')

cc = 1
for idx in srt_indx[:25]:
    plt.subplot(5,5,cc)
    plt.plot(hd_tun[idx],label='HD')
    plt.plot(ld_tun[idx],label='LD')
    plt.title('%.3f'%sts.pearsonr(hd_tun[idx], ld_tun[idx])[0])

    
    # plt.legend()
    cc+=1
    
plt.tight_layout()