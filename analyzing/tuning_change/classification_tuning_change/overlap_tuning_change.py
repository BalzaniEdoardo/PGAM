#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:56:20 2021

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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


plt.close('all')
condition_use = 'density'

area = 'PFC'

session = 'm53s41'
variable = 'rad_path'
plot_control = True

cond_extract = {'controlgain':{'controlgain':[1,2],'odd':[0,1]}, 'density':{'odd':[0,1],'density':[0.005,0.0001]},'ptb':{'odd':[0,1],'ptb':[0,1]}}

data_path = '/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_change/significance_tuning_function_change/'
npz_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'
list_fh = os.listdir(data_path)
first = True

neu_info_id = []
neu_info_session = []
pattern = '^newTest_m\d+s\d+_[a-z]+_tuningChange.npz$'


if not plot_control:
    condA, condB = cond_extract[condition_use][condition_use]
else:
    condA, condB = cond_extract[condition_use]['odd']

for name in list_fh:
    if not name.endswith('tuningChange.npz'):
        continue
    if not condition_use in name:
        continue
    if not re.match(pattern, name):
        continue
    if plot_control:
        condition = 'odd'
    else:
        condition = condition_use
    session = name.split('_')[1]
    print(session)
    dat = np.load(os.path.join(data_path,'newTest_%s_%s_tuningChange.npz'%(session,condition)),
        allow_pickle=True)
    npz_dat = np.load(npz_path%session,allow_pickle=True)
    unit_info = npz_dat['unit_info'].all()
    
    tensor_A = dat['tensor_A']
    tensor_B = dat['tensor_B']
    index_dict_A = dat['index_dict_A'].all()
    index_dict_B = dat['index_dict_B'].all()
    
    
    var_sign = dat['var_sign']
    sel_area = unit_info['brain_area'][var_sign['unit'] - 1] == area
    
    var_sign = var_sign[sel_area]
    
    unit_list = dat['unit_list']
    sel_area = unit_info['brain_area'][unit_list - 1] == area
    unit_list = unit_list[sel_area]
    tensor_A = tensor_A[sel_area]
    tensor_B = tensor_B[sel_area]
    
    pv_th = 0.01
    
    
    # Filter significant
    volume_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'
    
    dict_xlim = {'rad_vel':(0.,175),
                 'ang_vel':(-60,60),
                 'rad_path':(0,300),
                 'ang_path':(-60,60),
                 'rad_target':(5,330),
                 'ang_target':(-35,35),
                 'hand_vel1':(-100., 100),
                 'hand_vel2':(-100,100),
                 'phase':(-np.pi,np.pi),
                 'rad_acc':(-800,800),
                 'ang_acc':(-100,100),
                 
                'lfp_alpha':(-np.pi,np.pi),
                'lfp_beta':(-np.pi,np.pi),
                'lfp_theta':(-np.pi,np.pi),
                 't_move':(-0.36,0.36),
                 't_flyOFF':(-0.36,0.36),
                 't_stop':(-0.36,0.36),
                 't_reward':(-0.36,0.36),'eye_vert':(-2,2),
                 'eye_hori':(-2,2)}
    
    
    
    # filt = (var_sign['variable'] == variable) * (var_sign['p-val'] < pv_th)
    
    sbcnt = 1
    oldsub =-1
    
    for unit in range(tensor_A.shape[0]):
    
        # if sbcnt%20 == 1:
        #     if unit != 0:
        #         plt.tight_layout()
        #     sbcnt = 1
        #     if oldsub != sbcnt:
        #         oldsub = sbcnt
        #         plt.figure(figsize=(10, 8))
        #         sb_open = []
        # if not sbcnt in sb_open:
        #     plt.subplot(5, 4, sbcnt)
        #     oldsub = sbcnt
            
        #     sb_open += [ sbcnt]
        
        # print(sb_open)
    
        neuron_id = unit_list[unit]
        
    
        filt = (var_sign['variable'] == variable) * (var_sign['unit'] == neuron_id)
        pval = var_sign[filt]['p-val']
        # check_tuned = True
        cond_list = []
        cnt_tun=0
        for key in var_sign.dtype.names:
            if 'p-val ' in key:
                cond_list += [float(key.split(' ')[2])]
                if var_sign[filt][key] < 0.001:
                    cnt_tun += 1
                    # check_tuned = False
        check_tuned = cnt_tun == 0
        if pval < 0.001:
            
            val = cond_list[0]
            with open(volume_fld+'/gam_%s/fit_results_%s_c%d_%s_%.4f.dill'% (session,session,neuron_id,condition,val), "rb") as dill_file:
                gam_res_dict = dill.load(dill_file)
                gam_res_A = deepcopy(gam_res_dict['full'])
                
            val = cond_list[1]
            with open(volume_fld+'/gam_%s/fit_results_%s_c%d_%s_%.4f.dill'% (session,session,neuron_id,condition,val), "rb") as dill_file:
                gam_res_dict = dill.load(dill_file)
                gam_res_B = gam_res_dict['full']
                
            if gam_res_A.smooth_info[variable]['is_temporal_kernel']:
                dim_kern = gam_res_A.smooth_info[variable]['basis_kernel'].shape[0]
                knots_num = gam_res_A.smooth_info[variable]['knots'][0].shape[0]
    
                idx_select = np.arange(0,dim_kern,(dim_kern+1)//knots_num)
    
                impulse = np.zeros(dim_kern)
                impulse[(dim_kern-1)//2] = 1
                x = 0.006*np.linspace(-(dim_kern+1)/2,(dim_kern-1)/2,dim_kern)
                fX_A, fX_p_ci_A, fX_m_ci_A = gam_res_A.smooth_compute([impulse], variable, perc=0.99,)
                fX_B, fX_p_ci_B, fX_m_ci_B = gam_res_B.smooth_compute([impulse], variable, perc=0.99,)
                
            else:
                x = np.linspace(dict_xlim[variable][0], dict_xlim[variable][1], 100)
                fX_A, fX_p_ci_A, fX_m_ci_A = gam_res_A.smooth_compute([x], variable, perc=0.99,)
                fX_B, fX_p_ci_B, fX_m_ci_B = gam_res_B.smooth_compute([x], variable, perc=0.99,)
            ba = gam_res_dict['brain_area']
                
            fX_B = fX_B - np.nanmean(fX_B-fX_A)
            fX_m_ci_B = fX_m_ci_B - np.nanmean(fX_B-fX_A)
            fX_B = fX_B - np.nanmean(fX_B-fX_A)
            
            if first:
                matrix_cond_A = np.zeros((0,3,fX_A.shape[0]))
                matrix_cond_B = np.zeros((0,3,fX_B.shape[0]))
                first = False
            tmp = np.zeros((1,3,fX_A.shape[0]))
            tmp[0,0,:] = fX_A
            tmp[0,1,:] = fX_m_ci_A
            tmp[0,2,:] = fX_p_ci_A
            
            matrix_cond_A = np.vstack((matrix_cond_A,tmp))
            
            tmp = np.zeros((1,3,fX_A.shape[0]))
            tmp[0,0,:] = fX_B
            tmp[0,1,:] = fX_m_ci_B
            tmp[0,2,:] = fX_p_ci_B
            matrix_cond_B = np.vstack((matrix_cond_B,tmp))
            neu_info_session += [session]
            neu_info_id += [neuron_id]
            if (neuron_id == 36) and (session == 'm53s48'):
                print('ciao')
                xxxx=1
            
            # plt.title('%s c%d - %f' % (ba,neuron_id, pval[0]))
    
    
    
            
            # plt.figure()
            # pA, = plt.plot(x, fX_A)
            # pB, = plt.plot(x, fX_B)
            # plt.fill_between(x,fX_m_ci_A,fX_p_ci_A,color=pA.get_color(),alpha=0.4)
            # plt.fill_between(x,fX_m_ci_B,fX_p_ci_B,color=pB.get_color(),alpha=0.4)
    
    
    
       
            sbcnt += 1
            
    # plt.tight_layout()

model = PCA()
pca_fit = model.fit(matrix_cond_A[:,0])
pca_prj = pca_fit.transform(matrix_cond_A[:,0])

# dbs = DBSCAN(eps=.5)
# dbs = dbs.fit(pca_prj[:,:3])

# plt.figure()

# for label in np.unique(dbs.labels_):
#     if label == -1:
#         continue
#     plt.scatter(pca_prj[dbs.labels_==label,0],pca_prj[dbs.labels_==label,1])
    

kmn = KMeans(2)
kmn = kmn.fit(pca_prj[:,:3])
col_dict = {}
plt.figure()
plt.title('PCA - tuning function')
for label in np.unique(kmn.labels_):
    if label == -1:
        continue
 
    scat = plt.scatter(pca_prj[kmn.labels_==label,0],pca_prj[kmn.labels_==label,1])
    
    col_dict[label] = scat.get_facecolor()[0][0:3]
plt.savefig('pca_cluster_tun_%s_%s_%s.png'%(area,condition,variable))


plt.figure(figsize=(8,8))
k=0
for label in np.unique(kmn.labels_):
    if len(np.unique(kmn.labels_)) <= 4:
        plt.subplot(2,2,k+1)
    else:
        plt.subplot(3,2,k+1)
    
    
    if condition == 'controlgain' and 'vel' in variable:
        plt.plot(matrix_cond_A[kmn.labels_==label,0,::20].T,color=col_dict[label])
    else:
        plt.plot(matrix_cond_A[kmn.labels_==label,0,:].T,color=col_dict[label])
    k+=1
    
if plot_control:
    plt.savefig('raw_cluster_tun_%s_control_%s_%s.png'%(area,condition,variable))

else:
    plt.savefig('raw_cluster_tun_%s_%s_%s.png'%(area,condition,variable))


plt.figure(figsize=(8,8))
k=0
for label in np.unique(kmn.labels_):
    if len(np.unique(kmn.labels_)) <= 4:
        plt.subplot(2,2,k+1)
    else:
        plt.subplot(3,2,k+1)

    
    plt.title('cluster %d'%label)
    
    mn = matrix_cond_A[kmn.labels_==label,0,:].mean(axis=0)
    sd = matrix_cond_A[kmn.labels_==label,0,:].std(axis=0)

    if condition_use == 'controlgain' and 'vel' in variable:
        plt.plot(mn[::20],color='b',label='%s: %.4f'%(condition,condA))
        plt.fill_between(range(mn[::20].shape[0]),mn[::20]-sd[::20],mn[::20]+sd[::20],color='b',alpha=0.4)
    else:
        plt.plot(mn,color='b',label='%s: %.4f'%(condition,condA))
    
        plt.fill_between(range(mn.shape[0]),mn-sd,mn+sd,color='b',alpha=0.4)
    

    mn = matrix_cond_B[kmn.labels_==label,0,:].mean(axis=0)
    sd = matrix_cond_B[kmn.labels_==label,0,:].std(axis=0)
    if condition_use == 'controlgain' and 'vel' in variable:
        plt.plot(mn[::20],color='y',label='%s: %.4f'%(condition,condB))
        plt.fill_between(range(mn[::20].shape[0]),mn[::20]-sd[::20],mn[::20]+sd[::20],color='y',alpha=0.4)
    else:
  
        plt.plot(mn,color='y',label='%s: %.4f'%(condition,condB))
        
        plt.fill_between(range(mn.shape[0]),mn-sd,mn+sd,color='y',alpha=0.4)



    k+=1
plt.legend()
if plot_control:
    plt.savefig('mean_cluster_tun_%s_control_%s_%s.png'%(area,condition,variable))
else:
    plt.savefig('mean_cluster_tun_%s_%s_%s.png'%(area,condition,variable))

plt.figure(figsize=(10,6))

kk=0
for k in range(10,25):
    plt.subplot(3,5,kk+1)
    kk+=1
    unit = k
    # plt.figure()
    pB, = plt.plot(matrix_cond_B[unit,0,:],color='y')
    plt.fill_between(range(matrix_cond_B.shape[2]),matrix_cond_B[unit,1,:],matrix_cond_B[unit,2,:],alpha=0.4,color=pB.get_color())
    
    pA, = plt.plot(matrix_cond_A[unit,0,:],'b')
    plt.fill_between(range(matrix_cond_A.shape[2]),matrix_cond_A[unit,1,:],matrix_cond_A[unit,2,:],alpha=0.4,color=pA.get_color())
    
plt.tight_layout()
if plot_control:
    plt.savefig('example_tun_%s_control_%s_%s.png'%(area,condition,variable))
else:
    plt.savefig('example_tun_%s_%s_%s.png'%(area,condition,variable))
    
    