#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:52:31 2021

@author: edoardo
"""
from sklearn.linear_model import LinearRegression as lnreg
from sklearn.linear_model import Lasso as lasso
import numpy as np
import matplotlib.pylab as plt
import os,re, sys
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from statsmodels.distributions import ECDF

sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')

from data_handler import *


def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        sm_x[sel] = np.convolve(x[sel],filter,mode='same')
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk



filtwidth = 10

t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)


npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'
pattern = '^m\d+s\d+.npz$'
num_unit_x_area = np.zeros((0,4))
sess_list = []
for fh in os.listdir(npz_folder):
    if not re.match(pattern,fh):
        continue
    session = fh.split('.')[0]
    print(session)
    dat = np.load(os.path.join(npz_folder,fh),allow_pickle=True)
    unit_info = dat['unit_info'].all()
    trial_type = dat['info_trial'].all().trial_type 
    
    density = np.unique(trial_type['density'][~np.isnan(trial_type['density'])])
    if not ((0.005 in density) and (0.0001 in density)):
        continue
        
              
    tmp = np.zeros((1,4))
    kk = 0
    for area in ['MST','PPC','PFC','VIP']:
        tmp[0,kk] = (unit_info['brain_area'] == area).sum()
        kk += 1
        
    num_unit_x_area = np.vstack((num_unit_x_area,tmp))
    sess_list = np.hstack((sess_list, [session]))


bl_monkey = np.zeros(num_unit_x_area.shape[0],dtype=bool)

# select schro
kk = 0
for session in sess_list:
    if 'm53' in session:
        bl_monkey[kk] = True
    kk+=1
    

# DECODE PFC (set an arbitrary threshold of the units)
# ba = 'PFC'
# col_ba = 2
alpha_vec = [0.000001,0.001,1,10,100]
parameters = {'alpha':[0.0001,0.001,0.1,1,10,10]}

sel = ((num_unit_x_area[:,0]>=40) |  (num_unit_x_area[:,1]>=40) | (num_unit_x_area[:,2]>=40) ) & bl_monkey
alpha = 0.001
kk = 0
decoding_results_x_var = {}
# var_list = ['rad_path','ang_path','rad_vel','ang_vel','rad_target','ang_target']
var_list = ['rad_path']#,'eye_vert']
for var in var_list:
    decoding_results_x_var[var] = {'PFC':{},'PPC':{},'MST':{}}
for session in sess_list[sel]:
    print(session, '%d/%d'%(kk+1,sel.sum()))
    dat = np.load(os.path.join(npz_folder,session+'.npz'),allow_pickle=True)
    concat = dat['data_concat'].all()
    var_names = dat['var_names']
    X = concat['Xt']
    spikes = concat['Yt']
    sm_spikes = pop_spike_convolve(spikes,  concat['trial_idx'], h)
            
    for var in var_list  :
        endog = X[:,var_names==var]
        if var == 'ang_target':
            endog[np.abs(endog) > 50] = np.nan
        endog = np.squeeze(endog)
        keep  = ~np.isnan(endog)
        endog = endog[keep]
        

        trial_ind = concat['trial_idx'][keep]
        unit_info = dat['unit_info'].all()
        brain_area = unit_info['brain_area']
        
        
        trial_type = dat['info_trial'].all().trial_type 
        
        id_HD = np.where(trial_type['density'] == 0.005)[0]
        id_LD = np.where(trial_type['density'] == 0.0001)[0]

            
        # trial selection
        test_tr_HD = id_HD[::10]
        train_tr_HD = np.sort(np.array(list(set(id_HD).difference(set(test_tr_HD)))))
        
        test_tr_LD = id_LD[::10]
        train_tr_LD = np.sort(np.array(list(set(id_LD).difference(set(test_tr_LD)))))
        
        # subsample time points
        # train set
        bl_train_HD = np.zeros(trial_ind.shape[0],dtype=bool)
        bl_train_LD = np.zeros(trial_ind.shape[0],dtype=bool)
        for tr in train_tr_HD:
            bl_train_HD[trial_ind == tr] = True
            
        for tr in train_tr_LD:
            bl_train_LD[trial_ind == tr] = True
        
        if bl_train_HD.sum() >  bl_train_LD.sum():
            
            while bl_train_HD.sum() >  bl_train_LD.sum():
                # print(bl_train_HD.sum())
                tr = np.random.choice(train_tr_HD, size=1)
                bl_train_HD[trial_ind==tr] = False
                
        elif bl_train_HD.sum() <  bl_train_LD.sum():
            while bl_train_HD.sum() <  bl_train_LD.sum():
                tr = np.random.choice(train_tr_LD, size=1)
                bl_train_LD[trial_ind==tr] = False
        
        # train set
        bl_test_HD = np.zeros(trial_ind.shape[0],dtype=bool)
        bl_test_LD = np.zeros(trial_ind.shape[0],dtype=bool)
        for tr in test_tr_HD:
            bl_test_HD[trial_ind == tr] = True
            
        for tr in test_tr_LD:
            bl_test_LD[trial_ind == tr] = True
        
        if bl_test_HD.sum() >  bl_test_LD.sum():
            
            while bl_test_HD.sum() >  bl_test_LD.sum():
                # print(bl_train_HD.sum())
                tr = np.random.choice(test_tr_HD, size=1)
                bl_test_HD[trial_ind==tr] = False
                
        elif bl_test_HD.sum() <  bl_test_LD.sum():
            while bl_test_HD.sum() <  bl_test_LD.sum():
                tr = np.random.choice(test_tr_LD, size=1)
                bl_test_LD[trial_ind==tr] = False
                
        
        assert((bl_test_HD*bl_train_HD).sum()==0)
        assert((bl_test_LD*bl_train_LD).sum()==0)
        assert((bl_test_LD*bl_train_LD).sum()==0)
        assert((bl_test_HD*bl_train_HD).sum()==0)
        
    
        
        for ba,col in [('MST',0), ('PPC',1), ('PFC',2)]:
            if num_unit_x_area[sess_list==session, col] < 20:
                continue
            sm_spikes_ba = sm_spikes[keep]
            sm_spikes_ba = sm_spikes_ba[:, brain_area==ba]

            
            decoding_results_x_var[var][ba][session] = {}
            
            model = lasso()
            gdsHD = GridSearchCV(model, parameters)
            gdsHD.fit(sm_spikes_ba[bl_train_HD], endog[bl_train_HD])
            
            
        
            hd_pred_hd = gdsHD.predict(sm_spikes_ba[bl_test_HD])
            hd_pred_ld = gdsHD.predict(sm_spikes_ba[bl_test_LD])
            
            model = lasso()
            gdsLD = GridSearchCV(model, parameters)
            gdsLD.fit(sm_spikes_ba[bl_train_LD], endog[bl_train_LD])
            
            ld_pred_hd = gdsLD.predict(sm_spikes_ba[bl_test_HD])
            ld_pred_ld = gdsLD.predict(sm_spikes_ba[bl_test_LD])
            
            
            decoding_results_x_var[var][ba][session]['HD -> HD'] = hd_pred_hd
            decoding_results_x_var[var][ba][session]['HD -> LD'] = hd_pred_ld
            decoding_results_x_var[var][ba][session]['LD -> HD'] = ld_pred_hd
            decoding_results_x_var[var][ba][session]['LD -> LD'] = ld_pred_ld
            decoding_results_x_var[var][ba][session]['true HD'] = endog[bl_test_HD]
            decoding_results_x_var[var][ba][session]['true LD'] = endog[bl_test_LD]
    kk+=1
    
for var in decoding_results_x_var.keys():
    np.save('predict_density_%s.npy'%var,decoding_results_x_var[var])
    
var = 'rad_path'
for ba in ['PPC','PFC']:
    plt.suptitle(ba + ' rad path: predicted')

    plt.figure(figsize=(12,6))
    plt.suptitle(ba)
    cc = 1
    for session in  decoding_results_x_var[var][ba].keys():
        hd_hd = decoding_results_x_var[var][ba][session]['HD -> HD']
        ld_hd = decoding_results_x_var[var][ba][session]['LD -> HD']
        hd_ld = decoding_results_x_var[var][ba][session]['HD -> LD']
        ld_ld = decoding_results_x_var[var][ba][session]['LD -> LD']
        plt.subplot(2,5,cc)
        ldhd_cdf = ECDF(ld_hd)
        hdhd_cdf = ECDF(hd_hd)
        hdld_cdf = ECDF(hd_ld)
        ldld_cdf = ECDF(ld_ld)
        
        xx = np.linspace(-10,262,1000)
        
        p, = plt.plot(xx,hdhd_cdf(xx),label='HD->HD')
        plt.plot(xx,hdld_cdf(xx),label='HD->LD',color=p.get_color(),ls='--')
        q, = plt.plot(xx,ldld_cdf(xx),label='LD->LD')
        plt.plot(xx,ldhd_cdf(xx),label='LD->HD', color=q.get_color(),ls='--')
        
        


        plt.legend()
        plt.title(session)
        cc+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('%s_rad_path_decoding_comp.pdf'%ba)



    
var = 'rad_path'
for ba in ['PPC','PFC']:
    plt.figure(figsize=(12,6))
    plt.suptitle(ba + ' rad path: predicted - true')
    cc = 1
    for session in  decoding_results_x_var[var][ba].keys():
        true_hd = decoding_results_x_var[var][ba][session]['true HD']
        true_ld = decoding_results_x_var[var][ba][session]['true LD']
        
        hd_hd = decoding_results_x_var[var][ba][session]['HD -> HD'] - true_hd
        ld_hd = decoding_results_x_var[var][ba][session]['LD -> HD'] - true_hd 
        hd_ld = decoding_results_x_var[var][ba][session]['HD -> LD'] - true_ld
        ld_ld = decoding_results_x_var[var][ba][session]['LD -> LD'] - true_ld
        plt.subplot(2,5,cc)
        ldhd_cdf = ECDF(ld_hd)
        hdhd_cdf = ECDF(hd_hd)
        hdld_cdf = ECDF(hd_ld)
        ldld_cdf = ECDF(ld_ld)
        
        xx = np.linspace(-350,250,1000)
        
        p, = plt.plot(xx,hdhd_cdf(xx),label='HD->HD')
        plt.plot(xx,hdld_cdf(xx),label='HD->LD',color=p.get_color(),ls='--')
        q, = plt.plot(xx,ldld_cdf(xx),label='LD->LD')
        plt.plot(xx,ldhd_cdf(xx),label='LD->HD', color=q.get_color(),ls='--')
        
        


        plt.legend()
        plt.title(session)
        cc+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('%s_delta_rad_path_decoding_comp.pdf'%ba)
    
var = 'rad_path'
for ba in ['PPC','PFC']:
    plt.figure(figsize=(12,6))
    plt.suptitle(ba + ' rad path:  - true')
    cc = 1
    for session in  decoding_results_x_var[var][ba].keys():
        true_hd = decoding_results_x_var[var][ba][session]['true HD']
        true_ld = decoding_results_x_var[var][ba][session]['true LD']
        
        hd_m_ld = decoding_results_x_var[var][ba][session]['LD -> LD'] - decoding_results_x_var[var][ba][session]['HD -> LD']
        ld_m_hd = decoding_results_x_var[var][ba][session]['HD -> HD'] - decoding_results_x_var[var][ba][session]['LD -> HD']
        plt.subplot(2,5,cc)
        hdmld_cdf = ECDF(hd_m_ld)
        ldmhd_cdf = ECDF(ld_m_hd)
        
        
        xx = np.linspace(-350,250,1000)
        
        p, = plt.plot(xx,hdmld_cdf(xx),label='LDLD-HDLD')
        # plt.plot(xx,hdld_cdf(xx),label='HD->LD',color=p.get_color(),ls='--')
        # q, = plt.plot(xx,ldld_cdf(xx),label='LD->LD')
        plt.plot(xx,ldmhd_cdf(xx),label='HDHD-LDHD')
        if cc ==1:
            plt.legend()
        cc+=1

#         plt.legend()
#         plt.title(session)
#         cc+=1
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plt.savefig('%s_delta_rad_path_decoding_comp.pdf'%ba)
    

var = 'rad_path'
# plt.figure(figsize=(12,6))
   # plt.suptitle()
means_dict_hdhd = {}
means_dict_ldld = {}
for ba in ['PPC','PFC']:
    means_dict_hdhd[ba] = []
    means_dict_ldld[ba] = []
    cc = 1
    for session in  decoding_results_x_var[var][ba].keys():
        true_hd = decoding_results_x_var[var][ba][session]['true HD']
        true_ld = decoding_results_x_var[var][ba][session]['true LD']
        
        hd_hd = decoding_results_x_var[var][ba][session]['HD -> HD'] - true_hd
        ld_hd = decoding_results_x_var[var][ba][session]['LD -> HD'] - true_hd 
        hd_ld = decoding_results_x_var[var][ba][session]['HD -> LD'] - true_ld
        ld_ld = decoding_results_x_var[var][ba][session]['LD -> LD'] - true_ld
        # plt.subplot(2,5,cc)
        ldhd_cdf = ECDF(ld_hd)
        hdhd_cdf = ECDF(hd_hd)
        hdld_cdf = ECDF(hd_ld)
        ldld_cdf = ECDF(ld_ld)
        
        xx = np.linspace(-350,250,1000)
        means_dict_hdhd[ba] = np.hstack((means_dict_hdhd[ba], [hd_hd.mean()]))
        
        means_dict_ldld[ba] = np.hstack((means_dict_ldld[ba], [ld_ld.mean()]))
        # p, = plt.plot(xx,hdhd_cdf(xx),label='%s HD->HD'%ba)
        # p = plt.hist(hd_hd,bins=30,label='%s HD->HD'%ba,alpha=0.4)
        # q = plt.hist(ld_ld,bins=30,label='%s LD->LD'%ba,alpha=0.4)
        # plt.plot([hd_hd.mean(),hd_hd.mean()],[0,2000], color=p[2][0].get_facecolor())
        # plt.plot([ld_ld.mean(),ld_ld.mean()],[0,2000], color=q[2][0].get_facecolor())

        # plt.plot(xx,hdld_cdf(xx),label='%s HD->LD'%ba,color=p.get_color(),ls='--')
        # q, = plt.plot(xx,ldld_cdf(xx),label='%s LD->LD'%ba,ls='--')
        # plt.plot(xx,ldhd_cdf(xx),label='LD->HD', color=q.get_color(),ls='--')
        
        


    #     plt.legend()
    #     plt.title(session)
    #     cc+=1
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plt.savefig('%s_delta_rad_path_decoding_comp.pdf'%ba)
    