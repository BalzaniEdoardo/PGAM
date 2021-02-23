#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:16:36 2021

@author: edoardo
"""
import numpy as np
import os,re
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

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

def hist_equating_selection(ppc_rates,mst_rates):
    
    min_rate = np.min(np.hstack((ppc_rates,mst_rates)))
    max_rate = np.max(np.hstack((ppc_rates,mst_rates)))
    # set a threshold for 30hz
    max_rate = min(max_rate, 30)
    
    range_bin = np.linspace(min_rate,max_rate,10)
    sele_ppc = []
    sele_mst = []
    for k in range(range_bin.shape[0]-1):
        idx_ppc = np.where((ppc_rates >= range_bin[k]) * (ppc_rates < range_bin[k+1]))[0]
        idx_mst = np.where((mst_rates >= range_bin[k]) * (mst_rates < range_bin[k+1]))[0]
        
        num_sele = min(idx_mst.shape[0],idx_ppc.shape[0])
        
        if num_sele==0:
            continue
        
        sele_ppc = np.hstack((sele_ppc,np.random.choice(idx_ppc,size=num_sele,replace=False)))
        sele_mst = np.hstack((sele_mst,np.random.choice(idx_mst,size=num_sele,replace=False)))

    sele_mst = np.array(sele_mst,dtype=int)
    sele_ppc = np.array(sele_ppc,dtype=int)
    return sele_ppc,sele_mst

filtwidth = 10

t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)
var = 't_reward'

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

    tmp = np.zeros((1,4))
    kk = 0
    for area in ['MST','PPC','PFC','VIP']:
        tmp[0,kk] = (unit_info['brain_area'] == area).sum()
        kk += 1
    num_unit_x_area = np.vstack((num_unit_x_area,tmp))
    sess_list = np.hstack((sess_list, [session]))
    

session = 'm53s31'
thresh = num_unit_x_area  > 20
select = thresh[:,1] & thresh[:,0] 
resampling = 40
score_session = {}

firing_rates = {}
hist_matched_firing_rates = {}

for session in sess_list[select]:
    dat = np.load(os.path.join(npz_folder,session+'.npz'),allow_pickle=True)
    concat = dat['data_concat'].all()
    var_names = dat['var_names']
    X = concat['Xt']
    spikes = concat['Yt']
    trial_ind = concat['trial_idx']
    unit_info = dat['unit_info'].all()
    brain_area = unit_info['brain_area']
    
    sm_spikes = pop_spike_convolve(spikes,  concat['trial_idx'], h)
    
    endog = np.squeeze(X[:,var_names == var])
    idxOFF = np.where(endog==1)[0]
    for ii in idxOFF:
        if var == 't_stop' or var == 't_move':
            endog[ii-25:ii+25] = 1
        else:
            endog[ii-50:ii] = 1
            
    firing_rates[session] = {}
    firing_rates[session]['MST'] = (spikes[:,brain_area=='MST']).mean(axis=0)/0.006
    firing_rates[session]['PPC'] = (spikes[:,brain_area=='PPC']).mean(axis=0)/0.006
    firing_rates[session]['PFC'] = (spikes[:,brain_area=='PFC']).mean(axis=0)/0.006
    
    all_trial = np.unique(trial_ind)
    test_tr = all_trial[::10]
    train_tr = np.sort(np.array(list(set(all_trial).difference(set(test_tr)))))
    bool_test = np.zeros(trial_ind.shape,dtype=bool)
    bool_train = np.zeros(trial_ind.shape,dtype=bool)
    for tr in all_trial:
        if tr in test_tr:
            bool_test[trial_ind==tr] = True
        else:
            bool_train[trial_ind==tr] = True
    
    idx_ppc,idx_mst = hist_equating_selection(firing_rates[session]['PPC'], firing_rates[session]['MST'])

    num_ppc = ( unit_info['brain_area']  == 'PPC').sum()
    num_mst = ( unit_info['brain_area']  == 'MST').sum()
    
   
    
    alpha = 0.001
    score_session[session] = {}
    
    
    for kk in range(resampling):
        print(kk,resampling)
        sm_spk_ppc = sm_spikes[:, unit_info['brain_area'] == 'PPC']
        sm_spk_mst = sm_spikes[:, unit_info['brain_area'] == 'MST']
        if num_ppc > num_mst:
           # sub_sel = np.random.choice(np.arange(num_ppc), size=num_mst, replace=False)
           # sm_spk_ppc = sm_spk_ppc[:, sub_sel]
           sm_spk_ppc = sm_spk_ppc[:, idx_ppc]
           sm_spk_mst = sm_spk_mst[:, idx_mst]
           refit_ppc = True
        
        else:
           # sub_sel = np.random.choice(np.arange(num_mst),size=num_ppc,replace=False)
           # sm_spk_mst = sm_spk_mst[:, sub_sel]
           sm_spk_ppc = sm_spk_ppc[:, idx_ppc]
           sm_spk_mst = sm_spk_mst[:, idx_mst]
           refit_ppc = False
        
        
        clf = make_pipeline(StandardScaler(), 
                            SGDClassifier(max_iter=1000, tol=1e-3,class_weight="balanced"))
 
        
       
        if kk == 0:
            score_session[session]['PPC'] = []
            score_session[session]['MST'] = []
            if refit_ppc:
                train_spk = sm_spk_mst[bool_train]
                test_spk = sm_spk_mst[bool_test]
                clf.fit(train_spk,endog[bool_train])
                scr = balanced_accuracy_score(endog[bool_test],clf.predict(test_spk))
                score_session[session]['MST'] = np.hstack((score_session[session]['MST'],[scr]))
            else:
                train_spk = sm_spk_ppc[bool_train]
                test_spk = sm_spk_ppc[bool_test]
                clf.fit(train_spk,endog[bool_train])
                scr = balanced_accuracy_score(endog[bool_test],clf.predict(test_spk))
                score_session[session]['PPC'] = np.hstack((score_session[session]['PPC'],[scr]))
  
        if refit_ppc:    
            train_spk = sm_spk_ppc[bool_train]
            test_spk = sm_spk_ppc[bool_test]
            clf.fit(train_spk,endog[bool_train])
            scr = balanced_accuracy_score(endog[bool_test],clf.predict(test_spk))
            score_session[session]['PPC'] = np.hstack((score_session[session]['PPC'],[scr]))
        
        else:    
            train_spk = sm_spk_mst[bool_train]
            test_spk = sm_spk_mst[bool_test]
            clf.fit(train_spk,endog[bool_train])
            scr = balanced_accuracy_score(endog[bool_test],clf.predict(test_spk))
            score_session[session]['MST'] = np.hstack((score_session[session]['MST'],[scr]))
           
    print(session, 'PPC:',num_ppc, 'MST:',num_mst)
    print(score_session[session])
           # if kk == 0:
           #     model = lasso(alpha=alpha)
           #     res = model.fit(sm_spk_mst[bool_train], eye_pos[bool_train])
           #     scr = res.score(sm_spk_mst[bool_test], eye_pos[bool_test])
           #     score_session[session]['PFC'] = scr
           #     score_session[session]['PPC'] = []


np.save('MST_decoding_linSVM_%s.npy'%var,score_session)