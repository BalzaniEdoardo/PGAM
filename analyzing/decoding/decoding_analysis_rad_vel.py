#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:22:04 2021

@author: edoardo
"""
from sklearn.linear_model import LinearRegression as lnreg
from sklearn.linear_model import Lasso as lasso
import numpy as np
import matplotlib.pylab as plt
import os,re

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

    tmp = np.zeros((1,4))
    kk = 0
    for area in ['MST','PPC','PFC','VIP']:
        tmp[0,kk] = (unit_info['brain_area'] == area).sum()
        kk += 1
    num_unit_x_area = np.vstack((num_unit_x_area,tmp))
    sess_list = np.hstack((sess_list, [session]))
    

thresh = num_unit_x_area  > 20
select = thresh[:,0] & thresh[:,1] #& thresh[:,2] 

resampling = 50
score_session = {}
for session in sess_list[select]:
    dat = np.load(os.path.join(npz_folder,session+'.npz'),allow_pickle=True)

    concat = dat['data_concat'].all()
    var_names = dat['var_names']
    X = concat['Xt']
    rad_vel = X[:,var_names=='rad_vel']
    trial_ind = concat['trial_idx']
    unit_info = dat['unit_info'].all()
    brain_area = unit_info['brain_area']
    # rad_vel[trial_ind==5]
    
    spikes = concat['Yt']
    
    sm_spikes = pop_spike_convolve(spikes, trial_ind, h)


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


    num_ppc = (brain_area == 'PPC').sum()
    num_mst = (brain_area == 'MST').sum()
    
   
    
    alpha = 0.001
    score_session[session] = {}
    
    
    for kk in range(resampling):
        print(kk,resampling)
        sm_spk_ppc = sm_spikes[:, brain_area == 'PPC']
        sm_spk_mst = sm_spikes[:, brain_area == 'MST']
        
        if num_ppc > num_mst:
            sub_sel = np.random.choice(np.arange(num_ppc), size=num_mst, replace=False)
            sm_spk_ppc = sm_spk_ppc[:, sub_sel]
           
            if kk == 0:
                model = lasso(alpha=alpha)
                res = model.fit(sm_spk_mst[bool_train], rad_vel[bool_train])
                scr = res.score(sm_spk_mst[bool_test], rad_vel[bool_test])
                score_session[session]['MST'] = scr
                score_session[session]['PPC'] = []

               
            model = lasso(alpha=alpha)
            res = model.fit(sm_spk_ppc[bool_train], rad_vel[bool_train])
            scr = res.score(sm_spk_ppc[bool_test], rad_vel[bool_test])
            score_session[session]['PPC'] = np.hstack((score_session[session]['PPC'], [scr]))
            
           
        elif num_ppc < num_mst:
            sub_sel = np.random.choice(np.arange(num_mst),size=num_ppc,replace=False)
            sm_spk_mst = sm_spk_mst[:, sub_sel]
           
           
            if kk == 0:
                model = lasso(alpha=alpha)
                res = model.fit(sm_spk_ppc[bool_train], rad_vel[bool_train])
                scr = res.score(sm_spk_ppc[bool_test], rad_vel[bool_test])
                score_session[session]['PPC'] = scr
                score_session[session]['MST'] = []
                
            model = lasso(alpha=alpha)
            res = model.fit(sm_spk_mst[bool_train], rad_vel[bool_train])
            scr = res.score(sm_spk_mst[bool_test], rad_vel[bool_test])
            score_session[session]['MST'] = np.hstack((score_session[session]['MST'], [scr]))
            
        else:
            model = lasso(alpha=alpha)
           
            res = model.fit(sm_spk_ppc[bool_train], rad_vel[bool_train])
            scr = res.score(sm_spk_ppc[bool_test], rad_vel[bool_test])
            score_session[session]['PPC'] = scr
            
            res = model.fit(sm_spk_mst[bool_train], rad_vel[bool_train])
            scr = res.score(sm_spk_mst[bool_test], rad_vel[bool_test])
            score_session[session]['MST'] = scr

np.save('MST_decoding_rad_vel_with_subsamp.npy', score_session)
