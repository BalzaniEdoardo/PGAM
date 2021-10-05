#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:05:20 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys,os
main_dir = '/Users/edoardo/Work/Code/GAM_code/'
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(main_dir,'GAM_library'))
from behav_class import behavior_experiment,load_trial_types
from data_handler import *
from sklearn.linear_model import Lasso

sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')
from utils_loading import unpack_preproc_data
from scipy.interpolate import interp1d
from scipy.io import loadmat

session = 'm53s113'
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

def interpolate(xvar,tp, ev0, ev1, trial_idx, Xt, var_names):
    unq_tr = np.unique(trial_idx)

    rescaled_xvar = np.zeros((unq_tr.shape[0], tp, xvar.shape[1])) * np.nan

    cc = 0
    nannonidx = []
    for tr in unq_tr:
        print('tr: %d'%tr)
        try:
            idx0 = np.where(Xt[trial_idx==tr,var_names==ev0])[0][0]
            idx1 = np.where(Xt[trial_idx==tr,var_names==ev1])[0][0]
        except:
            print('skip tr: %d'%tr)
           
            cc += 1
            continue
        nannonidx += [cc]
        ts = np.arange(idx1-idx0)*0.006
        ts_resc = np.linspace(ts[0],ts[-1],tp)
        for k in range(xvar.shape[1]):
            interp = interp1d(ts, xvar[trial_idx==tr][idx0:idx1,k])
    
            rescaled_xvar[cc,:,k] = interp(ts_resc)
        # print(tr,np.sum(np.isnan(interp(ts_resc))) )
        cc += 1
    # rescaled_xvar = rescaled_xvar[np.array(nannonidx,dtype=int)]
    return rescaled_xvar

xdim = 60
dat = np.load('FA_dimEstimation_%d.npz'%xdim)
C = dat['C']
mu = dat['mu_post']
B = dat['B']
R = dat['sigma2'] #noise variances
trial_idx_fit = dat['trial_idx']
Cmu = np.dot(mu,C.T)
Bs = np.einsum('ij,tj->ti',B,dat['modelS'])

# smooth traj
filtwidth = 15
t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)
sm_rate = pop_spike_convolve(Cmu,trial_idx_fit,h)

# quick diim red
model = PCA(15)
fit = model.fit(sm_rate)
pca_sm = fit.transform(sm_rate)

 
# extract variables
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session

par_list = ['Xt', 'Yt','var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']

(Xt, yt,  var_names, trial_type,
  trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
  cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
  channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)


# check if info about internal is present?
non_nan = ~np.isnan(Xt[:,var_names=='rad_path']).flatten()
lasso = Lasso(alpha=0.001)
fit_lasso = lasso.fit(Cmu[non_nan],Xt[non_nan,var_names=='ang_path'])
print(fit_lasso.score(Cmu[non_nan],Xt[non_nan,var_names=='ang_path']))

rsc_firing = interpolate(dat['fr'],70, 't_move', 't_stop', trial_idx, Xt, var_names)


# extract subclass trials
base_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'
dat = loadmat(os.path.join(base_file, '%s.mat' % (session)))

pre_trial_dur = 0.0
post_trial_dur = 0.0
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'

exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur,
                        post_trial_dur=post_trial_dur,
                        lfp_beta=None, lfp_alpha=None,
                        lfp_theta=None, extract_lfp_phase=False,
                        use_eye='right', fhLFP=None, extract_fly_and_monkey_xy=True)

exp_data.set_filters('all', True)
# impose all replay trials
exp_data.filter = exp_data.filter + exp_data.info.get_replay(0, skip_not_ok=False)

t_targ = dict_to_vec(exp_data.behav.events.t_targ)
t_move = dict_to_vec(exp_data.behav.events.t_move)

t_start = np.min(np.vstack((t_move, t_targ)), axis=0) - pre_trial_dur
t_stop = dict_to_vec(exp_data.behav.events.t_end) + post_trial_dur

# bin_ts = time_stamps_rebin(exp_data.behav.time_stamps, binwidth_ms=20)
exp_data.spikes.bin_spikes(exp_data.behav.time_stamps, t_start=t_start, t_stop=t_stop, select=exp_data.filter)

var_names_conc = ('rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target',
              'eye_vert', 'eye_hori',
              'rad_acc', 'ang_acc')

time_pts, rate, sm_traj, raw_traj, fly_pos, cov_dict = exp_data.GPFA_YU_preprocessing_noTW(t_start, t_stop,
                                                                                            var_list=var_names_conc,binwidth_ms=None)

spikes, var_dict, trial_idx_concat = exp_data.concatenate_inputs('t_move','t_stop','t_reward',t_start=t_start,t_stop=t_stop)


brain_area = exp_data.spikes.brain_area


cLeft = np.array([0,50])
cCenter = np.array([0,150])
cRight = np.array([0,250])

radius = 40
radiius_center = 25
end_pt = np.zeros(sm_traj.shape)
for k in range(sm_traj.shape[0]):
    last_nonNan = np.where(~np.isnan(sm_traj[k, 0]))[0][-1]
    end_pt[k, 0] = sm_traj[k, 0][last_nonNan]
    end_pt[k, 1] = sm_traj[k, 1][last_nonNan]



selLeft = np.linalg.norm(end_pt - cLeft,axis=1) < radius
selCenter = np.linalg.norm(end_pt - cCenter,axis=1) < radiius_center
selRight = np.linalg.norm(end_pt - cRight,axis=1) < radius
print(selLeft.sum(),selCenter.sum(),selRight.sum())

plt.figure()
for k in np.where(selLeft)[0]:
    plt.plot(sm_traj[k, 0],sm_traj[k, 1],color='b',lw=0.4)
for k in np.where(selCenter)[0]:
    plt.plot(sm_traj[k, 0],sm_traj[k, 1],color='r',lw=0.4)
for k in np.where(selRight)[0]:
    plt.plot(sm_traj[k, 0],sm_traj[k, 1],color='g',lw=0.4)
plt.scatter(end_pt[selLeft,0],end_pt[selLeft,1],c='b')
plt.scatter(end_pt[selCenter,0],end_pt[selCenter,1],c='r')
plt.scatter(end_pt[selRight,0],end_pt[selRight,1],c='g')

    
plt.scatter([0],[0],marker='*',s=120,c='y')

plt.xlim(-100,100)
plt.title('trial divided by initial distance from origin')
plt.xlabel('cm')
plt.ylabel('cm')
plt.savefig('target_dist_trials.png')


# check that the content of the trjectory firing as info about the trial type
selected = np.hstack((np.where(selLeft)[0], 
                      np.where(selCenter)[0], 
                      np.where(selRight)[0]))
trial_type = np.hstack((np.zeros(selLeft.sum()),
                        np.ones(selCenter.sum()),
                        2*np.ones(selRight.sum()) ))

test = np.zeros(trial_type.shape,dtype=bool)
test[::10] = True

train = ~test

train_idx = selected[train]
test_idx = selected[test]

rsc_ang_path = interpolate(Xt[:, var_names=='ang_path'], 
                           70, 't_move', 't_stop', trial_idx_fit, Xt, var_names)
# rescale stim and latent
rsc_resid = interpolate(Cmu, 70, 't_move', 't_stop', trial_idx_fit, Xt, var_names)
rsc_stim = interpolate(Bs, 70, 't_move', 't_stop', trial_idx_fit, Xt, var_names)


modelFRLDA = rsc_firing.reshape(rsc_firing.shape[0],-1)
mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
fit_lda = mdl.fit(modelFRLDA[train_idx],trial_type[train])
print('score fr based',fit_lda.score(modelFRLDA[test_idx],trial_type[test]))



modelSTIMLDA = rsc_stim.reshape(rsc_stim.shape[0],-1)
mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
fit_stim = mdl.fit(modelSTIMLDA[train_idx],trial_type[train])
print('score stim based',fit_stim.score(modelSTIMLDA[test_idx],trial_type[test]))


modelLATENTLDA = rsc_resid.reshape(rsc_resid.shape[0],-1)
mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
fit_resid = mdl.fit(modelLATENTLDA[train_idx],trial_type[train])
print('score latent based',fit_resid.score(modelLATENTLDA[test_idx],trial_type[test]))

plt.figure(figsize=(8,8))

plt.subplot(3,2,1)
plt.title('LDA TRAIN')
dict_lab = {0:'close',1:'average',2:'far'}

dict_color = {0:'b',1:'r',2:'g'}

plt.ylabel('FIRING BASED')
for k in [0,1,2]:
    sel = train_idx[trial_type[train] == k]
    proj = fit_lda.transform(modelFRLDA[sel])
    plt.scatter(proj[:,0],proj[:,1],label=dict_lab[k],color=dict_color[k])
plt.legend()

plt.subplot(3,2,2)
plt.title('LDA TEST\n acc: %.2f'%fit_lda.score(modelFRLDA[test_idx],trial_type[test]))

for k in [0,1,2]:
    sel = test_idx[trial_type[test] == k]
    proj = fit_lda.transform(modelFRLDA[sel])
    plt.scatter(proj[:,0],proj[:,1],label=dict_lab[k],color=dict_color[k])
    
    
plt.subplot(3,2,3)
plt.ylabel('STIM. BASED $B\cdot s$')
for k in [0,1,2]:
    sel = train_idx[trial_type[train] == k]
    proj = fit_stim.transform(modelSTIMLDA[sel])
    plt.scatter(proj[:,0],proj[:,1],label=dict_lab[k],color=dict_color[k])
plt.legend()

plt.subplot(3,2,4)
plt.title('acc: %.2f'%fit_stim.score(modelSTIMLDA[test_idx],trial_type[test]))

for k in [0,1,2]:
    sel = test_idx[trial_type[test] == k]
    proj = fit_stim.transform(modelSTIMLDA[sel])
    plt.scatter(proj[:,0],proj[:,1],label=dict_lab[k],color=dict_color[k])

plt.subplot(3,2,5)
plt.ylabel('LATENT BASED $C\cdot E[x|y]$')
for k in [0,1,2]:
    sel = train_idx[trial_type[train] == k]
    proj = fit_resid.transform(modelLATENTLDA[sel])
    plt.scatter(proj[:,0],proj[:,1],label=dict_lab[k],color=dict_color[k])
plt.legend()

plt.subplot(3,2,6)

plt.title('acc: %.2f'%fit_resid.score(modelLATENTLDA[test_idx],trial_type[test]))

for k in [0,1,2]:
    sel = test_idx[trial_type[test] == k]
    proj = fit_resid.transform(modelLATENTLDA[sel])
    plt.scatter(proj[:,0],proj[:,1],label=dict_lab[k],color=dict_color[k])

plt.tight_layout()
    
plt.savefig('radDist_LDA_trialGrouped.png')
# # align to a specific event
# event = 't_stop'
# dt = 6 #ms
# interval = 500 #ms pre and post
# unq_tr = np.unique(trial_idx)

# int_pre = int(np.ceil(interval/dt))
# time_bins = 1+2*int_pre
# aligned_pred = np.zeros((unq_tr.shape[0], time_bins, 2))*np.nan
# stim_resp = np.zeros((unq_tr.shape[0], time_bins, 81))*np.nan
# cc = 0
# for tr in unq_tr:
#     # print(tr)
#     tstop_tr = Xt[trial_idx==tr,var_names=='t_stop']
#     idx = np.where(tstop_tr)[0][0]
    
#     pre = np.max([idx - int_pre, 0])
#     post = np.min([ idx+int_pre, tstop_tr.shape[0]-1])
    
#     if pre == 0:
#         idx0 = int_pre - idx
#     else:
#         idx0 = 0
#     if post ==  tstop_tr.shape[0]-1:
#         idx1 = tstop_tr.shape[0] - pre - 1
#     else:
#         idx1 = 2* int_pre
 
#     aligned_pred[cc,idx0:idx1+1,0] = pca_sm[trial_idx==tr][np.arange(pre,post+1),0]
#     aligned_pred[cc,idx0:idx1+1,1] = pca_sm[trial_idx==tr][np.arange(pre,post+1),1]
#     stim_resp[cc, idx0:idx1+1, :] = Bs[trial_idx==tr][np.arange(pre,post+1),:]

    
#     cc += 1
    
# # align and rescale
# rescaled_pca = np.zeros((unq_tr.shape[0], 70, pca_sm.shape[1]))
# cc = 0
# for tr in unq_tr:
#     # print(tr)
#     idx1 = np.where(Xt[trial_idx==tr,var_names=='t_stop'])[0][0]
#     idx0 = np.where(Xt[trial_idx==tr,var_names=='t_move'])[0][0]
#     ts = np.arange(idx1-idx0)*0.006
#     ts_resc = np.linspace(ts[0],ts[-1],70)
    
#     for k in range(pca_sm.shape[1]):
#         interp = interp1d(ts,pca_sm[trial_idx==tr][idx0:idx1,k])

#         rescaled_pca[cc,:,k] = interp(ts_resc)
    
    
    
#     cc += 1

# ## LDA traj

# # extract the traj

#



# selected = np.where(selLeft | selRight | selCenter)[0]
# test_idx = selected[::10]
# train_idx = np.array(list(set(selected).difference(set(test_idx))),dtype=int)
# modelXLDA = np.reshape(rescaled_pca,(rescaled_pca.shape[0],
#                                      rescaled_pca.shape[1]*rescaled_pca.shape[2]))


# trial_type = np.zeros(selected.shape[0])
# cc=0
# train_bool =  np.zeros(selected.shape[0],dtype=bool)
# test_bool =  np.zeros(selected.shape[0],dtype=bool)

# for tr in selected:
#     if cc % 10 == 0:
#         test_bool[cc] = True
#     else:
#         train_bool[cc] = True
        
#     if selLeft[tr]:
#         trial_type[cc] = 0
#     elif selCenter[tr]:
#         trial_type[cc] = 1
#     elif selRight[tr]:
#         trial_type[cc] = 2
#     else:
#         raise ValueError
#     cc+=1

# # align and rescale
# rescaled_stim = np.zeros((unq_tr.shape[0], 70, stim_resp.shape[1]))
# cc = 0
# for tr in unq_tr:
#     # print(tr)
#     idx1 = np.where(Xt[trial_idx==tr,var_names=='t_stop'])[0][0]
#     idx0 = np.where(Xt[trial_idx==tr,var_names=='t_move'])[0][0]
#     ts = np.arange(idx1-idx0)*0.006
#     ts_resc = np.linspace(ts[0],ts[-1],70)
    
#     for k in range(pca_sm.shape[1]):
#         interp = interp1d(ts, BS[trial_idx==tr][idx0:idx1,k])

#         rescaled_stim[cc,:,k] = interp(ts_resc)
    
    
    
#     cc += 1
    
    
# # Pca
# # 
# mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
# modelXLDA = modelXLDA[selected]
# mdl.fit(modelXLDA[train_bool],trial_type[train_bool])


# plt.scatter()
# print(mdl.score(modelXLDA[test_bool],trial_type[test_bool]))
    
