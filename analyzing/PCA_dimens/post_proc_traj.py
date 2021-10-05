#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:05:20 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
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
    if ev0 == 't_flyON':
        ev0 = 't_flyOFF'
        subtract = 50
    else:
        subtract = 0
    unq_tr = np.unique(trial_idx)

    rescaled_xvar = np.zeros((unq_tr.shape[0], tp, xvar.shape[1])) * np.nan

    cc = 0
    nannonidx = []
    for tr in unq_tr:
        print('tr: %d'%tr)
        try:
            
            idx0 = np.where(Xt[trial_idx==tr,var_names==ev0])[0][0] - subtract
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


def interpolate2(xvar, tp, ev0, ev1, trial_idx, Xt, var_names):
    if ev0 == 't_flyON':
        ev0 = 't_flyOFF'
        subtract = 50
    else:
        raise ValueError('this works only for targ onset')
    unq_tr = np.unique(trial_idx)

    rescaled_xvar = np.zeros((unq_tr.shape[0], tp, xvar.shape[1])) * np.nan

    cc = 0
    nannonidx = []
    for tr in unq_tr:

        print('tr: %d' % tr)
        # if tr == 633:
        #     xx=0
        # else:
        #     continue
        try:

            idx0 = np.where(Xt[trial_idx == tr, var_names == ev0])[0][0] - subtract
            idx1 = np.where(Xt[trial_idx == tr, var_names == ev1])[0][0]
            idx2 = np.where(Xt[trial_idx == tr, var_names == ev1])[0][0]
        except:
            print('skip tr: %d' % tr)

            cc += 1
            continue
        nannonidx += [cc]
        ts = np.arange(idx2 - idx0) * 0.006
        ts_off = subtract * 0.006
        if ts[-1] < ts_off:
            print('skip tr: %d' % tr)

            cc += 1
            continue
        ts_resc = np.hstack((np.linspace(ts[0], ts_off, 20)[:-1], np.linspace(ts_off, ts[-1], tp-20+1)))
        for k in range(xvar.shape[1]):
            interp = interp1d(ts, xvar[trial_idx == tr][idx0:idx2, k])

            rescaled_xvar[cc, :, k] = interp(ts_resc)
        # print(tr,np.sum(np.isnan(interp(ts_resc))) )
        cc += 1
    # rescaled_xvar = rescaled_xvar[np.array(nannonidx,dtype=int)]
    return rescaled_xvar

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
    
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

rsc_firing = interpolate2(dat['fr'],70, 't_flyON', 't_stop', trial_idx, Xt, var_names)


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
cLeft = np.array([-130,200])
cCenter = np.array([0,200])
cRight = np.array([130,200])

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

# stack rates and idx_create
idx_tr_rate = []
tp_rate = 0
for k in rate.keys():
    tp_rate += rate[k].shape[1]
    idx_tr_rate += [k]*rate[k].shape[1]
idx_tr_rate = np.array(idx_tr_rate)
rate_mat = np.zeros((rate[0].shape[0],tp_rate))
spk_tot_mat = np.zeros((rate[0].shape[0],len(rate.keys())))
traj_mat = np.zeros((tp_rate,2))
endPt_mat = np.zeros((2,len(rate.keys())))
cc = 0
numit = 0
for k in rate.keys():
    rate_mat[:,cc:cc+rate[k].shape[1]] = rate[k]
    traj_mat[cc:cc+rate[k].shape[1], 0] = sm_traj[k,0]
    traj_mat[cc:cc+rate[k].shape[1], 1] = sm_traj[k,1]
    spk_tot_mat[:,numit] = rate[k].sum(axis=1)
    nnan = ~np.isnan(sm_traj[k,0])
    endPt_mat[0,numit] = sm_traj[k,0][nnan][-1]
    endPt_mat[1,numit] = sm_traj[k,1][nnan][-1]

    
    cc+=rate[k].shape[1]
    numit+=1

# sm_rate_mat = pop_spike_convolve(rate_mat.T,filter=h,trials_idx=idx_tr_rate)

# mdl_cca_fulTraj = CCA(2)
# non_nan = ~np.isnan(traj_mat.sum(axis=1))
# fit_fullTraj = mdl_cca_fulTraj.fit(sm_rate_mat[non_nan],traj_mat[non_nan])
# plt.figure()
# plt.suptitle('CCA latent projection',fontsize=15)
# plt.title('CCA projected time-warped rates')
# for k in range(10):
#     XX = np.dot(rsc_firing[np.where(selLeft)[0][k]], fit_fullTraj.x_weights_)
#     if k == 0:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b',label='left')
#     else:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b')

# for k in range(10):
#     XX = np.dot(rsc_firing[np.where(selCenter)[0][k]], fit_fullTraj.x_weights_)
#     if k == 0:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r',label='center')
#     else:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r')
    
# for k in range(10):
#     XX = np.dot(rsc_firing[np.where(selRight)[0][k]], fit_fullTraj.x_weights_)  
#     if k == 0:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g',label='right')
#     else:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g')

# plt.legend()

#
# mdl_cca_Cnts = CCA(2)
# fit_Cnts = mdl_cca_Cnts.fit(spk_tot_mat.T, endPt_mat.T)
# plt.figure()
# plt.suptitle('CCA latent projection',fontsize=15)
# plt.title('CCA projected time-warped rates')
# for k in range(10):
#     XX = np.dot(rsc_firing[np.where(selLeft)[0][k]], fit_Cnts.x_weights_)
#     if k == 0:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b',label='left')
#     else:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b')
#
# for k in range(10):
#     XX = np.dot(rsc_firing[np.where(selCenter)[0][k]], fit_Cnts.x_weights_)
#     if k == 0:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r',label='center')
#     else:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r')
#
# for k in range(10):
#     XX = np.dot(rsc_firing[np.where(selRight)[0][k]], fit_Cnts.x_weights_)
#     if k == 0:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g',label='right')
#     else:
#         plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g')
#
# plt.legend()


spikes, var_dict, trial_idx_concat = exp_data.concatenate_inputs('t_move','t_stop','t_reward',t_start=t_start,t_stop=t_stop)


brain_area = exp_data.spikes.brain_area





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

rsc_ang_path = interpolate2(Xt[:, var_names=='ang_path'],
                           70, 't_flyON', 't_stop', trial_idx_fit, Xt, var_names)
modelFRLDA = rsc_firing.reshape(rsc_firing.shape[0],-1)
mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
fit_lda = mdl.fit(modelFRLDA[train_idx],trial_type[train])
print('score fr based',fit_lda.score(modelFRLDA[test_idx],trial_type[test]))


# rescale stim and latent
rsc_resid = interpolate2(Cmu, 70, 't_flyON', 't_stop', trial_idx_fit, Xt, var_names)
rsc_stim = interpolate2(Bs, 70, 't_flyON', 't_stop', trial_idx_fit, Xt, var_names)

## divide by brain region
for ba in ['MST','PPC','PFC']:
    sel_ba = brain_area==ba
    rsc_firing_ba = rsc_firing[:,:,sel_ba]
    mdl = CCA(2)
    ccaRegrY = rsc_firing_ba[selected].sum(axis=1).reshape(selected.shape[0],-1)
    ccaRegrX = end_pt[selected]
    cca_fit = mdl.fit(ccaRegrX,ccaRegrY)

    # M = np.hstack((cca_fit.y_weights_,np.eye(81)))[:,:-1]
    # OB = gs(M,row_vecs=False)
    # mdl = CCA(1)
    # ccaRegrY2 = np.dot(np.dot(ccaRegrY,OB[:,1:]),OB[:,1:].T)
    # cca_fit2 = mdl.fit(ccaRegrX,ccaRegrY2)
    stack_left = np.zeros((0,2))
    for k in range(selLeft.sum()):

        XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
        stack_left = np.vstack((stack_left,XX))
        # plt.plot(XX[:,0],XX[:,1],color='b')

    stack_right = np.zeros((0,2))
    for k in range(selRight.sum()):
        XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
        stack_right = np.vstack((stack_right,XX))



    plt.figure()
    plt.suptitle('CCA latent projection',fontsize=15)
    plt.title('CCA projected time-warped rates')
    for k in range(10):
        XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
        if k == 0:
            plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b',label='left')
        else:
            plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b')

    for k in range(10):
        XX = np.dot(rsc_firing_ba[np.where(selCenter)[0][k]], cca_fit.y_weights_)
        if k == 0:
            plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r',label='center')
        else:
            plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r')

    for k in range(10):
        XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
        if k == 0:
            plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g',label='right')
        else:
            plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g')

    plt.legend()
    plt.savefig('%s 1D projection.png'%ba)


    plt.figure()
    plt.suptitle('CCA latent projection',fontsize=15)
    plt.title('CCA projected time-warped rates')
    for k in range(selLeft.sum()):
        XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
        if k == 0:
            plt.plot(XX[:,0],XX[:,1],color='b',label='left')
        else:
            plt.plot(XX[:,0],XX[:,1],color='b')

    for k in range(selCenter.sum()):
        XX = np.dot(rsc_firing_ba[np.where(selCenter)[0][k]], cca_fit.y_weights_)
        if k == 0:
            plt.plot(XX[:,0],XX[:,1],color='r',label='center')
        else:
            plt.plot(XX[:,0],XX[:,1],color='r')

    for k in range(selRight.sum()):
        XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
        if k == 0:
            plt.plot(XX[:,0],XX[:,1],color='g',label='right')
        else:
            plt.plot(XX[:,0],XX[:,1],color='g')

    plt.legend()

    plt.savefig('%s 2D projection.png'%ba)

rsc_firing_ba = rsc_firing
mdl = CCA(2)
ccaRegrY = rsc_firing_ba[selected].sum(axis=1).reshape(selected.shape[0],-1)
ccaRegrX = end_pt[selected]
cca_fit = mdl.fit(ccaRegrX,ccaRegrY)

# M = np.hstack((cca_fit.y_weights_,np.eye(81)))[:,:-1]
# OB = gs(M,row_vecs=False)
# mdl = CCA(1)
# ccaRegrY2 = np.dot(np.dot(ccaRegrY,OB[:,1:]),OB[:,1:].T)
# cca_fit2 = mdl.fit(ccaRegrX,ccaRegrY2)
stack_left = np.zeros((0,2))
for k in range(selLeft.sum()):

    XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
    stack_left = np.vstack((stack_left,XX))
    # plt.plot(XX[:,0],XX[:,1],color='b')

stack_right = np.zeros((0,2))
for k in range(selRight.sum()):
    XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
    stack_right = np.vstack((stack_right,XX))



plt.figure()
plt.suptitle('CCA latent projection',fontsize=15)
plt.title('CCA projected time-warped rates')
for k in range(10):
    XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
    if k == 0:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b',label='left')
    else:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b')

for k in range(10):
    XX = np.dot(rsc_firing_ba[np.where(selCenter)[0][k]], cca_fit.y_weights_)
    if k == 0:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r',label='center')
    else:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r')

for k in range(10):
    XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
    if k == 0:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g',label='right')
    else:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g')

plt.legend()
plt.savefig('1D projection.png')


plt.figure()
plt.suptitle('CCA latent projection',fontsize=15)
plt.title('CCA projected time-warped rates')
for k in range(selLeft.sum()):
    XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
    if k == 0:
        plt.plot(XX[:,0],XX[:,1],color='b',label='left')
    else:
        plt.plot(XX[:,0],XX[:,1],color='b')

for k in range(selCenter.sum()):
    XX = np.dot(rsc_firing_ba[np.where(selCenter)[0][k]], cca_fit.y_weights_)
    if k == 0:
        plt.plot(XX[:,0],XX[:,1],color='r',label='center')
    else:
        plt.plot(XX[:,0],XX[:,1],color='r')

for k in range(selRight.sum()):
    XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
    if k == 0:
        plt.plot(XX[:,0],XX[:,1],color='g',label='right')
    else:
        plt.plot(XX[:,0],XX[:,1],color='g')

plt.legend()

plt.savefig('2D projection.png')



## divide by brain region
for ba in ['MST','PPC','PFC']:
    sel_ba = brain_area==ba
    rsc_firing_ba = rsc_firing[:,:,sel_ba]
    mdl = CCA(2)
    ccaRegrY = rsc_firing_ba[selected].sum(axis=1).reshape(selected.shape[0],-1)
    ccaRegrX = end_pt[selected]
    cca_fit = mdl.fit(ccaRegrX,ccaRegrY)

    # M = np.hstack((cca_fit.y_weights_,np.eye(81)))[:,:-1]
    # OB = gs(M,row_vecs=False)
    # mdl = CCA(1)
    # ccaRegrY2 = np.dot(np.dot(ccaRegrY,OB[:,1:]),OB[:,1:].T)
    # cca_fit2 = mdl.fit(ccaRegrX,ccaRegrY2)
    stack_left = np.zeros((0,2))
    for k in range(selLeft.sum()):

        XX = np.dot(rsc_firing_ba[np.where(selLeft)[0][k]], cca_fit.y_weights_)
        stack_left = np.vstack((stack_left,XX))
        # plt.plot(XX[:,0],XX[:,1],color='b')

    stack_right = np.zeros((0,2))
    for k in range(selRight.sum()):
        XX = np.dot(rsc_firing_ba[np.where(selRight)[0][k]], cca_fit.y_weights_)
        stack_right = np.vstack((stack_right,XX))



    plt.figure()
    plt.suptitle('CCA latent projection - %s'%ba,fontsize=15)
    plt.title('CCA projected time-warped rates')

    XX = np.dot(rsc_firing_ba[np.where(selLeft)[0]].mean(axis=0), cca_fit.y_weights_)
    SS = np.dot(rsc_firing_ba[np.where(selLeft)[0]], cca_fit.y_weights_).std(axis=0)
    k=0
    if k == 0:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b',label='left')
    else:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='b')
    plt.fill_between(np.linspace(0,1,XX.shape[0]),XX[:,0]-SS[:,0],XX[:,0]+SS[:,0],color='b',alpha=0.4)


    XX = np.dot(rsc_firing_ba[np.where(selCenter)[0]].mean(axis=0), cca_fit.y_weights_)
    SS = np.dot(rsc_firing_ba[np.where(selCenter)[0]], cca_fit.y_weights_).std(axis=0)

    if k == 0:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r',label='center')
    else:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='r')
    plt.fill_between(np.linspace(0,1,XX.shape[0]),XX[:,0]-SS[:,0],XX[:,0]+SS[:,0],color='r',alpha=0.4)



    XX = np.dot(rsc_firing_ba[np.where(selRight)[0]].mean(axis=0), cca_fit.y_weights_)
    SS = np.dot(rsc_firing_ba[np.where(selRight)[0]], cca_fit.y_weights_).std(axis=0)

    if k == 0:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g',label='right')
    else:
        plt.plot(np.linspace(0,1,XX.shape[0]),XX[:,0],color='g')
    plt.fill_between(np.linspace(0,1,XX.shape[0]),XX[:,0]-SS[:,0],XX[:,0]+SS[:,0],color='g',alpha=0.4)
    plt.legend()
    plt.savefig('mean %s 1D projection.png'%ba)


    np.savez('res_CCA_%s.npz'%ba, cca_fit = cca_fit,ccaRegr_spk=ccaRegrY,ccaRegr_targLoc=ccaRegrX,
             rsc_firing=rsc_firing_ba)




# for k in range(10):
#     trY = np.dot(np.dot(rsc_firing[np.where(selRight)[0][k]],OB[:,1:]),OB[:,1:].T)
#     yy = np.dot(trY, cca_fit2.y_weights_).flatten()
#     xx = np.dot(rsc_firing[np.where(selRight)[0][k]], cca_fit.y_weights_).flatten()
    # plt.plot(xx,yy,color='g')

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
    
