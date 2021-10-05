import numpy as np
import sys,os,dill

main_dir = '/Users/edoardo/Work/Code/GAM_code/'
# sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
# sys.path.append('/scratch/eb162/GAM_library/')
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(main_dir,'GAM_library'))
from GAM_library import *
from data_handler import *
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV

base_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'
print('loading session %s...' % session)
pre_trial_dur = 0.0
post_trial_dur = 0.0


# keys in the mat file generated by the preprocessing script of  K.
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'

dat = loadmat(os.path.join(base_file, '%s.mat' % (session)))


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

var_names = ('rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target',
             'eye_vert', 'eye_hori',
             'rad_acc', 'ang_acc')

time_pts, rate, sm_traj, raw_traj, fly_pos, cov_dict = exp_data.GPFA_YU_preprocessing_noTW(t_start, t_stop,
                                                                                           var_list=var_names,binwidth_ms=None)

spikes, var_dict, trial_idx = exp_data.concatenate_inputs('t_move','t_stop','t_reward',t_start=t_start,t_stop=t_stop)


brain_area = exp_data.spikes.brain_area


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

plt.figure()
for kk in np.where(selLeft)[0]:
    plt.plot(sm_traj[kk,0],sm_traj[kk,1],'b')

for kk in np.where(selCenter)[0]:
    plt.plot(sm_traj[kk,0],sm_traj[kk,1],'r')

for kk in np.where(selRight)[0]:
    plt.plot(sm_traj[kk,0],sm_traj[kk,1],'g')

plt.savefig('separate_traj.png')



rateLeft = np.zeros((0,81))
for key in np.where(selLeft)[0]:
    rateLeft = np.vstack((rateLeft,rate[key].T))

rateCenter = np.zeros((0,81))
for key in np.where(selRight)[0]:
    rateCenter = np.vstack((rateCenter,rate[key].T))

rateRight = np.zeros((0,81))
for key in np.where(selCenter)[0]:
    rateRight = np.vstack((rateRight,rate[key].T))


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
dt = 0.006

firing_rate_est = pop_spike_convolve(spikes.T, trial_idx, h)/dt



kk = 1
# divide by periods and compute pca:
    
ev_list = [('t_move', 't_stop'),('t_stop','t_reward')]
fig1 = plt.figure(figsize=[9.55, 4.8])
dict_pcs = {}


for ev0,ev1 in  ev_list:
    plt.subplot(2,2,kk)
    rate_ev_left = np.zeros((0, firing_rate_est.shape[1]))
    if ev0 == 't_move':
        tr_list = []
        cnttr = 0
    ii=0
    for tr in np.unique(trial_idx)[selLeft][:]:
        bl_tr = trial_idx == tr
        idx0 = np.where(var_dict[ev0][bl_tr] == 1)[0]
        if idx0.shape[0] == 0:
            idx0 = [0]
            
        idx1 = np.where(var_dict[ev1][bl_tr] == 1)[0]
        if len(idx0) != 1 or len(idx1) != 1:
            continue
        idx0 = idx0[0]
        idx1 = idx1[0]
        rate_tr = firing_rate_est[bl_tr][idx0:idx1]
        rate_ev_left = np.vstack((rate_ev_left, rate_tr))
        if ev0 == 't_move':
            tr_list += [tr]*rate_tr.shape[0]
            # print(len(tr_list),rate_ev_left.shape[0])
            cnttr+=1
        if ii >= 40:
            break
        ii += 1
    
    model = PCA()
    pca_left = model.fit(rate_ev_left)

    rate_ev_center = np.zeros((0, firing_rate_est.shape[1]))
    ii = 0
    for tr in np.unique(trial_idx)[selCenter][:]:
        bl_tr = trial_idx == tr
        idx0 = np.where(var_dict[ev0][bl_tr] == 1)[0]
        if idx0.shape[0] == 0:
            idx0 = [0]
        idx1 = np.where(var_dict[ev1][bl_tr] == 1)[0]
        if len(idx0) != 1 or len(idx1) != 1:
            continue
        
        idx0 = idx0[0]
        idx1 = idx1[0]
        rate_tr = firing_rate_est[bl_tr][idx0:idx1]
        rate_ev_center = np.vstack((rate_ev_center, rate_tr))
        if ev0 == 't_move':
            tr_list += [tr]*rate_tr.shape[0]
            cnttr+=1
            print(tr,idx0,idx1)
        if ii >= 40:
            break
        ii += 1

    model = PCA()
    pca_center = model.fit(rate_ev_center)
    rate_ev_right = np.zeros((0, firing_rate_est.shape[1]))
    ii=0
    for tr in np.unique(trial_idx)[selRight][:]:
        bl_tr = trial_idx == tr
        idx0 = np.where(var_dict[ev0][bl_tr] == 1)[0]
        if idx0.shape[0] == 0:
            idx0 = [0]
        idx1 = np.where(var_dict[ev1][bl_tr] == 1)[0]
        if len(idx0) != 1 or len(idx1) != 1:
            continue
        # print(idx0,idx1)
        idx0 = idx0[0]
        idx1 = idx1[0]
        rate_tr = firing_rate_est[bl_tr][idx0:idx1]
        rate_ev_right = np.vstack((rate_ev_right, rate_tr))
        if ev0 == 't_move':
            tr_list += [tr] * rate_tr.shape[0]
            cnttr+=1
            
        if ii >= 40:
            break
        ii += 1

    if ev0 == 't_move':
        rate_all = np.vstack((rate_ev_left, rate_ev_center, rate_ev_right))

    model = PCA()
    pca_right = model.fit(rate_ev_right)


    plt.title('%s-%s'%(ev0,ev1))
    plt.ylabel('weights')
    plt.xlabel('neurons')
    plt.plot(pca_left.components_[0,:],'b')
    plt.plot(pca_center.components_[0,:],'r')
    plt.plot(pca_right.components_[0,:],'g')

    plt.subplot(2, 2, kk+2)
    plt.plot(np.arange(1,82),np.cumsum(pca_left.explained_variance_ratio_),'-b',label='left')
    plt.plot(np.arange(1,82),np.cumsum(pca_center.explained_variance_ratio_),'-r',label='center')
    plt.plot(np.arange(1,82),np.cumsum(pca_right.explained_variance_ratio_),'-g',label='right')


    plt.legend()

    kk+=1

fig1.get_size_inches()
plt.tight_layout()
plt.savefig('pca_per_trajectory.png')


# fix event rate
tr_list = np.array(tr_list)

cnts = np.zeros((np.unique(tr_list).shape[0],rate_all.shape[1]))
cc = 0
yy = np.zeros(np.unique(tr_list).shape[0])
for tr in np.unique(tr_list):
    bl = tr_list == tr
    cnts[cc, :] = rate_all[bl].mean(axis=0)
    if tr in np.unique(trial_idx)[selCenter][:]:
        yy[cc] = 1
    elif tr in np.unique(trial_idx)[selRight][:]:
        yy[cc] = 2


    cc += 1


mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
mdl.fit(cnts,yy)
cnts_proj = mdl.transform(cnts)

plt.figure(figsize=[12,5])
plt.subplot(131)
plt.title('trajectories')
for kk in np.where(selLeft)[0]:
    plt.plot(sm_traj[kk,0],sm_traj[kk,1],'b')

for kk in np.where(selCenter)[0]:
    plt.plot(sm_traj[kk,0],sm_traj[kk,1],'r')

for kk in np.where(selRight)[0]:
    plt.plot(sm_traj[kk,0],sm_traj[kk,1],'g')

plt.subplot(132)
plt.title('LDA')
plt.scatter(cnts_proj[yy==0,0],cnts_proj[yy==0,1],color='b')
plt.scatter(cnts_proj[yy==1,0],cnts_proj[yy==1,1],color='r')
plt.scatter(cnts_proj[yy==2,0],cnts_proj[yy==2,1],color='g')

plt.subplot(133)
correct = []
plt.title('LDA TEST SET')
cnts_test = np.zeros((0,rate_all.shape[1]))
ev0 = 't_move'
ev1 = 't_stop'
for tr in np.unique(trial_idx)[selLeft]:
    bl = trial_idx == tr
    idx0 = np.where(var_dict[ev0][bl] == 1)[0]
    if idx0.shape[0] == 0:
        idx0 = [0]
    idx1 = np.where(var_dict[ev1][bl] == 1)[0]
    if len(idx0) != 1 or len(idx1) != 1:
        continue
    idx0 = idx0[0]
    idx1 = idx1[0]
    # print('here',tr ,idx0,idx1)
    rate_tr = firing_rate_est[bl][idx0:idx1]
    if not tr in tr_list:
        cnts_test = np.vstack((cnts_test,rate_tr.mean(axis=0)))

# yy = np.zeros(rate_ev_right.shape[0]+rate_ev_center.shape[0]+rate_ev_left.shape[0])
# yy[rate_ev_right.shape[0]:rate_ev_right.shape[0]+rate_ev_center.shape[0]] = 1
# yy[rate_ev_right.shape[0]+rate_ev_center.shape[0]:] = 2
correct = np.hstack((correct,mdl.predict(cnts_test) == 0))
print('correct left',(mdl.predict(cnts_test) == 0).mean())


all_sel_trials = np.hstack((np.where(selLeft)[0],np.where(selCenter)[0],np.where(selRight)[0]))
unq_trials = np.unique(trial_idx)

# ## LDA FULL TRAJ
nunits = firing_rate_est.shape[1]
modelX = np.zeros((all_sel_trials.shape[0],60*nunits))
xx = np.linspace(0,1,60)
rate_zero = np.zeros((nunits,60))

cc = 0
y_proj = np.zeros((all_sel_trials.shape[0]))
for kk in all_sel_trials:

    if kk in np.where(selLeft)[0]:
        y_proj[cc] = 0
    elif kk in np.where(selRight)[0]:
        y_proj[cc] = 2
    if kk in np.where(selCenter)[0]:
        y_proj[cc] = 1
    tr = unq_trials[kk]
    bl = trial_idx == tr

    idx0 = np.where(var_dict[ev0][bl] == 1)[0]
    if idx0.shape[0] == 0:
        idx0 = [0]
    idx1 = np.where(var_dict[ev1][bl] == 1)[0]
    if len(idx0) != 1 or len(idx1) != 1:
        continue
    idx0 = idx0[0]
    idx1 = idx1[0]
    # print('here',tr ,idx0,idx1)
    rate_tr = firing_rate_est[bl][idx0:idx1]

    ts = np.linspace(0,1,rate_tr.shape[0])
    for un in range(rate_tr.shape[1]):
        interp = interp1d(ts, rate_tr[:, un])
        rate_zero[un, :] = interp(xx)
        # print(max(rate_zero[un, :]))

    modelX[cc,:] = rate_zero.flatten()
    cc += 1
ind_test = np.arange(y_proj.shape[0])[::8]
ind_train = np.array(list(set(np.arange(y_proj.shape[0])).difference(set(ind_test))))

# plt.plot(interp(xx))
# interp = interp1d(ts, rate_tr[:, 12])
# plt.plot(interp(xx))
#


mdl = LinearDiscriminantAnalysis(shrinkage='auto',solver='eigen')
mdl.fit(modelX[ind_train],y_proj[ind_train])
cnts_proj = mdl.transform(modelX[ind_train, :])
pred = mdl.predict(modelX[ind_test])

accur = (pred == y_proj[ind_test]).mean()

y_test = y_proj[ind_test]
X_proj_test = mdl.transform(modelX[ind_test, :])
X_proj_train = mdl.transform(modelX[ind_train, :])
y_train = y_proj[ind_train]

plt.figure(figsize=(9,3.6))

plt.subplot(131)
plt.title('LDA TRAIN')
cols = {1:'r',2:'g',0:'b'}
for kk in [0,1,2]:
    plt.scatter(X_proj_train[y_train==kk,0], X_proj_train[y_train==kk,1],color=cols[kk])


plt.subplot(132)
plt.title('LDA TEST - true label')
cols = {1:'r',2:'g',0:'b'}
for kk in [0,1,2]:
    plt.scatter(X_proj_test[y_test==kk,0], X_proj_test[y_test==kk,1],color=cols[kk])

plt.tight_layout()

plt.subplot(133)
plt.title('LDA TEST - perd label')
cols = {1:'r',2:'g',0:'b'}
for kk in [0,1,2]:
    plt.scatter(X_proj_test[pred==kk,0], X_proj_test[pred==kk,1],color=cols[kk])

plt.tight_layout()

plt.savefig('LDA_traj.png')


