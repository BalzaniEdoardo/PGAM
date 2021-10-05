import sys
import numpy as np
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from GAM_library import *
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
from data_handler import *
from scipy.io import loadmat
import matplotlib.pylab as plt
import scipy.linalg as linalg
import scipy.stats as sts

from sklearn.linear_model import TweedieRegressor


activity_fit = loadmat('/Users/edoardo/Work/Code/gpfa_v0203/AnalyzeSession/m53s113_gpfa_xDim03.mat')
dat = loadmat('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/test_m53s113_gpfa.mat')

seqTrain = activity_fit['seqTrain'][0]
trial_id_train = np.zeros(len(seqTrain),dtype=int)
for j in range(len(seqTrain)):
    trial_id_train[j] = seqTrain[j]['trialId'][0][0]

sm_traj = dat['sm_trajectory']
raw_traj = dat['raw_trajectory']
activity_dict = dat['dat'][0]
spikes = activity_dict['spikes'][0]
trial_id_tw = activity_dict['trialId'][0].flatten()

# filter the trials
sm_traj = sm_traj[trial_id_tw,:,:]
raw_traj = raw_traj[trial_id_tw,:,:]

# get the trial info
behav_stat_key = 'behv_stats'
behav_dat_key = 'trials_behv'
dat_mat = loadmat('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/m53s113.mat')
info = load_trial_types(dat_mat[behav_stat_key].flatten(),dat_mat[behav_dat_key].flatten())

del dat_mat

# divide by density
hd_trials = np.array(list(set(np.where(info.trial_type['density'] == 0.005)[0]).intersection(set(trial_id_train))),dtype=int)
ld_trials = np.array(list(set(np.where(info.trial_type['density'] == 0.0001)[0]).intersection(set(trial_id_train))),dtype=int)


# extract latent
xDim = seqTrain[0]['xsm'].shape[0]
T = seqTrain[0]['xsm'].shape[1]
mean_latent_hd = np.zeros((len(hd_trials),xDim,T))
mean_latent_ld = np.zeros((len(ld_trials),xDim,T))
cc=0
for tr in hd_trials:
    mean_latent_hd[cc] = seqTrain[np.where(trial_id_train==tr)[0][0]]['xsm']
    cc+=1

cc=0
for tr in ld_trials:
    mean_latent_ld[cc] = seqTrain[np.where(trial_id_train==tr)[0][0]]['xsm']
    cc+=1



mean_HD = np.nanmean(mean_latent_hd,axis=0)
std_HD = np.nanstd(mean_latent_hd,axis=0)

mean_LD = np.nanmean(mean_latent_ld,axis=0)
std_LD = np.nanstd(mean_latent_ld,axis=0)

plt.figure(figsize=(10,5))
plt.subplot(131)
p, = plt.plot(mean_HD[0,:],label='HD')
plt.fill_between(np.arange(mean_HD.shape[1]),mean_HD[0,:]-1.96*std_HD[0,:]/np.sqrt(mean_HD.shape[0]),mean_HD[0,:]+1.96*std_HD[0,:]/np.sqrt(mean_HD.shape[0]),color=p.get_color(),alpha=0.5)

p, = plt.plot(mean_LD[0,:],label='LD')
plt.fill_between(np.arange(mean_LD.shape[1]),mean_LD[0,:]-1.96*std_LD[0,:]/np.sqrt(mean_LD.shape[0]),mean_LD[0,:]+1.96*std_LD[0,:]/np.sqrt(mean_LD.shape[0]),color=p.get_color(),alpha=0.5)

plt.subplot(132)
p, = plt.plot(mean_HD[1,:],label='HD')
plt.fill_between(np.arange(mean_HD.shape[1]),mean_HD[1,:]-1.96*std_HD[1,:]/np.sqrt(mean_HD.shape[0]),mean_HD[1,:]+1.96*std_HD[1,:]/np.sqrt(mean_HD.shape[0]),color=p.get_color(),alpha=0.5)

p, = plt.plot(mean_LD[1,:],label='LD')
plt.fill_between(np.arange(mean_LD.shape[1]),mean_LD[1,:]-1.96*std_LD[1,:]/np.sqrt(mean_LD.shape[0]),mean_LD[1,:]+1.96*std_LD[1,:]/np.sqrt(mean_LD.shape[0]),color=p.get_color(),alpha=0.5)


plt.subplot(133)
p, = plt.plot(mean_HD[2,:],label='HD')
plt.fill_between(np.arange(mean_HD.shape[1]),mean_HD[2,:]-1.96*std_HD[2,:]/np.sqrt(mean_HD.shape[0]),mean_HD[2,:]+1.96*std_HD[2,:]/np.sqrt(mean_HD.shape[0]),color=p.get_color(),alpha=0.5)

p, = plt.plot(mean_LD[2,:],label='LD')
plt.fill_between(np.arange(mean_LD.shape[1]),mean_LD[2,:]-1.96*std_LD[2,:]/np.sqrt(mean_LD.shape[0]),mean_LD[2,:]+1.96*std_LD[2,:]/np.sqrt(mean_LD.shape[0]),color=p.get_color(),alpha=0.5)

