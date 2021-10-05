import numpy as np
import sys,os,dill
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library/')
sys.path.append('/scratch/eb162/GAM_library/')
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


filtwidth = 15
t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)

session = 'm53s113'
file_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'
dat = np.load(os.path.join(file_fld,session+'.npz'),allow_pickle=True)

concat = dat['data_concat'].all()
trial_idx = concat['trial_idx']


yt = concat['Yt']
X = concat['Xt']
var_names = dat['var_names']


dt = 0.006
firing_rate_est = pop_spike_convolve(yt, trial_idx, h)/dt

kk = 1
# divide by periods and compute pca:
ev_list = [('t_move', 't_stop'),('t_stop','t_reward')]
plt.figure()
dict_pcs = {}
for ev0,ev1 in  ev_list:
    rate_ev = np.zeros((0, firing_rate_est.shape[1]))
    for tr in np.unique(trial_idx):
        bl_tr = trial_idx == tr
        x_tr = X[bl_tr]
        idx0 = np.where(x_tr[:, var_names == ev0] == 1)[0]
        idx1 = np.where(x_tr[:, var_names == ev1] == 1)[0]
        if len(idx0) != 1 or len(idx1) != 1:
            continue
        idx0 = idx0[0]
        idx1 = idx1[0]
        rate_tr = firing_rate_est[bl_tr]
        rate_ev = np.vstack((rate_ev, rate_tr))

    model = PCA()
    pca_fit = model.fit(rate_ev)

    dict_pcs['%s-%s'%(ev0,ev1)] = deepcopy(pca_fit)
    # plt.subplot(1,2,kk)
    plt.plot(np.arange(1,82),np.cumsum(pca_fit.explained_variance_ratio_),'-o',label='%s-%s'%(ev0,ev1))

    kk += 1
plt.legend()
plt.plot([1,81],[0.9,0.9],'k')
plt.ylabel('explained variance')
plt.xlabel('PCs')

plt.savefig('pca_analysis_filterwidth%d.pdf'%filtwidth)


plt.figure()
plt.subplot(121)
plt.plot(dict_pcs['%s-%s'%('t_move','t_stop')].components_[0,:],label='%s-%s'%('t_move','t_stop'))
plt.plot(dict_pcs['%s-%s'%('t_stop','t_reward')].components_[0,:],label='%s-%s'%('t_stop','t_reward'))
plt.legend()
plt.xlabel('neurons')
plt.ylabel('weights')
plt.title('first pc')


plt.subplot(122)
plt.plot(dict_pcs['%s-%s'%('t_move','t_stop')].components_[1,:],label='%s-%s'%('t_move','t_stop'))
plt.plot(dict_pcs['%s-%s'%('t_stop','t_reward')].components_[1,:],label='%s-%s'%('t_stop','t_reward'))
plt.legend()
plt.xlabel('neurons')
plt.ylabel('weights')
plt.title('second pc')


# let me have some


