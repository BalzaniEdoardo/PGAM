import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
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
import dtw


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




session = 'm53s113'
# smooth traj
filtwidth = 15
t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)


# extract variables
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz' % session

par_list = ['Xt', 'Yt', 'var_names', 'info_trial',
            'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
            'unit_type', 'channel_id', 'electrode_id', 'cluster_id']

(Xt, yt, var_names, trial_type,
 trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
 cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
 channel_id, electrode_id, cluster_id) = unpack_preproc_data(fhName, par_list)


## extract trial sel

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

# %%


unq_trials = np.unique(trial_idx)





# plt.subplot(211)
# plt.plot(x_series[res.index1])
# plt.plot(y_series[res.index2])
#
# plt.subplot(212)
# plt.plot(x_series[res2.index1])
# plt.plot(z_series[res2.index2])

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/PCA_dimens/res_CCA_PPC.npz',allow_pickle=True)
cca_fit = dat['cca_fit'].all()

sm_spk = pop_spike_convolve(np.sqrt(yt),trial_idx,h)

proj = np.dot(sm_spk[:,brain_area=='PPC'],cca_fit.y_weights_)


tr_left = unq_trials[selLeft]
tr_right = unq_trials[selRight]

plt.figure(figsize=(8, 8))
kk=1
for k in range(3):
    trA=35
    trB=1+k
    rad_vel = Xt[:, var_names == 'ang_vel']

    x_left = rad_vel[trial_idx==tr_left[trA]]
    y_left = rad_vel[trial_idx==tr_left[trB]]

    dtw_res = dtw.dtw(x_left,y_left)


    plt.subplot(3,3,kk)
    plt.title('aligend rad vel')
    plt.plot(rad_vel[trial_idx==tr_left[trA]][dtw_res.index1s])
    plt.plot(rad_vel[trial_idx==tr_left[trB]][dtw_res.index2s])


    plt.subplot(3,3,kk+1)
    plt.title('raw')
    plt.plot(np.linspace(0,1,(trial_idx==tr_left[trA]).sum()),proj[trial_idx==tr_left[trA],0])
    plt.plot(np.linspace(0,1,(trial_idx==tr_left[trB]).sum()),proj[trial_idx==tr_left[trB],0])

    plt.subplot(3,3,kk+2)
    plt.title('dtw')
    plt.plot(proj[trial_idx==tr_left[trA],0][dtw_res.index1s])
    plt.plot(proj[trial_idx==tr_left[trB],0][dtw_res.index2s])
    kk+=3

plt.tight_layout()

