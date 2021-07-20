import numpy as np
from factor_analysis import *
import sys
from sklearn.decomposition import PCA

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
from utils_loading import unpack_preproc_data
import matplotlib.pylab as plt
from copy import deepcopy
# model prediction
def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])*np.nan
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        tmp = np.convolve(x[sel], filter, mode='same')
        if tmp.shape[0] == sel.sum():
            sm_x[sel] = tmp
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk



fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s95.npz'


par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']

dat= np.load('test_m53s95.npz',allow_pickle=True)
Xt = dat['Xt']
yt = dat['yt']
lfp_beta = dat['lfp_beta']
lfp_alpha = dat['lfp_alpha']
lfp_theta = dat['lfp_theta']
var_names = dat['var_names']
trial_idx = dat['trial_idx']
cont_rate_filter = dat['cont_rate_filter']
presence_rate_filter = dat['presence_rate_filter']
isi_v_filter = dat['isi_v_filter']
unq_tr = np.unique(trial_idx)
brain_area = dat['brain_area']

trial_type = dat['trial_type']
# sele_tr = np.where(trial_type['all'])[0]
# keep = np.zeros(trial_idx.shape[0], dtype=bool)
# for tr in sele_tr:
#     keep[trial_idx==tr] = True



combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
brain_area = brain_area[combine_filter]

# unit number according to matlab indexing

## use 18ms bins
y_rebin = np.zeros((0,yt.shape[1]))
trial_idx_rebin = []
t_stop_rebin = []
t_start_rebin = []
t_flyOFF_rebin = []
cond_rebin = []
for tr in unq_tr:
    sel = trial_idx == tr
    ytr = yt[sel]

    Xtr = Xt[sel]
    num_bins = ytr.shape[0] // 3
    ytr = ytr[:num_bins*3,:]
    tmp = ytr[::3,:] + ytr[1::3,:] + ytr[2::3,:]

    cond_rebin = np.hstack((cond_rebin,
                            tmp.shape[0]*[trial_type[tr]['density']]))

    Xtr = Xtr[:num_bins*3,:]

    t_stop_tr = np.zeros(tmp.shape[0])
    ii_stop = np.where(Xtr[:,var_names=='t_stop'] == 1)[0]//3
    t_stop_tr[ii_stop] = 1
    t_stop_rebin = np.hstack((t_stop_rebin,t_stop_tr))

    t_start_tr = np.zeros(tmp.shape[0])
    ii_start = np.where(Xtr[:, var_names == 't_move'] == 1)[0] // 3
    t_start_tr[ii_start] = 1
    t_start_rebin = np.hstack((t_start_rebin, t_start_tr))

    t_flyOFF_tr = np.zeros(tmp.shape[0])
    ii_flyOFF = np.where(Xtr[:, var_names == 't_flyOFF'] == 1)[0] // 3
    t_flyOFF_tr[ii_flyOFF] = 1
    t_flyOFF_rebin = np.hstack((t_flyOFF_rebin, t_flyOFF_tr))

    y_rebin = np.vstack((y_rebin, tmp))
    trial_idx_rebin = np.hstack((trial_idx_rebin, [tr]*tmp.shape[0]))


#
#
# # create filter
# trial_idx = trial_idx_rebin
# yt = y_rebin
# filt_width_list = [3]
# dim_list = [6]
# pred_error = []
# result_dict = {}
# for cond in [0.005,0.0001]:
#
#     filtwidth = filt_width_list[0]
#     print('FW', filtwidth)
#     t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
#     h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
#     h = h - h[0]
#     h = h / np.sum(h)
#
#
#     sm_spike = pop_spike_convolve(np.sqrt(yt), trial_idx, h)
#
#     sm_spike_centered = sm_spike#-np.nanmean(sm_spike,axis=0)
#     non_nan = np.where(~np.isnan(sm_spike_centered))
#     # CC, diagR, mu, cov,ll_iter = EM_step(sm_spike_centered, D, epsi=10**-6, maxiter=5000)
#
#     # pred_mu, pred_sigma, predict_error = mean_yj_given_ymj(sm_spike_centered, CC, diagR)
#     sm_spk_non_nan = sm_spike_centered[non_nan].reshape(sm_spike_centered[non_nan].shape[0] // yt.shape[1], yt.shape[1])
#     yt_non_nan = yt[non_nan].reshape(sm_spike_centered[non_nan].shape[0] // yt.shape[1], yt.shape[1])
#     trial_idx_non_nan = trial_idx[non_nan[0][non_nan[1] == 0]]
#     idx_endTrain = int(0.9 * sm_spk_non_nan.shape[0])
#
#
#
#     for D in dim_list:
#
#
#
#
#         model = FactorAnalysis(n_components=D)
#
#         fit = model.fit(sm_spk_non_nan[:idx_endTrain])
#         M = yt.shape[1]
#         loglike = lambda R, C: sts.multivariate_normal.logpdf(sm_spk_non_nan[:idx_endTrain], mean=np.zeros(M), cov=(np.dot(C, C.T) + R)).mean()
#         # print('SKL', loglike(np.diag(fit.noise_variance_), fit.components_.T), 'EM', ll_iter[-1])
#         pred_mu_skl, pred_sigma_skl, predict_error_skl = mean_yj_given_ymj(sm_spk_non_nan[idx_endTrain:], fit.components_.T,
#                                                                            fit.noise_variance_,
#                                                         np.sqrt(yt_non_nan[idx_endTrain:]))
#         pred_error += [predict_error_skl]
#         result_dict[D] = {'pred_mu':deepcopy(pred_mu_skl),'sm_spke':sm_spk_non_nan[idx_endTrain:],'trial_idx_non_nan':trial_idx_non_nan[idx_endTrain:],
#                           'fit':deepcopy(fit)}
#
#
# plt.plot( dim_list, pred_error,'-ok')
#
# fit = result_dict[dim_list[np.argmin(pred_error)]]['fit']

# ## density plot
# train_trials = trial_idx[:idx_endTrain]
# hd_bool = np.zeros(train_trials.shape[0],dtype=bool)
# for tr in np.where(trial_type['density']==0.005)[0]:
#     hd_bool[train_trials == tr] = True
#
# latents = result_dict[6]['fit'].transform(sm_spk_non_nan[:idx_endTrain])
#
# cov_HD = np.cov(latents[hd_bool].T)
# cov_LD = np.cov(latents[~hd_bool].T)