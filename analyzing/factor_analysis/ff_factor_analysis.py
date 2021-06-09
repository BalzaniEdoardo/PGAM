import numpy as np
from factor_analysis import *
import sys

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


## use 18ms bins
y_rebin = np.zeros((0,yt.shape[1]))
trial_idx_rebin = []
for tr in unq_tr:
    sel = trial_idx == tr
    ytr = yt[sel]
    num_bins = ytr.shape[0] // 3
    ytr = ytr[:num_bins*3,:]
    tmp = ytr[::3,:] + ytr[1::3,:] + ytr[2::3,:]
    y_rebin = np.vstack((y_rebin, tmp))
    trial_idx_rebin = np.hstack((trial_idx_rebin, [tr]*tmp.shape[0]))

# create filter
trial_idx = trial_idx_rebin
yt = y_rebin
filt_width_list = [1,2,3,4,5,6,7]
pred_error = []
result_dict = {}
for filtwidth in filt_width_list:
    t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
    h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
    h = h - h[0]
    h = h / np.sum(h)


    sm_spike = pop_spike_convolve(np.sqrt(yt), trial_idx, h)

    sm_spike_centered = sm_spike#-np.nanmean(sm_spike,axis=0)
    non_nan = np.where(~np.isnan(sm_spike_centered))
    D = 10
    # CC, diagR, mu, cov,ll_iter = EM_step(sm_spike_centered, D, epsi=10**-6, maxiter=5000)

    # pred_mu, pred_sigma, predict_error = mean_yj_given_ymj(sm_spike_centered, CC, diagR)
    model = FactorAnalysis(n_components=D)

    fit = model.fit(sm_spike_centered[non_nan].reshape(sm_spike_centered[non_nan].shape[0]//yt.shape[1],yt.shape[1]))
    M = yt.shape[1]
    loglike = lambda R, C: sts.multivariate_normal.logpdf(sm_spike_centered[non_nan].reshape(sm_spike_centered[non_nan].shape[0]//yt.shape[1],yt.shape[1]), mean=np.zeros(M), cov=(np.dot(C, C.T) + R)).mean()
    # print('SKL', loglike(np.diag(fit.noise_variance_), fit.components_.T), 'EM', ll_iter[-1])
    pred_mu_skl, pred_sigma_skl, predict_error_skl = mean_yj_given_ymj(sm_spike_centered[non_nan].reshape(sm_spike_centered[non_nan].shape[0]//yt.shape[1],yt.shape[1]), fit.components_.T,
                                                                       fit.noise_variance_,
                                                    np.sqrt(yt)[non_nan].reshape(sm_spike_centered[non_nan].shape[0]//yt.shape[1],yt.shape[1]))
    pred_error += [predict_error_skl]
    result_dict[filtwidth] = {'fit':deepcopy(fit),'pred_mu':deepcopy(pred_mu_skl),'non_nan':deepcopy(non_nan)}

    print('FILT WIDTH %d'%filtwidth)
    for k in range(yt.shape[1]):
        cr, pp = sts.pearsonr(result_dict[filtwidth]['pred_mu'][:, k], sm_spike_centered[non_nan].reshape(sm_spike_centered[non_nan].shape[0]//yt.shape[1],yt.shape[1])[:, k])
        print(k, cr, pp < 0.05, yt.sum(axis=0)[k] / (yt.shape[0] * 0.006))
#
# plt.plot( pred_error,'-ok')

# plt.figure(figsize=(11,8))
# for k in range(25):
#     tr = 300 + k
#     plt.subplot(5,5,k+1)
#     mu_tr = mu[:,trial_idx==unq_tr[tr]]
#     X_tr = Xt[trial_idx==unq_tr[tr]]
#     plt.plot(mu_tr[0,:],mu_tr[1,:])
#     plt.scatter(mu_tr[0,np.where(X_tr[:, var_names=='t_stop'])[0]],mu_tr[1,np.where(X_tr[:, var_names=='t_stop'])[0]],color='g')
#     plt.scatter(mu_tr[0,np.where(X_tr[:, var_names=='t_move'])[0]],mu_tr[1,np.where(X_tr[:, var_names=='t_move'])[0]],color='r')
#     plt.scatter(mu_tr[0,np.where(X_tr[:, var_names=='t_flyOFF'])[0]-50],mu_tr[1,np.where(X_tr[:, var_names=='t_flyOFF'])[0]-50],color='k')
# plt.tight_layout()
#
#
# import statsmodels.api as sm
# non_nan = ~np.isnan(Xt[:,var_names=='rad_target'])
# model = sm.OLS(Xt[non_nan.flatten(),var_names=='rad_target'],sm.add_constant(mu[:,non_nan.flatten()].T))
# res = model.fit()
# prd = res.predict(sm.add_constant(mu.T))
#
# unq_tr = np.unique(trial_idx)
# plt.figure(figsize=(11,8))
# for k in range(25):
#     tr = 300 + k
#     plt.subplot(5,5,k+1)
#
#     X_tr = Xt[trial_idx==unq_tr[tr],var_names=='rad_target']
#     plt.plot(X_tr)
#     plt.plot(prd[trial_idx==unq_tr[tr]])
#
# plt.tight_layout()
#
# import statsmodels.api as sm
# non_nan = ~np.isnan(Xt[:,var_names=='rad_vel'])
# model = sm.OLS(Xt[non_nan.flatten(),var_names=='rad_vel'],sm.add_constant(mu[:,non_nan.flatten()].T))
# res = model.fit()
# prd = res.predict(sm.add_constant(mu.T))
#
# unq_tr = np.unique(trial_idx)
# plt.figure(figsize=(11,8))
# for k in range(25):
#     tr = 300 + k
#     plt.subplot(5,5,k+1)
#
#     X_tr = Xt[trial_idx==unq_tr[tr],var_names=='rad_vel']
#     plt.plot(X_tr)
#     plt.plot(prd[trial_idx==unq_tr[tr]])
#
# plt.tight_layout()
#
#
#
#
