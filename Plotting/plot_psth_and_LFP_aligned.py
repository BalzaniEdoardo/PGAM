import sys,os
sys.path.append('/Users/edoardo/Work/Code/dPCA-master/python/dPCA')
from dPCA import dPCA_CNoise
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
from seaborn import heatmap
from scipy.io import savemat
use_monkey = 'm53'
monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}
monkey_fld = {'m44':'PPC+MST','m91':'PPC+PFC','m53':'PPC+PFC+MST','m51':'PPC'}
file_fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET/%s/'%monkey_fld[use_monkey]

# session = 'm53s110'

list_dir = os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/results/processed_dPCA/')

plt_rewarded = True

matrix_rates_all = np.zeros((0,376))
matrix_beta_all = np.zeros((0,376))
matrix_alpha_all = np.zeros((0,376))
matrix_theta_all = np.zeros((0,376))

brain_area_all = []

dtype_dict = {'names':('monkey','session','brain_area','channel_id','electrode_id','cluster_id','peak_firing_pos','targ_ON_pos',
              'targ_OFF_pos','t_stop_pos','t_reward_pos'),#'median IEI on/off','median IEI off/stop','median IEI stop/reward'
              'formats':('U30',int,'U10',int,int,int,float,float,float,float,float)}


info_all = np.zeros(0,dtype=dtype_dict)
for fh_session in list_dir:

    if not fh_session.endswith('_multiresc_trials.npz'):
        continue
    if fh_session.startswith('flyON'):
        continue

    session = fh_session.split('_multiresc_trials')[0]

    #print(session)
    if not session.startswith(use_monkey):
        continue

    # if 'm51s12' in session:
    #     continue

    if session == 'm51s38':
        continue
    if session != 'm53s113':
        continue
    print(fh_session)
    dat = np.load(
        '/Volumes/WD_Edo/firefly_analysis/LFP_band/results/processed_dPCA/flyON_%s_multiresc_trials.npz' % session)
    dat_info = np.load(os.path.join(file_fld + session + '.npz'), allow_pickle=True)

    unit_info = dat_info['unit_info'].all()
    brain_area = unit_info['brain_area']
    channel_id = unit_info['channel_id']
    electrode_id = unit_info['electrode_id']
    cluster_id = unit_info['cluster_id']

    trial_rad_targ = dat['trial_rad_targ']
    time_bounds = dat['time_bounds']
    time_rescale = dat['time_rescale']
    rescaled_rate = dat['rescaled_rate']  # neu x tria x time point
    rescaled_beta = dat['rescaled_lfp_beta']
    rescaled_alpha = dat['rescaled_lfp_alpha']
    rescaled_theta = dat['rescaled_lfp_theta']
    
    
    
    is_reward = dat['isREW'] == plt_rewarded
    timeOFF = dat['timeOFF']
    timeMOVE = dat['timeMOVE']

    print(np.diff(time_bounds))
    selection = is_reward# (timeOFF < timeMOVE) * is_reward
    rescaled_rate = rescaled_rate[:, selection, :]
    is_reward = is_reward[selection]
    trial_rad_targ = trial_rad_targ[selection]
    rates_sess = np.nanmean(rescaled_rate,axis=1)
    beta_sess = np.nanmean(rescaled_beta, axis=1)
    alpha_sess = np.nanmean(rescaled_alpha,axis=1)
    theta_sess = np.nanmean(rescaled_theta,axis=1)
    
    if brain_area.shape[0] != rates_sess.shape[0]:
        print(session,'brain area not matched with rates')
        continue
    matrix_rates_all = np.vstack((matrix_rates_all,rates_sess))
    brain_area_all = np.hstack((brain_area_all,brain_area))
    matrix_beta_all = np.vstack((matrix_beta_all,beta_sess))
    matrix_alpha_all = np.vstack((matrix_alpha_all,alpha_sess))
    matrix_theta_all = np.vstack((matrix_theta_all,theta_sess))

    
    info_tmp = np.zeros(brain_area.shape[0], dtype=dtype_dict)
    info_tmp['monkey'] = monkey_dict[use_monkey]
    info_tmp['session'] = int(session.split('s')[1])
    info_tmp['channel_id'] = channel_id
    info_tmp['electrode_id'] = electrode_id
    info_tmp['cluster_id'] = cluster_id
    info_tmp['brain_area'] = brain_area

    # extract peak
    z_rate = sts.zscore(rates_sess, axis=1)
    keep = np.sum(np.isnan(z_rate), axis=1) == 0
    idx = np.argmax(z_rate, axis=1)
    info_tmp['peak_firing_pos'][keep] = idx[keep] / (z_rate.shape[1] - 1)
    info_tmp['peak_firing_pos'][~keep] = np.nan

    # time_rescale = np.arange(z_rate.shape[1]) * 0.006
    # extract position of events
    idx_vline = []
    for k in time_bounds[1:]:
        idx_vline += [np.where(time_rescale <= k)[0][-1]]
    idx_vline = np.hstack(([0], idx_vline))

    info_tmp['targ_ON_pos'] = idx_vline[0] / (z_rate.shape[1] - 1)
    info_tmp['targ_OFF_pos'] = idx_vline[1] / (z_rate.shape[1] - 1)
    info_tmp['t_stop_pos'] = idx_vline[2] / (z_rate.shape[1] - 1)
    info_tmp['t_reward_pos'] = idx_vline[3] / (z_rate.shape[1] - 1)
    # info_tmp['median IEI on/off'] = time_bounds[1] - time_bounds[0]
    # info_tmp['median IEI off/stop'] = time_bounds[2] - time_bounds[1]
    # info_tmp['median IEI stop/reward'] = time_bounds[3] - time_bounds[2]

    info_all = np.hstack((info_all,info_tmp))

# savemat('%s_peak_location_isRew_%s.mat'%(monkey_dict[use_monkey],plt_rewarded),{'peak_location':info_all})

for ba in ['MST','PPC','PFC']:
    if not ba in file_fld:
        continue
    plt.figure()
    cc_plot = 0
    plt.suptitle('RATE: %s - %s'%(use_monkey,ba))

    ba_rate = matrix_rates_all[brain_area_all==ba,:]
    zba_rate = sts.zscore(ba_rate,axis=1)
    keep = np.sum(np.isnan(zba_rate),axis=1) == 0
    zba_rate = zba_rate[keep,:]
    idx = np.argmax(zba_rate,axis=1)
    sort_idx = np.argsort(idx)
    heatmap(zba_rate[sort_idx,:],vmin=-3.,vmax=4.5)
    idx_vline = []
    for k in time_bounds[1:]:
        idx_vline += [np.where(time_rescale <= k)[0][-1]]
    ylim = plt.ylim()

    plt.vlines(idx_vline,ylim[0],ylim[1])
    plt.ylim(ylim)
    plt.yticks([])
    plt.ylabel('unit')
    xaxis = np.hstack(([0],idx_vline))
    plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('%s_rescaled_rate_multialign_%s_isRew_%s.png'%(ba,use_monkey,plt_rewarded))

    
for ba in ['MST','PPC','PFC']:
    if not ba in file_fld:
        continue
    plt.figure()
    cc_plot = 0
    plt.suptitle('BETA POWER: %s - %s'%(use_monkey,ba))
    
    # sort based on the rate
    ba_rate = matrix_rates_all[brain_area_all==ba,:]
    zba_rate = sts.zscore(ba_rate,axis=1)
    keep = np.sum(np.isnan(zba_rate),axis=1) == 0
    zba_rate = zba_rate[keep,:]
    idx = np.argmax(zba_rate,axis=1)
    sort_idx = np.argsort(idx)
    
    ba_beta = matrix_beta_all[brain_area_all==ba,:]
    zba_beta = sts.zscore(ba_beta,axis=1)
    zba_beta = zba_beta[keep,:]
    
    heatmap(zba_beta[sort_idx,:],vmin=-3.,vmax=4.5)
    idx_vline = []
    for k in time_bounds[1:]:
        idx_vline += [np.where(time_rescale <= k)[0][-1]]
    ylim = plt.ylim()

    plt.vlines(idx_vline,ylim[0],ylim[1])
    plt.ylim(ylim)
    plt.yticks([])
    plt.ylabel('unit')
    xaxis = np.hstack(([0],idx_vline))
    plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('%s_rescaled_BETA_multialign_%s_isRew_%s.png'%(ba,use_monkey,plt_rewarded))
    
    
for ba in ['MST','PPC','PFC']:
    if not ba in file_fld:
        continue
    plt.figure()
    cc_plot = 0
    plt.suptitle('THETA POWER: %s - %s'%(use_monkey,ba))
    
    # sort based on the rate
    ba_rate = matrix_rates_all[brain_area_all==ba,:]
    zba_rate = sts.zscore(ba_rate,axis=1)
    keep = np.sum(np.isnan(zba_rate),axis=1) == 0
    zba_rate = zba_rate[keep,:]
    idx = np.argmax(zba_rate,axis=1)
    sort_idx = np.argsort(idx)
    
    ba_beta = matrix_theta_all[brain_area_all==ba,:]
    zba_beta = sts.zscore(ba_beta,axis=1)
    zba_beta = zba_beta[keep,:]
    
    heatmap(zba_beta[sort_idx,:],vmin=-3.,vmax=4.5)
    idx_vline = []
    for k in time_bounds[1:]:
        idx_vline += [np.where(time_rescale <= k)[0][-1]]
    ylim = plt.ylim()

    plt.vlines(idx_vline,ylim[0],ylim[1])
    plt.ylim(ylim)
    plt.yticks([])
    plt.ylabel('unit')
    xaxis = np.hstack(([0],idx_vline))
    plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('%s_rescaled_THETA_multialign_%s_isRew_%s.png'%(ba,use_monkey,plt_rewarded))
    

for ba in ['MST','PPC','PFC']:
    if not ba in file_fld:
        continue
    plt.figure()
    cc_plot = 0
    plt.suptitle('ALPHA POWER: %s - %s'%(use_monkey,ba))
    
    # sort based on the rate
    ba_rate = matrix_rates_all[brain_area_all==ba,:]
    zba_rate = sts.zscore(ba_rate,axis=1)
    keep = np.sum(np.isnan(zba_rate),axis=1) == 0
    zba_rate = zba_rate[keep,:]
    idx = np.argmax(zba_rate,axis=1)
    sort_idx = np.argsort(idx)
    
    ba_beta = matrix_alpha_all[brain_area_all==ba,:]
    zba_beta = sts.zscore(ba_beta,axis=1)
    zba_beta = zba_beta[keep,:]
    
    heatmap(zba_beta[sort_idx,:],vmin=-3.,vmax=4.5)
    idx_vline = []
    for k in time_bounds[1:]:
        idx_vline += [np.where(time_rescale <= k)[0][-1]]
    ylim = plt.ylim()

    plt.vlines(idx_vline,ylim[0],ylim[1])
    plt.ylim(ylim)
    plt.yticks([])
    plt.ylabel('unit')
    xaxis = np.hstack(([0],idx_vline))
    plt.xticks(xaxis,['targ ON','targ OFF','stop','reward'],rotation=90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('%s_rescaled_ALPHA_multialign_%s_isRew_%s.png'%(ba,use_monkey,plt_rewarded))
    
    # neu_list = np.arange(rescaled_rate.shape[0])
    # sele = np.where(brain_area == ba)[0]
    # neu_list = np.array(neu_list[sele],dtype=int)
    # for neu in neu_list:
    #     if cc_plot == 16:
    #         cc_plot = 0
    #
    #         plt.figure(figsize=[11,8])
    #         plt.suptitle(ba)
    #     plt.subplot(4,4,cc_plot+1)
    #
    #     mn = np.nanmean(rescaled_rate[neu,is_reward,:],axis=0)
    #     plt.plot(time_rescale,mn)
    #     cc_plot+=1
    #     plt.vlines(time_bounds,mn.min(),mn.max(),colors='k')

#
# iei_start_stop = []
# sess_id = np.sort(np.unique(info_all['session']))
# for session in sess_id:
#     info_sess = info_all[info_all['session']==session][0]
#     iei_start_stop += [info_sess['median IEI off/stop']]