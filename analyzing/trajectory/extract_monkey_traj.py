import numpy as np
import sys,os,dill,inspect
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
mainDir = os.path.dirname(os.path.dirname(thisPath))
sys.path.append(os.path.join(mainDir,'GAM_library'))
sys.path.append(os.path.join(mainDir,'firefly_utils'))
sys.path.append(os.path.join(mainDir,'preprocessing_pipeline','util_preproc'))
from spike_times_class import spike_counts
from lfp_class import lfp_class
from copy import deepcopy
from scipy.io import loadmat,savemat
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PIL import ImageColor
from data_handler import *
from path_class import get_paths_class
gen_path = get_paths_class()

# session = 'm53s91'
session_list = ['m53s113', 'm53s83', 'm53s114', 'm53s86',
                'm53s128', 'm53s123', 'm53s124', 'm53s47', 'm53s46', 'm53s31']


# base_file = os.path.join'/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/%s/%s.mat'
# base_file =  '/Users/edoardo/Downloads/%s/%s.mat'
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'
use_left_eye = ['m53s83','m53s84','m53s86','m53s90','m53s92','m53s133','m53s134']

trajectory_all_session = np.zeros(0,dtype=object)

dtype_initial_cond = {'names':('x_fly','y_fly','rad_vel','ang_vel','ang_target','rad_target'),'formats':(float,float,float,float,float,float)}
init_cond = np.zeros(0,dtype=dtype_initial_cond)
dtype_dict = {'names':('session','all', 'reward', 'density', 'ptb', 'microstim', 'landmark', 'replay','controlgain','firefly_fullON'),
            'formats':('U20',bool,int,float,int,int,int,int,float,int)}
info_all_session = np.zeros(0,dtype=dtype_dict)
for session in session_list:

   

    if session in use_left_eye:
        use_eye = 'left'
    else:
        use_eye = 'right'

    base_file = os.path.join(gen_path.get_path('mat_preproc', session),'%s.mat'%session)
    # sv_folder = '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/%s/' % excel_file['Area Recorded'][idx_session]
    # if not os.path.exists(sv_folder):
    #     os.mkdir(sv_folder)



    print('loading session %s...'%session)
    pre_trial_dur = 0.2
    post_trial_dur = 0.2
    try:
        dat = loadmat(base_file)
    except:
        print('could not find', session)

        continue
    #
    # trials_behv = dat[behav_dat_key].flatten()
    # flyon_full = trials_behv['logical'][0]['firefly_fullON'][0,0].flatten()

    is_phase = False
    exp_data = data_handler(dat, behav_dat_key, spike_key, None, behav_stat_key, pre_trial_dur=pre_trial_dur,
                            post_trial_dur=post_trial_dur,
                            extract_lfp_phase=False,
                            use_eye=use_eye,extract_fly_and_monkey_xy=True)
    with open('exp_data_struct_%s.dill'%session,'wb') as fh:
        fh.write(dill.dumps(exp_data))
    
#     exp_data.set_filters('all', True)
#     #del dat
#     print('set info')
#     tmp = np.zeros(exp_data.behav.n_trials,dtype=dtype_dict)
#     for name in tmp.dtype.names[1:]:
#         tmp[name] = exp_data.info.trial_type[name]
#     tmp['session'] = session
#     info_all_session = np.hstack((info_all_session,tmp))
#     del tmp

#     print('extract trials')
#     trajectories = np.zeros(exp_data.behav.n_trials,dtype=object)
#     init_cond_session = np.zeros(exp_data.behav.n_trials,dtype=dtype_initial_cond)
#     for k in range(exp_data.behav.n_trials):
#         sele = (exp_data.behav.time_stamps[k] > exp_data.behav.events.t_targ[k]) * (exp_data.behav.time_stamps[k] <= exp_data.behav.events.t_stop[k])
#         if np.sum(sele)==0:
#             v0 = np.nan
#             w0 = np.nan
#             a0 = np.nan
#             d0 = np.nan
#             x_fly = np.nan
#             y_fly = np.nan
#             xy = np.zeros(0,dtype={'names':('x_monk','y_monk'),'formats':(float,float)})
#         else:
#             idx_on_stop = np.where(sele)[0]
#             v0 = exp_data.behav.continuous.rad_vel[k][idx_on_stop[0]]
#             w0 = exp_data.behav.continuous.ang_vel[k][idx_on_stop[0]]
#             a0 = exp_data.behav.continuous.ang_target[k][idx_on_stop]
#             d0 = exp_data.behav.continuous.rad_target[k][idx_on_stop]

#             try:
#                 a0 = a0[~np.isnan(a0)][0]
#                 d0 = d0[~np.isnan(d0)][0]
#             except:
#                 a0 = np.nan
#                 d0 = np.nan
#             x_fly = exp_data.behav.continuous.x_fly[k]
#             y_fly = exp_data.behav.continuous.y_fly[k]
#             if session.startswith('m91'):
#                 x_fly = -dat['trials_behv'][0]['prs'][k]['xfp'][0][0][0][0]
#                 y_fly = dat['trials_behv'][0]['prs'][k]['yfp'][0][0][0][0]
#             xy = np.zeros(idx_on_stop.shape[0],dtype={'names':('x_monk','y_monk'),'formats':(float,float)})
#             xy['x_monk'] = exp_data.behav.continuous.x_monk[k][sele]
#             xy['y_monk'] = exp_data.behav.continuous.y_monk[k][sele]
#         trajectories[k] = xy
#         init_cond_session['x_fly'][k] = x_fly
#         init_cond_session['y_fly'][k] = y_fly
#         init_cond_session['rad_vel'][k] = v0
#         init_cond_session['ang_vel'][k] = w0
#         init_cond_session['rad_target'][k] = d0
#         init_cond_session['ang_target'][k] = a0

#     init_cond = np.hstack((init_cond,init_cond_session))
#     trajectory_all_session = np.hstack((trajectory_all_session,trajectories))
#     break

# # np.savez('/Volumes/WD Edo/firefly_analysis/LFP_band/processed_data/trajectories_and_info/traj_info.npz',
# #          trajectories=trajectory_all_session,info_all=info_all_session,init_cond=init_cond)


# exp_data.set_filters('reward', True)
# filt_noFLYON = exp_data.info.trial_type['firefly_fullON'] == 0
# idx = np.where(exp_data.filter * filt_noFLYON)[0]
# # exp_data.info.trial_type['reward'] == 1
# plt.close('all')
# plt.figure(figsize=(10,6))
# plt.suptitle('REWARDED')
# for k in range(16):

#     ax=plt.subplot(4,4,k+1)
#     xx = exp_data.behav.continuous.x_monk[idx[k]]
#     yy = exp_data.behav.continuous.y_monk[idx[k]]
#     sele = (exp_data.behav.time_stamps[idx[k]] > exp_data.behav.events.t_targ[idx[k]]) * (exp_data.behav.time_stamps[idx[k]] <= exp_data.behav.events.t_stop[idx[k]])
#     if session.startswith('m91'):
#         x_fly = -dat['trials_behv'][0]['prs'][idx[k]]['xfp'][0][0][0][0]
#         y_fly = -dat['trials_behv'][0]['prs'][idx[k]]['yfp'][0][0][0][0]
#     else:
#         x_fly = exp_data.behav.continuous.x_fly[idx[k]]
#         y_fly = exp_data.behav.continuous.y_fly[idx[k]]
#     print(x_fly,y_fly)
#     plt.plot(xx[sele],yy[sele])
#     plt.plot([x_fly],[y_fly],'ok')
#     tp = np.linspace(0,np.pi*2,1000)
#     cc = np.cos(tp)*60
#     ss = np.sin(tp)*60
#     plt.plot(cc+x_fly,ss+y_fly,'r')
#     plt.ylim(-0,450)
#     plt.xlim(-350,450)
#     ax.set_aspect('equal')


# filt_noFLYON = exp_data.info.trial_type['firefly_fullON'] == 1
# filt_reward = exp_data.info.trial_type['reward'] == 1
# idx = np.where(filt_noFLYON*filt_reward)[0]
# # exp_data.info.trial_type['reward'] == 1
# plt.figure(figsize=(10,6))
# plt.suptitle('FLY FULL ON')
# for k in range(16):

#     ax=plt.subplot(4,4,k+1)
#     xx = exp_data.behav.continuous.x_monk[idx[k]]
#     yy = exp_data.behav.continuous.y_monk[idx[k]]
#     sele = (exp_data.behav.time_stamps[idx[k]] > exp_data.behav.events.t_targ[idx[k]]) * (exp_data.behav.time_stamps[idx[k]] <= exp_data.behav.events.t_stop[idx[k]])
#     if session.startswith('m91'):
#         x_fly = -dat['trials_behv'][0]['prs'][k]['xfp'][0][0][0][0]
#         y_fly = -dat['trials_behv'][0]['prs'][k]['yfp'][0][0][0][0]
#     else:
#         x_fly = exp_data.behav.continuous.x_fly[idx[k]]
#         y_fly = exp_data.behav.continuous.y_fly[idx[k]]
#     print(x_fly,y_fly)
#     plt.plot(xx[sele],yy[sele])
#     plt.plot([x_fly],[y_fly],'ok')
#     tp = np.linspace(0,np.pi*2,1000)
#     cc = np.cos(tp)*60
#     ss = np.sin(tp)*60
#     plt.plot(cc+x_fly,ss+y_fly,'r')
#     plt.ylim(-0,350)
#     plt.xlim(-350,350)
#     ax.set_aspect('equal')

# #
# plt.subplot(111)
# for k in range(100):

#     xx = exp_data.behav.continuous.x_monk[idx[k]]
#     yy = exp_data.behav.continuous.y_monk[idx[k]]
#     sele = (exp_data.behav.time_stamps[idx[k]] > exp_data.behav.events.t_targ[idx[k]]) * (exp_data.behav.time_stamps[idx[k]] <= exp_data.behav.events.t_stop[idx[k]])
#     x_fly = exp_data.behav.continuous.x_fly[idx[k]]
#     y_fly = exp_data.behav.continuous.y_fly[idx[k]]
#     # print(x_fly,y_fly)
#     plt.plot(xx[sele],yy[sele])
#     # plt.plot([x_fly],[y_fly],'ok',ms=10)


