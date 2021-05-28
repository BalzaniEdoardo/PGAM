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
from scipy.stats import linregress
import statsmodels.api as sm


def compute_traj(vel,ang_vel,dt=0.006):
    x = [0]
    y = [0]
    theta = 0
    theta_vec = [0]
    for k in range(vel.shape[0]):
        theta = theta + ang_vel[k]*dt/360.
        x += [x[-1] + dt * vel[k] * np.sin(theta)]
        y += [y[-1] + dt * vel[k] * np.cos(theta)]
        theta_vec += [theta]

    return x,y,theta_vec
gen_path = get_paths_class()

# session = 'm53s91'
session_list = ['m73s3']


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

    base_file = os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/','%s.mat'%session)
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
    
    exp_data.set_filters('all',True)

    # for k in range(100,140):
    #     tr = np.where(exp_data.filter)[0][k]
    #
    #     istart = np.where(exp_data.behav.time_stamps[tr] >= 0)[0][0]
    #     istop = np.where(exp_data.behav.time_stamps[tr] >= exp_data.behav.events.t_stop[tr])[0][0]
    #
    #     x_smr = exp_data.behav.continuous.x_monk[tr][istart+1:istop]
    #     y_smr = exp_data.behav.continuous.y_monk[tr][istart + 1:istop]
    #     vel = exp_data.behav.continuous.rad_vel[tr][istart + 1:istop]
    #     w = exp_data.behav.continuous.ang_vel[tr][istart + 1:istop]
    #
    #     x_rec,y_rec,tht = compute_traj(vel,w)
    #     # plt.subplot(121)
    #     # plt.plot(x_smr-x_smr[0],y_smr-y_smr[0])
    #     # plt.subplot(122)
    #     # plt.plot(x_rec[1:], y_rec[1:])
    #     print((x_smr[0] - x_smr[-1]) / (x_rec[0] - x_rec[-1]),
    #           exp_data.behav.events.t_stop[tr])

    break

    tr_vec = np.arange(exp_data.behav.n_trials)[exp_data.filter]
    plt.figure(figsize=(10,8))
    kk=1
    for tr in tr_vec[:10]:
        plt.subplot(2,5,kk)
        r_path = exp_data.behav.continuous.rad_path[tr]
        r_path3 = exp_data.behav.continuous.rad_path_from_xy[tr]
        x_monk = exp_data.behav.continuous.x_monk[tr]
        y_monk = exp_data.behav.continuous.y_monk[tr]
        t_end = exp_data.behav.events.t_end[tr]/0.006
        t_start = exp_data.behav.events.t_targ[tr]/0.006
        t_stop = exp_data.behav.events.t_stop[tr]/0.006

        sele = ~np.isnan(x_monk)
        r_path2 = np.sqrt(x_monk**2 + (y_monk+32)**2)
        plt.plot(r_path[sele],label='orig r_path')
        plt.plot(r_path2[sele],label='new r_path')
        plt.plot(r_path3[sele],label='new r_path sm')


        fr = 20./sele.sum()
        # x = np.arange(sele.sum())*0.006

        # rsm = sm.nonparametric.lowess(r_path2[sele], exp_data.behav.time_stamps[tr][sele],fr)
        # plt.plot(rsm[:,1])
        ylim = plt.ylim()
        plt.plot([t_end,t_end],ylim,'k')
        plt.plot([t_stop,t_stop],ylim,'b')
        plt.plot([t_start,t_start],ylim,'b')
        # plt.plot([np.argmin(r_path2[sele]),np.argmin(r_path2[sele])],ylim,'r--')
        # print(np.argmin(r_path2[sele]))
        plt.ylim(ylim)
        kk+=1
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10,8))
    kk=1
    for tr in tr_vec[:20]:
        plt.subplot(4,5,kk)
        r_path = exp_data.behav.continuous.rad_path[tr]
        x_monk = exp_data.behav.continuous.x_monk[tr]
        y_monk = exp_data.behav.continuous.y_monk[tr]
        sele = ~np.isnan(x_monk)
        r_path2 = np.sqrt(x_monk**2 + y_monk**2)
        # rad_path
        # plt.plot(r_path[sele],label='orig r_path')
        # plt.plot(r_path2[sele],label='new r_path')
        res = linregress(r_path2[sele][100:-60],r_path[sele][100:-60])

        lab = 'slope: %.3f'%(res.slope)

        plt.scatter(r_path2[sele][100:-60],r_path[sele][100:-60],label=lab)
        plt.legend(fontsize=8)




        kk+=1
    # plt.legend()
    plt.tight_layout()

    break
    # with open('exp_data_struct_%s.dill'%session,'wb') as fh:
    #     fh.write(dill.dumps(exp_data))
    


