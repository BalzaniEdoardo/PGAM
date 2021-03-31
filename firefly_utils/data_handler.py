import numpy as np
from spike_times_class import spike_counts
from behav_class import behavior_experiment,load_trial_types
from lfp_class import lfp_class
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt

def dict_to_vec(dictionary):
    return np.hstack(list(dictionary.values()))




class data_handler(object):
    def __init__(self,dat,beh_key,spike_key,lfp_key,behav_stat_key,time_aligned_to_beh=True,dt=0.006,
                 flyON_dur=0.3,pre_trial_dur=0.25,post_trial_dur=0.25,is_lfp_binned=True, extract_lfp_phase=True,
                 lfp_beta=None,lfp_alpha=None,lfp_theta=None,use_eye=None,extract_fly_and_monkey_xy=False,
                 extract_cartesian_eye_and_firefly=False,fhLFP=''):

        self.info = load_trial_types(dat[behav_stat_key].flatten(),dat[beh_key].flatten())
        # import all data and trial info
        self.spikes = spike_counts(dat,spike_key,time_aligned_to_beh=time_aligned_to_beh)
        if lfp_key is None:
            self.lfp = None
        else:
            self.lfp = lfp_class(dat,lfp_key, binned=is_lfp_binned,
                                 lfp_beta=lfp_beta,lfp_alpha=lfp_alpha,
                                 lfp_theta=lfp_theta,compute_phase=extract_lfp_phase,
                                 fhLFP=fhLFP)
        self.behav = behavior_experiment(dat,beh_key,behav_stat_key=behav_stat_key,dt=dt,flyON_dur=flyON_dur,
                                         pre_trial_dur=pre_trial_dur,post_trial_dur=post_trial_dur,info=self.info,use_eye=use_eye,
                                         extract_fly_and_monkey_xy=extract_fly_and_monkey_xy,
                                         extract_cartesian_eye_and_firefly=extract_cartesian_eye_and_firefly)

        self.date_exp = datetime.strptime(dat['prs']['sess_date'][0, 0][0],'%d-%b-%Y')


        # set the filter to trials in which the monkey worked
        self.filter = self.info.get_all(True)
        # save a dicitonary with the info regarding the selected trial
        self.filter_descr = {'all':True}

    def align_spike_times_to_beh(self):
        print('Method still empty')
        return

    def compute_train_and_test_filter(self,perc_train_trial=0.8,seed=None):
        if ~ (seed is None):
            np.random.seed(seed)

        # compute how many of the selected trials will be in the training set
        num_selected = np.sum(self.filter)
        tot_train = int(perc_train_trial * num_selected)


        # make sure that the trial we select are in the filtered
        choiche_idx = np.arange(self.spikes.n_trials)[self.filter]
        # select the training set
        train = np.zeros(self.spikes.n_trials, dtype=bool)
        train_idx = np.random.choice(choiche_idx,size=tot_train,replace=False)
        train[train_idx] = True

        test = (~train) * self.filter

        return train,test

    def concatenate_inputs(self,*varnames,t_start=None,t_stop=None):
        time_stamps = deepcopy(self.behav.time_stamps)

        self.spikes.bin_spikes(time_stamps, t_start=t_start, t_stop=t_stop, select=self.filter)

        edges_sel = np.arange(self.spikes.n_trials)[self.filter]

        spikes = self.spikes.binned_spikes

        # count the input data shape
        cc = 0
        for tr in range(spikes.shape[1]):
            cc += spikes[0,tr].shape[0]

        # stack all spike counts in a single vector per each unit
        tmp_spikes = np.zeros((spikes.shape[0],cc))
        trial_idx = np.zeros(cc,dtype=int)

        for unt in range(spikes.shape[0]):
            cc = 0
            for tr in range(spikes.shape[1]):
                d_idx = spikes[unt,tr].shape[0]
                tmp_spikes[unt,cc:cc+d_idx] = spikes[unt,tr]
                trial_idx[cc:cc+d_idx] = edges_sel[tr]
                cc += d_idx

        spikes = tmp_spikes



        event_names = list(self.behav.events.__dict__.keys())
        continuous_names = list(self.behav.continuous.__dict__.keys())
        var_dict = {}

        for var in varnames:
            if var in event_names:
                events = self.behav.events.__dict__[var]
                var_dict[var] = self.behav.create_event_time_binned(events,time_stamps,t_start=t_start,t_stop=t_stop,select=self.filter)

            elif var in continuous_names:
                continuous = self.behav.continuous.__dict__[var]
                var_dict[var] = self.behav.cut_continuous( continuous, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)
            elif var == 'phase':
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                phase = self.lfp.extract_phase(all_tr,self.spikes.channel_id, self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(phase, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)

            elif var == 'lfp_beta':
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # assert (self.lfp.compute_phase)
                phase = self.lfp.extract_phase_x_unit(self.lfp.lfp_beta,all_tr,self.spikes.channel_id,
                                                      self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(phase, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)
            elif var == 'lfp_beta_power':
               # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # assert (self.lfp.compute_phase)
                amplitude = self.lfp.extract_phase_x_unit(self.lfp.lfp_beta_power,all_tr,self.spikes.channel_id,
                                                      self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(amplitude, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)
            elif var == 'lfp_alpha':
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # assert (self.lfp.compute_phase)
                phase = self.lfp.extract_phase_x_unit(self.lfp.lfp_alpha, all_tr, self.spikes.channel_id,
                                                      self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(phase, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)
            elif var == 'lfp_alpha_power':
               # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # assert (self.lfp.compute_phase)
                amplitude = self.lfp.extract_phase_x_unit(self.lfp.lfp_alpha_power,all_tr,self.spikes.channel_id,
                                                      self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(amplitude, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)
            elif var == 'lfp_theta':
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # assert (self.lfp.compute_phase)
                phase = self.lfp.extract_phase_x_unit(self.lfp.lfp_theta, all_tr, self.spikes.channel_id,
                                                      self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(phase, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)
            elif var == 'lfp_theta_power':
               # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # assert (self.lfp.compute_phase)
                amplitude = self.lfp.extract_phase_x_unit(self.lfp.lfp_theta_power,all_tr,self.spikes.channel_id,
                                                      self.spikes.brain_area)
                var_dict[var] = self.lfp.cut_phase(amplitude, time_stamps, t_start=t_start, t_stop=t_stop,
                                                          select=self.filter,idx0=None,idx1=None)

            # elif var == 'phase':
                # # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                # all_tr = np.ones(self.lfp.n_trials,dtype=bool)
                # phase = self.lfp.extract_phase(all_tr,self.spikes.channel_id, self.spikes.brain_area)
                # var_dict[var] = self.lfp.cut_phase(phase, time_stamps, t_start=t_start, t_stop=t_stop,
                #                                           select=self.filter,idx0=None,idx1=None)

            else:
                raise ValueError('variable %s is unknown'%var)
            if not (var in ['phase','lfp_beta','lfp_alpha','lfp_theta',
                            'lfp_beta_power','lfp_theta_power','lfp_alpha_power']):
                var_dict[var] = dict_to_vec(var_dict[var])
            else:
                first = True
                for unit in range(var_dict[var].shape[0]):
                    phase = np.hstack(var_dict[var][unit,:])
                    if first:
                        first = False
                        phase_stack = np.zeros((var_dict[var].shape[0],phase.shape[0]))
                    phase_stack[unit,:] = phase
                var_dict[var] = phase_stack

            # check that the variables have same sizes
            if not (var in ['phase','lfp_beta','lfp_alpha','lfp_theta',
                            'lfp_beta_power','lfp_theta_power','lfp_alpha_power']):
                if var_dict[var].shape[0] != spikes.shape[1]:
                    raise ValueError('%s counts and spike counts have different sizes'%var)
            else:
                if var_dict[var].shape[1] != spikes.shape[1]:
                    raise ValueError('%s counts and spike counts have different sizes'%var)

        return spikes,var_dict,trial_idx

    def set_filters(self,*filter_settings):
        # check that the required input is even
        if len(filter_settings) % 2 != 0:
            raise ValueError('Must input a list of field names and input values')
        # list of acceptable field names
        trial_type_list = list(self.info.dytpe_names)
        print(trial_type_list)
        # number of trials
        n_trials = self.behav.n_trials
        filter = np.ones(n_trials, dtype=bool)
        descr = {}
        for k in range(0,len(filter_settings),2):
            # get the name and check that is valid
            field_name = filter_settings[k]
            if not (field_name in trial_type_list):
                print('Filter not set. Invalid field name: "%s"'%field_name)
                return
            value = filter_settings[k+1]
            func = self.info.__getattribute__('get_' + field_name)
            if np.isscalar(value):
                filter = filter * func(value)
            else:
                filter = filter * func(*value)

            descr[field_name] = value

        self.filter = filter
        self.filter_descr = descr
        print('Succesfully set filter')




if __name__ == '__main__':

    from copy import deepcopy
    from scipy.io import loadmat
    from behav_class import *
    print('start loading...')
    dat = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/m53s127_new.mat')
    lfp_beta = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/lfp_beta_m53s127.mat')
    lfp_alpha = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/lfp_alpha_m53s127.mat')
    lfp_theta = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/lfp_theta_m53s127.mat')
    print(dat.keys())
    behav_stat_key = 'behv_stats'
    spike_key = 'units'
    behav_dat_key = 'trials_behv'
    lfp_key = 'lfps'

    pre_trial_dur = 0.5
    post_trial_dur = 0.5
    exp_data = data_handler(dat,behav_dat_key,spike_key,lfp_key,behav_stat_key,pre_trial_dur=pre_trial_dur,post_trial_dur=post_trial_dur,
                            lfp_beta=lfp_beta['lfp_beta'],lfp_alpha=lfp_alpha['lfp_alpha'],extract_lfp_phase=True)
    exp_data.set_filters('all',True)
    train,test = exp_data.compute_train_and_test_filter(seed=3)

    t_targ = dict_to_vec(exp_data.behav.events.t_targ)
    t_move = dict_to_vec(exp_data.behav.events.t_move)

    t_start = np.min(np.vstack((t_move,t_targ)),axis=0) - pre_trial_dur
    t_stop = dict_to_vec(exp_data.behav.events.t_end) + post_trial_dur

    var_names = ['phase']# 'rad_vel','ang_vel','rad_path','ang_path','hand_vel1','hand_vel2','phase','t_move','t_flyOFF','t_stop','t_reward','rad_path','ang_path'
    var_alias = {'rad_vel':'v',
                 'ang_vel':'w',
                 'rad_path':'d',
                 'ang_path':'phi',
                 'hand_vel1':'h1',
                 'hand_vel2':'h2',
                 'lfp_beta':'lfp_beta',
                 'lfp_alpha': 'lfp_alpha',
                 't_move':'move',
                 't_flyOFF':'target_OFF',
                 't_stop': 'stop',
                 't_reward':'reward'}
    # var_names = ['t_flyOFF','t_move']
    var_names = ['lfp_alpha','lfp_beta','phase']
    y,X,trial_idx = exp_data.concatenate_inputs(*var_names,t_start=t_start,t_stop=t_stop)
    for key in X.keys():
        print(key, X[key].shape)

    # res = loadmat('/Users/edoardo/Work/Code/Angelaki-Savin/Kaushik/concatenated_trials.mat')
    # Yt=res['Yt']
    # print('tot spike count diff',np.prod(y.shape)-np.sum(y.T==Yt))
    # for key in var_alias.keys():
    #     try:
    #         exp_data.behav.continuous.__getattribute__(key)
    #         if key == 'hand_vel1' or key == 'hand_vel2':
    #             print(key, np.max(np.abs(res[var_alias[key]].flatten() - X[key]/100.)))
    #         else:
    #             print(key, np.max(np.abs(res[var_alias[key]].flatten() - X[key])))
    #     except AttributeError:
    #         pass
    #     try:
    #         exp_data.behav.events.__getattribute__(key)
    #         print(key, np.sum(np.abs(res[var_alias[key]].flatten() - X[key])!=0))
    #     except AttributeError:
    #         pass

    # for k in list(range(6))+list(range(7,12)):
    #     if var_names[:2] == 'ha':
    #         print(var_names[k], np.max(np.abs(res['xt'][:, k] - X[var_names[k]]/100)))
    #     else:
    #         print(var_names[k],np.max(np.abs(res['xt'][:,k] - X[var_names[k]])))

    # for k in range(1,182):
    #     tmp = loadmat('/Volumes/WD Edo/test_lfp_phase/concatenated_lfp_%d.mat'%k)['phaseComp'].flatten()
    #     print(np.max(np.abs(X['phase'][k-1,:]-tmp)))