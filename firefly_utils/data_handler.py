import numpy as np
from spike_times_class import spike_counts
from behav_class import behavior_experiment,load_trial_types
from lfp_class import lfp_class
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import interp1d

def dict_to_vec(dictionary):
    return np.hstack(list(dictionary.values()))

def time_stamps_rebin(time_stamps, binwidth_ms=20):
    rebin = {}
    for tr in time_stamps.keys():
        ts = time_stamps[tr]
        tp_num = np.floor((ts[-1] - ts[0]) * 1000 / (binwidth_ms))
        rebin[tr] = ts[0] + np.arange(tp_num) * binwidth_ms / 1000.
    return rebin



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

    def GPFA_YU_preprocessing(self, list_timepoints=None, var_list=[], pcaPrep=False, sqrtIfPCA=True,
                              filt_window=None,smooth=True):
        if list_timepoints is None:
            list_timepoints = [('t_move','t_stop',75),
                               ('t_stop','t_reward',15)
                               ]


        trial_use = np.arange(self.spikes.n_trials)[self.filter]#self.spikes.n_trials
        n_trials = trial_use.shape[0]

        # check if the events are consecutive
        check_ev0 = []
        check_ev1 = []

        tot_tp = 0
        for ev0, ev1, tp in list_timepoints:
            check_ev0 += [ev0]
            check_ev1 += [ev1]
            tot_tp += tp - 1

        # tot_tp -= 1

        tp_matrix = np.zeros((n_trials, tot_tp)) * np.nan


        rate_tensor = np.zeros((n_trials,self.spikes.num_units, tot_tp)) * np.nan
        sm_traj = np.zeros((n_trials, 2, tot_tp)) * np.nan # xy position
        raw_traj = np.zeros((n_trials,2, tot_tp)) * np.nan
        fly_pos = np.zeros((n_trials, 2)) * np.nan

        tw_correlates = {}
        for var in var_list:
            tw_correlates[var] = np.zeros((n_trials, tot_tp)) * np.nan



        check_ev0 = np.array(check_ev0[1:])
        check_ev1 = np.array(check_ev1[:-1])
        assert(all(check_ev0==check_ev1))

        if pcaPrep:
            # smooth spikes
            ev0 = list_timepoints[0][0]
            ev1 = list_timepoints[-1][1]
            add = 0
            add_stop = 0
            if ev0 == 't_flyON':
                ev0 = 't_targ'
                add = 0

            elif ev0 == 't_flyOFF' or ev0 == 't_targ_off':
                ev0 = 't_targ'
                add = self.behav.flyON_dur

            if ev1 == 't_flyON':
                ev1 = 't_targ'
                add_stop = 0
            elif ev1 == 't_flyOFF' or ev1 == 't_targ_off':
                ev1 = 't_targ'
                add_stop = self.behav.flyON_dur

            if smooth:
                t_start = dict_to_vec(self.behav.events.__dict__[ev0]) + add - self.behav.pre_trial_dur
                t_stop = dict_to_vec(self.behav.events.__dict__[ev1]) + add_stop + self.behav.pre_trial_dur
                time_dict = self.spikes.bin_spikes(self.behav.time_stamps, t_start=t_start,t_stop=t_stop)
                DT = time_dict[0][1] - time_dict[0][0]
                print('begin smoothing spikes for PCA')
                sm_spikes = np.zeros(self.spikes.binned_spikes.shape,dtype=object)
                for tr in range(self.spikes.binned_spikes.shape[1]):
                    for un in range(self.spikes.binned_spikes.shape[0]):
                        sm_spikes[un,tr] = np.convolve(self.spikes.binned_spikes[un,tr]/DT,filt_window, mode='same')
                print('end smoothing spikes for PCA')

        if sqrtIfPCA and pcaPrep:
            transFun = lambda x:np.sqrt(x)
        elif pcaPrep:
            transFun = lambda x:x
        # loop over trials
        for indx_tr in range(n_trials):

            tr = trial_use[indx_tr]
            # spk_times = self.spikes.spike_times[:,tr]
            time_bins = []

            # extract smooth trajectories
            #trajectory_tr = np.zeros(len(self.behav.time_stamps[tr]))*np.nan
            traj_sele = (self.behav.time_stamps[tr] > self.behav.events.t_targ[tr]) * (
                         self.behav.time_stamps[tr] <= self.behav.events.t_stop[tr])

            Num = traj_sele.sum()
            valid_tr = any(traj_sele) and (Num > 20)
            if valid_tr:
                x_fly = self.behav.continuous.x_fly[tr]
                y_fly = self.behav.continuous.y_fly[tr]
                ts = self.behav.time_stamps[tr][traj_sele]
                x_monk = self.behav.continuous.x_monk[tr][traj_sele]
                y_monk = self.behav.continuous.y_monk[tr][traj_sele]


                fr = 20. / Num
                # print(tr,fr)
                non_nan = ~np.isnan(x_monk)
                x_smooth = np.nan * np.zeros((x_monk.shape[0],2))
                y_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                x_smooth[non_nan,:] = sm.nonparametric.lowess(x_monk, np.arange(x_monk.shape[0]), fr)
                y_smooth[non_nan,:] = sm.nonparametric.lowess(y_monk, np.arange(y_monk.shape[0]), fr)
                x_smooth = x_smooth[:, 1]
                y_smooth = y_smooth[:, 1]

                fly_pos[indx_tr, 0] = x_fly
                fly_pos[indx_tr, 1] = y_fly

            skip_trial = False
            cc = 1
            for ev0, ev1, tp in list_timepoints:

                if ev0 == 't_flyON':
                    ev0 = 't_targ'

                elif ev0 == 't_flyOFF':
                    ev0 = 't_targ_off'

                if ev1 == 't_flyON':
                    ev1 = 't_targ'

                elif ev1 == 't_flyOFF':
                    ev1 = 't_targ_off'

                if ev0 != 't_targ_off':
                    t0 = self.behav.events.__dict__[ev0][tr][0]
                else:
                    t0 = self.behav.events.__dict__['t_targ'][tr][0] + self.behav.flyON_dur

                if ev1 != 't_targ_off':
                    t1 = self.behav.events.__dict__[ev1][tr][0]
                else:
                    t1 = self.behav.events.__dict__['t_targ'][tr][0] + self.behav.flyON_dur

                if any(np.isnan([t0,t1])):
                    skip_trial = True
                    break

                if t1 < t0:
                    skip_trial = True
                    break

                time_lst = np.linspace(t0, t1, tp)
                if cc != len(list_timepoints):
                    time_lst = time_lst[:-1]

                time_bins = np.hstack((time_bins,time_lst))
                cc += 1

            if skip_trial:
                print('skipping trial %d'%tr)
                continue

            tp_matrix[indx_tr, :] = 0.5*(time_bins[:-1] + time_bins[1:])
            time_int_dur = np.diff(time_bins)

            if (not pcaPrep) or (not smooth):
                for unt in range(self.spikes.num_units):
                    rate_tensor[indx_tr, unt, :] = np.histogram(self.spikes.spike_times[unt, tr], bins=time_bins)[0] / time_int_dur
            else:
                # print('start interp smooth spike for PCA')
                for unt in range(self.spikes.num_units):
                    interp = interp1d(time_dict[tr], transFun(sm_spikes[unt,tr]),bounds_error=False)
                    rate_tensor[indx_tr, unt, :] = interp(tp_matrix[indx_tr, :])
                # print('end interp smooth spike for PCA')

            # compute linearly interp trajectory position
            sele_tp = (ts >= time_bins[0]) & (ts < time_bins[-1])
            if not any(sele_tp):
                continue
            # smooth interp
            intrp = interp1d(ts[sele_tp], x_smooth[sele_tp],bounds_error=False)
            sm_traj[indx_tr, 0] = intrp(tp_matrix[indx_tr, :])
            intrp = interp1d(ts[sele_tp], y_smooth[sele_tp],bounds_error=False)
            sm_traj[indx_tr, 1] = intrp(tp_matrix[indx_tr, :])

            # raw interp
            intrp = interp1d(ts[sele_tp], x_monk[sele_tp],bounds_error=False)
            raw_traj[indx_tr, 0] = intrp(tp_matrix[indx_tr, :])
            intrp = interp1d(ts[sele_tp], y_monk[sele_tp],bounds_error=False)
            raw_traj[indx_tr, 1] = intrp(tp_matrix[indx_tr, :])

            # interp variables
            for var in var_list:
                time_pts = self.behav.time_stamps[tr]
                y_val = self.behav.continuous.__dict__[var][tr]
                non_nan = ~np.isnan(y_val)
                intrp = interp1d(time_pts[non_nan], y_val[non_nan], bounds_error=False)
                tw_correlates[var][indx_tr,:] = intrp(tp_matrix[indx_tr, :])


        return tp_matrix, rate_tensor, sm_traj, raw_traj, fly_pos, tw_correlates, trial_use


    def GPFA_YU_preprocessing_noTW(self, t_start, t_stop, var_list=[],binwidth_ms=20):
        if binwidth_ms is None:
            bin_ts = self.behav.time_stamps
        else:
            bin_ts = time_stamps_rebin(self.behav.time_stamps, binwidth_ms=binwidth_ms)
        bin_list = self.spikes.bin_spikes(bin_ts, t_start=t_start, t_stop=t_stop, select=self.filter)
        trialId = {}
        spikes = {}
        tr_sel = np.array(np.arange(self.spikes.n_trials)[self.filter], dtype=int)
        ydim = self.spikes.binned_spikes.shape[0]

        sm_traj = np.zeros((tr_sel.shape[0], 2), dtype=object)
        raw_traj = np.zeros((tr_sel.shape[0], 2), dtype=object)
        fly_pos = np.zeros((tr_sel.shape[0], 2))*np.nan

        tw_correlates = {}
        bbin_ts = {}

        for var in var_list:
            tw_correlates[var] = np.zeros((tr_sel.shape[0],),dtype=object)

        for cc in range(tr_sel.shape[0]):

            tr = tr_sel[cc]
            # extract smooth trajectories
            trajectory_tr = np.zeros(len(self.behav.time_stamps[tr])) * np.nan
            traj_sele = (self.behav.time_stamps[tr] > self.behav.events.t_targ[tr]) * (
                    self.behav.time_stamps[tr] <= self.behav.events.t_stop[tr])

            Num = traj_sele.sum()
            valid_tr = any(traj_sele) and (Num > 20)
            if valid_tr:
                x_fly = self.behav.continuous.x_fly[tr]
                y_fly = self.behav.continuous.y_fly[tr]
                ts = self.behav.time_stamps[tr][traj_sele]
                x_monk = self.behav.continuous.x_monk[tr][traj_sele]
                y_monk = self.behav.continuous.y_monk[tr][traj_sele]

                fr = 20. / Num
                # print(tr,fr)
                non_nan = ~np.isnan(x_monk)
                x_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                y_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                x_smooth[non_nan, :] = sm.nonparametric.lowess(x_monk, np.arange(x_monk.shape[0]), fr)
                y_smooth[non_nan, :] = sm.nonparametric.lowess(y_monk, np.arange(y_monk.shape[0]), fr)
                x_smooth = x_smooth[:, 1]
                y_smooth = y_smooth[:, 1]

                fly_pos[cc, 0] = x_fly
                fly_pos[cc, 1] = y_fly


            tdim = self.spikes.binned_spikes[0, cc].shape[0]
            spikes[cc] = np.zeros((ydim, tdim))
            for i in range(ydim):
                spikes[cc][i, :] = self.spikes.binned_spikes[i, cc]
            trialId[cc] = tr_sel[cc]

            # smooth interp
            intrp = interp1d(self.behav.time_stamps[tr][traj_sele], x_smooth, bounds_error=False)
            sm_traj[cc, 0] = intrp(bin_list[tr])
            intrp = interp1d(self.behav.time_stamps[tr][traj_sele], y_smooth, bounds_error=False)
            sm_traj[cc, 1] = intrp(bin_list[tr])

            # raw interp
            intrp = interp1d(self.behav.time_stamps[tr][traj_sele], x_monk, bounds_error=False)
            raw_traj[cc, 0] = intrp(bin_list[tr])
            intrp = interp1d(self.behav.time_stamps[tr][traj_sele], y_monk, bounds_error=False)
            raw_traj[cc, 1] = intrp(bin_list[tr])

            # interp variables
            for var in var_list:
                time_pts = self.behav.time_stamps[tr]
                y_val = self.behav.continuous.__dict__[var][tr]
                non_nan = ~np.isnan(y_val)
                # try:
                intrp = interp1d(time_pts[non_nan], y_val[non_nan], bounds_error=False)
                # except:
                #     ccc=1
                tw_correlates[var][cc] = intrp(bin_list[tr])
            bbin_ts[cc] = bin_list[tr]
            # cc += 1
        # remove ts of other trials
        # cc = 1
        # for tr in tr_sel:
        #     bbin_ts[cc] = bin_ts[tr]
        #     cc+=1

        return bbin_ts, spikes, sm_traj, raw_traj, fly_pos, tw_correlates, trialId


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