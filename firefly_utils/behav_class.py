import numpy as np
import statsmodels.api as sm

def create_dict_beahv(trialVec,var_type,field):
    """
    Description
    ===========
        This function parses matlab array of nested structure with 2 fields, var_type and field. It returns a dictionary
        that has as keys the number of trials and values the extracted vector with the time series
    """
    new_dict = {}
    for key in range(trialVec.shape[0]):
        new_dict[key] = trialVec[key][var_type][field].all().flatten()
    return new_dict

def creat_dict_from_beahv_stat(beahv_stat,vartype,field):
    new_dict = {}
    vec_trials = beahv_stat[vartype][0][field][0, 0].flatten()
    for k in range(vec_trials.shape[0]):
        new_dict[k] = vec_trials[k].flatten()
    return new_dict

def load_eye_pos(trials_behv,use_eye=None):
    """
    Description
    ===========
        This function loads horizontal end vertical eye position in time per each trial. It returns a dict with
        keys the trials and values the time series of eye positions.
    """
    # Extraxt the dictionary from the nested matlab structure
    eye_hori_left = create_dict_beahv(trials_behv, 'continuous', 'yle')
    eye_hori_right = create_dict_beahv(trials_behv, 'continuous', 'yre')

    eye_vert_left = create_dict_beahv(trials_behv, 'continuous', 'zle')
    eye_vert_right = create_dict_beahv(trials_behv, 'continuous', 'zre')

    # check if right or left eye has been tracked and save the horizontal position
    eye_hori = {}
    eye_vert = {}
    for key in eye_hori_left.keys():


        if (use_eye is None) or (use_eye == 'right'):
            use_eye = 'right'
            if np.prod(np.isnan(eye_hori_right[key])):
                eye_hori[key] = eye_hori_left[key]
            else:
                eye_hori[key] = eye_hori_right[key]


            if np.prod(np.isnan(eye_vert_right[key])):
                eye_vert[key] = eye_vert_left[key]
            else:
                eye_vert[key] = eye_vert_right[key]

        elif use_eye == 'left':
            if np.prod(np.isnan(eye_hori_left[key])):
                eye_hori[key] = eye_hori_right[key]
            else:
                eye_hori[key] = eye_hori_left[key]

            if np.prod(np.isnan(eye_vert_left[key])):
                eye_vert[key] = eye_vert_right[key]
            else:
                eye_vert[key] = eye_vert_left[key]

    return eye_hori,eye_vert, use_eye

# def load_eye_pos_cartesian(trials_behv,use_eye=None):
#     """
#     Description
#     ===========
#         This function loads horizontal end vertical eye position in time per each trial. It returns a dict with
#         keys the trials and values the time series of eye positions.
#     """
#     # Extraxt the dictionary from the nested matlab structure
#     eye_x_left = create_dict_beahv(trials_behv, 'continuous', 'xlep')
#     eye_x_right = create_dict_beahv(trials_behv, 'continuous', 'xrep')
#
#     eye_y_left = create_dict_beahv(trials_behv, 'continuous', 'ylep')
#     eye_y_right = create_dict_beahv(trials_behv, 'continuous', 'yrep')
#
#     # check if right or left eye has been tracked and save the horizontal position
#     eye_x = {}
#     eye_y = {}
#     for key in eye_x_left.keys():
#
#
#         if (use_eye is None) or (use_eye == 'right'):
#             if np.prod(np.isnan(eye_x_right[key])):
#                 eye_x[key] = eye_x_left[key]
#             else:
#                 eye_x[key] = eye_x_right[key]
#
#
#             if np.prod(np.isnan(eye_y_right[key])):
#                 eye_y[key] = eye_y_left[key]
#             else:
#                 eye_y[key] = eye_y_right[key]
#
#         elif use_eye == 'left':
#             if np.prod(np.isnan(eye_x_left[key])):
#                 eye_x[key] = eye_x_right[key]
#             else:
#                 eye_x[key] = eye_x_left[key]
#
#             if np.prod(np.isnan(eye_y_left[key])):
#                 eye_y[key] = eye_y_right[key]
#             else:
#                 eye_y[key] = eye_y[key]
#
#     return eye_x,eye_y


class emptyStruct(object):
    def __init__(self):
        return
    def __repr__(self):
        strprint = 'emptyStruct with attr: \n' 
        for key in self.__dict__.keys():
            strprint += '%s:\t%s\n'%(key,self.__dict__[key])
        return strprint
            
                                    

class behavior_experiment(object):
    """
    Description
    ===========
        This funciton imports the data from a behavioural experiment. It splits the data into two categories:
            1. continuous:  all the time series data sampled in time, with a time resolution of dt
                i) eye_hori, eye_vert: monkey eye position in the horizontal and vertical axis
                ii) rad_vel, ang_vel: radial and angular velocity in the virtual space, 1 to 1 corrispondence with the
                joystick position (cm/s)
                iii) rad_path, ang_path: position in the virtual reality computed by integrating the vel in time (cm)
                iv) hand_vel1, hand_vel2: PCA 1st and 2nd component of hand velocity. Hand movement is tracked using
                8 different nodes on the hand, this 8dim hand position data are projected into 2 pcs and a velocity is
                computed in pc space (check out the data to see if the extimation of the velocity is not too noisy)
            2. events: single time stamped events
                i) t_move: time stamp of movement start
                ii) t_flyOFF: time stamp of target OFF
                iii) t_stop: time stamp of movement stop
                iv) t_reward: time stamps of reward releas (if rewarded, otherwise NaN)
        Time bins are saved into behavior_experiment.time_stamps.
    """
    def __init__(self,dat,behav_key,behav_stat_key='behav_stat',dt=0.006,flyON_dur=0.3,pre_trial_dur=0.25,post_trial_dur=0.25,info=None,
                 use_eye=None,extract_fly_and_monkey_xy=False,
                 extract_cartesian_eye_and_firefly=False):
        # get the behavioral variable out
        trials_behv = dat[behav_key].flatten()
        behav_stat = dat[behav_stat_key].flatten()

        self.events = emptyStruct()
        self.continuous = emptyStruct()
        
        self.prs = np.zeros(len(trials_behv), dtype=object)
        self.add_prs(trials_behv)

        self.n_trials = trials_behv.shape[0]
        self.trbeh = trials_behv
        # only for replay trials
        if 'trial_id' in trials_behv.dtype.names:
            self.trial_id = np.squeeze(np.hstack(trials_behv['trial_id']))

        # sample freq for the behavior
        self.dt =  dt
        # duration of the pre/post-trial (experiment dependent)
        self.pre_trial_dur = pre_trial_dur
        self.post_trial_dur = post_trial_dur

        # duration in sec of target presentation
        self.flyON_dur = flyON_dur

        # trial timestamps (for each trial dt ms sequence of timepoints)
        self.time_stamps = create_dict_beahv(trials_behv,'continuous','ts')

        # extract eye position
        self.continuous.eye_hori, self.continuous.eye_vert, self.use_eye = load_eye_pos(trials_behv,use_eye=use_eye)
        # extrct virtualr real velocity (cm/s)
        self.continuous.rad_vel = create_dict_beahv(trials_behv,'continuous','v')
        self.continuous.ang_vel = create_dict_beahv(trials_behv, 'continuous', 'w')
        
        try:
            self.continuous.rad_vel_diff = create_dict_beahv(trials_behv,'continuous','v_diff')
            self.continuous.ang_vel_diff = create_dict_beahv(trials_behv, 'continuous', 'w_diff')
            self.continuous.rad_vel_ptb = create_dict_beahv(trials_behv,'continuous','v_ptb')
            self.continuous.ang_vel_ptb = create_dict_beahv(trials_behv, 'continuous', 'w_ptb')
            
        except:
            self.continuous.rad_vel_diff = None
            self.continuous.ang_vel_diff = None
            self.continuous.rad_vel_ptb = None
            self.continuous.ang_vel_ptb = None
        # extract PCs hand velocities (cm/s)
        try:
            self.continuous.hand_vel1 = create_dict_beahv(trials_behv, 'continuous', 'h1')
            self.continuous.hand_vel2 = create_dict_beahv(trials_behv, 'continuous', 'h2')
        except:
            self.continuous.hand_vel1 = None
            self.continuous.hand_vel2 = None

        self.continuous.rad_target = creat_dict_from_beahv_stat(behav_stat, 'pos_rel', 'r_targ')
        self.continuous.ang_target = creat_dict_from_beahv_stat(behav_stat, 'pos_rel', 'theta_targ')


        try:
            self.continuous.rad_acc = creat_dict_from_beahv_stat(behav_stat,'accel','radial')
            self.continuous.ang_acc = creat_dict_from_beahv_stat(behav_stat,'accel','angular')
        
        except:
            self.continuous.rad_acc = None
            self.continuous.ang_acc = None
        
        try:
            self.continuous.rad_acc_diff = creat_dict_from_beahv_stat(behav_stat,'accel','radial_diff')
            self.continuous.ang_acc_diff = creat_dict_from_beahv_stat(behav_stat,'accel','angular_diff')
        except:
            self.continuous.rad_acc_diff = None
            self.continuous.ang_acc_diff = None
        
        # integrate radial and angular path
        self.continuous.rad_path = self.itegrate_path(self.continuous.rad_vel)
        self.continuous.ang_path = self.itegrate_path(self.continuous.ang_vel)
        try:
            self.continuous.true_hor_mean = self.extract_eye_track(behav_stat,info)
        except:
            print('no eyetracking...')

        # time of perturbation
        try:
            self.events.t_ptb = create_dict_beahv(trials_behv, 'events', 't_ptb')
        except:
            print('no t_ptb')
        try:
            self.events.t_ptb = create_dict_beahv(trials_behv, 'events', 't_ptbn')
            print('SUBSTITUTE T_PTB with realigned')
        except:
            print('no normalized t_ptb (t_ptbn)')
        # time of movement start
        self.events.t_move = create_dict_beahv(trials_behv, 'events', 't_move')

        # compute the time of fly off
        self.events.t_flyOFF = self.t_flyOFF_compute(trials_behv)

        # time of stop movement
        self.events.t_stop = create_dict_beahv(trials_behv, 'events', 't_stop')

        # time of reward
        self.events.t_reward = create_dict_beahv(trials_behv, 'events', 't_rew')
        # t_targ ??
        self.events.t_targ = create_dict_beahv(trials_behv, 'events', 't_targ')
        # t_end??
        self.events.t_end = create_dict_beahv(trials_behv, 'events', 't_end')
        
        
        # here set the vel_diff as the normal vel if no ptb and the vel_ptb as 0
        if not self.continuous.rad_vel_diff is None:
            for tr in self.continuous.rad_vel_diff.keys():
                # if trial is without ptb add vel
                if np.isnan(self.events.t_ptb[tr][0]):
                    self.continuous.rad_vel_diff[tr] = self.continuous.rad_vel[tr]
                    self.continuous.ang_vel_diff[tr] = self.continuous.ang_vel[tr]
                    self.continuous.rad_vel_ptb[tr] = np.zeros(self.continuous.rad_vel[tr].shape)
                    self.continuous.ang_vel_ptb[tr] = np.zeros(self.continuous.rad_vel[tr].shape)
        
        if extract_fly_and_monkey_xy:
            self.continuous.x_monk = creat_dict_from_beahv_stat(behav_stat, 'pos_abs', 'x_monk')
            self.continuous.y_monk = creat_dict_from_beahv_stat(behav_stat, 'pos_abs', 'y_monk')
            self.get_fly_pos(trials_behv)
            #self.continuous.rad_path_from_xy = self.radial_distance_from_position()

        if extract_cartesian_eye_and_firefly:
            height = dat['prs']['height'][0][0][0][0]
            screen_dist = dat['prs']['screendist'][0][0][0][0]
            interocular_dist = dat['prs']['interoculardist'][0][0][0][0]
            self.get_fly_pos(trials_behv)
            xmp = create_dict_beahv(trials_behv, 'continuous', 'xmp')
            ymp = create_dict_beahv(trials_behv, 'continuous', 'ymp')

            # rotation
            R = lambda theta : np.array([[np.cos(theta/180*np.pi),-np.sin(theta/180*np.pi)],[np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)]])

            # fly position in monkey cartesian coord (monkey center is at 0, shoulder are x-axis, heading is y-axis)
            xfp_rel = {}
            yfp_rel = {}
            # eye position in monkey cartesiona
            xep_rel = {}
            yep_rel = {}

            # coordinate of the fly on the screen (avoid effect of horizon in the tangent transform)
            fly_screen_x = {}
            fly_screen_z = {}

            eye_screen_x = {}
            eye_screen_z = {}
            for tr in xmp.keys():
                ts = self.time_stamps[tr]
                w = self.continuous.ang_vel[tr]
                x_fly_rel = self.continuous.x_fly[tr] - xmp[tr]
                y_fly_rel = self.continuous.y_fly[tr] - ymp[tr]

                phi = dt * np.cumsum(w * (ts>0))

                XY = np.zeros((2,x_fly_rel.shape[0]))
                XY[0,:] = x_fly_rel
                XY[1,:] = y_fly_rel

                rot = R(phi)
                XY = np.einsum('ijk,jk->ik', rot, XY)

                xfp_rel[tr] = XY[0, :]
                yfp_rel[tr] = XY[1, :]

                eye_hori,eye_vert = self.continuous.eye_hori[tr], self.continuous.eye_vert[tr]
                vert_rad = eye_vert * np.pi / 180
                hori_rad = eye_hori * np.pi / 180


                yrep = height / np.tan(-vert_rad)
                yrep[yrep < 0] = np.nan
                yep_rel[tr] = yrep

                xrep = yrep * np.tan(hori_rad)
                xrep[xrep < 0] = np.nan
                if self.use_eye == 'right':
                    xep_rel[tr] = xrep# - interocular_dist / 2
                else:
                    xep_rel[tr] = xrep #+ interocular_dist / 2

                # get the screen coordinate of fly
                fly_screen_z[tr] = height - screen_dist * height / yfp_rel[tr]
                fly_screen_x[tr] = screen_dist * xfp_rel[tr] / yfp_rel[tr]
                
                # when y relative to monkey is negative, the target is behind,
                # therefore it is not on the screen
                fly_screen_z[tr][yfp_rel[tr] <= 0] = np.nan
                fly_screen_x[tr][yfp_rel[tr] <= 0] = np.nan

                # get the screen coord of eye position (better to use the displacement)
                eye_screen_z[tr] = height - screen_dist * np.tan(-vert_rad)
                if self.use_eye == 'right':
                    eye_screen_x[tr] = screen_dist * np.tan(hori_rad) #+ interocular_dist / 2
                else:
                    eye_screen_x[tr] = screen_dist * np.tan(hori_rad) #- interocular_dist / 2

            self.continuous.x_eye_rel, self.continuous.y_eye_rel = xep_rel, yep_rel
            self.continuous.x_fly_rel, self.continuous.y_fly_rel = xfp_rel, yfp_rel

            self.continuous.x_eye_screen, self.continuous.z_eye_screen = eye_screen_x, eye_screen_z
            self.continuous.x_fly_screen, self.continuous.z_fly_screen = fly_screen_x, fly_screen_z


    def add_prs(self, trials_behv):
        prs = trials_behv['prs']
        for k in range(len(prs)):
            self.prs[k] = emptyStruct()
            for name in prs[k].dtype.names:
                 setattr(self.prs[k], name, prs[k][name][0][0][0,0])
                
            
        
    def get_fly_pos(self,trials_behv):
        """
        indx_beg = find(continuous(i).ts > events(i).t_targ, 1); % sample number of target onset time
    indx_stop = find(continuous(i).ts > events(i).t_stop, 1); % sample number of stopping time
    x_fly(i) = nanmedian(continuous(i).xfp(indx_beg:indx_stop)); y_fly(i) = nanmedian(continuous(i).yfp(indx_beg:indx_stop));
        :return:
        """
        x_fly = create_dict_beahv(trials_behv, 'continuous', 'xfp')
        y_fly = create_dict_beahv(trials_behv, 'continuous', 'yfp')
        self.continuous.x_fly = {}
        self.continuous.y_fly = {}
        for i in self.time_stamps.keys():
            i_beg = np.where(self.time_stamps[i] > self.events.t_targ[i])[0][0]
            i_stop = np.where(self.time_stamps[i] > self.events.t_stop[i])[0][0]
            self.continuous.x_fly[i] = np.nanmedian(x_fly[i][i_beg: i_stop])
            self.continuous.y_fly[i] = np.nanmedian(y_fly[i][i_beg: i_stop])
        return

    def extract_eye_track(self,behav_stat,info):

        ver_mean_true = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['true'].all()[
            'ver_mean'].all()['val'][0,0]
        ver_diff_true = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['true'].all()[
            'ver_diff'].all()['val'][0,0]
        hor_mean_true = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['true'].all()[
            'hor_mean'].all()['val'][0,0]
        hor_diff_true = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['true'].all()[
            'hor_diff'].all()['val'][0,0]

        ver_mean_pred = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['pred'].all()[
            'ver_mean'].all()['val'][0,0]
        ver_diff_pred = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['pred'].all()[
            'ver_diff'].all()['val'][0,0]
        hor_mean_pred = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['pred'].all()[
            'hor_mean'].all()['val'][0,0]
        hor_diff_pred = behav_stat['trialtype'].all()['all'].all()['eye_movement'].all()['eyepos'].all()['pred'].all()[
            'hor_diff'].all()['val'][0,0]


        self.continuous.ver_mean_true = {}
        self.continuous.ver_diff_true = {}
        self.continuous.hor_mean_true = {}
        self.continuous.hor_diff_true = {}

        self.continuous.ver_mean_pred = {}
        self.continuous.ver_diff_pred = {}
        self.continuous.hor_mean_pred = {}
        self.continuous.hor_diff_pred = {}

        # this trials are related to all trials index, set to None when the trial is not in all...
        all_idx = np.where(info.trial_type['all'])[0]
        for k in range(ver_mean_true.shape[1]):
            self.continuous.ver_mean_true[all_idx[k]] = ver_mean_true[0, k].flatten()
            self.continuous.ver_diff_true[all_idx[k]] = ver_diff_true[0, k].flatten()
            self.continuous.hor_mean_true[all_idx[k]] = hor_mean_true[0, k].flatten()
            self.continuous.hor_diff_true[all_idx[k]] = hor_diff_true[0, k].flatten()

            self.continuous.ver_mean_pred[all_idx[k]] = ver_mean_pred[0, k].flatten()
            self.continuous.ver_diff_pred[all_idx[k]] = ver_diff_pred[0, k].flatten()
            self.continuous.hor_mean_pred[all_idx[k]] = hor_mean_pred[0, k].flatten()
            self.continuous.hor_diff_pred[all_idx[k]] = hor_diff_pred[0, k].flatten()


        return

    def itegrate_path(self,velocity):
        """
        Description
        ===========
            This function compute the trajectory in time given a time course of velocities.
        """
        # set a function that cumulates the distance trabelled and zero pad negative times (before stim presentation)
        zeropad_and_cumulative = lambda ts,vel: np.hstack((np.zeros(np.sum(ts <= 0)), np.cumsum(vel[ts > 0]*self.dt)))
        # this function cicles on dictionaries and combines the results_radTarg into a dicitonary
        dictFuct = lambda ts,vel: {key : zeropad_and_cumulative(ts[key],vel[key]) for key in vel.keys()}

        integr_path = dictFuct(self.time_stamps,velocity)

        return integr_path
    
    def radial_distance_from_position(self):
        """
        Description
        ===========
            This function compute the trajectory in time given a time course of velocities.
        """
        path = {}
        for tr in range(self.n_trials):
            print('path smoothing: %d/%d'%(tr+1,self.n_trials))
            sele = (self.time_stamps[tr] > 0) & ( self.time_stamps[tr] < self.events.t_end[tr])
            x_monk = self.continuous.x_monk[tr][sele]
            y_monk = self.continuous.y_monk[tr][sele]
            
            non_nan_idx = np.where(~np.isnan(x_monk))[0]
            # if np.isnan(x_monk[0]):
            #     sele[np.where(sele)[0][0] + 1] = False
            #     x_monk = x_monk[1:]
            #     y_monk = y_monk[1:]
            # sele[sele][np.isnan(x_monk)] = False
            raw_path = np.sqrt((x_monk)**2 + (y_monk + 32)**2)
            fr = 20. / sele.sum()
            rsm = sm.nonparametric.lowess(raw_path, self.time_stamps[tr][sele],fr)
            sm_path = np.zeros(x_monk.shape)*np.nan
            sm_path[non_nan_idx] = rsm[:,1]
            path[tr] = np.zeros(sele.shape)
            # the origin is (0cm,-32cm) 
            path[tr][sele] = sm_path
            
        return path
            
        # # set a function that cumulates the distance trabelled and zero pad negative times (before stim presentation)
        # zeropad_and_cumulative = lambda ts,vel: np.hstack((np.zeros(np.sum(ts <= 0)), np.cumsum(vel[ts > 0]*self.dt)))
        # # this function cicles on dictionaries and combines the results_radTarg into a dicitonary
        # dictFuct = lambda ts,vel: {key : zeropad_and_cumulative(ts[key],vel[key]) for key in vel.keys()}

        # integr_path = dictFuct(self.time_stamps,velocity)

        return integr_path

    def t_flyOFF_compute(self,trials_behv):
        """
        Description
        ===========
            The function simply adds to the t_flyON the duration of the target on time.
        """
        dict_offtime = {key: trials_behv[key]['events']['t_targ'].all().flatten() + self.flyON_dur for key in range(trials_behv.shape[0])}
        return dict_offtime

    def create_event_time_binned(self,event,edges,t_start=None,t_stop=None,select=None):
        bin_event = {}
        if not select is None:
            edges_sel = np.arange(self.n_trials)[select]
        else:
            edges_sel = np.arange(self.n_trials)

        # get the event time in the selected trial
        ii = 0
        for tr in edges_sel:
            event_tr = np.squeeze(event[tr])
            edge_tr = np.array(edges[tr][1:-1],dtype=float)
            bin_event[ii] = np.hstack((np.diff(edge_tr > event_tr),[0]))

            # if t_start is set to None it means that edges must not be cut
            if t_start is None:
                t0 = -np.inf
                t1 = np.inf
            else:
                # if start is a scalar, always takes times greater than t_start
                if np.isscalar(t_start):
                    t0 = t_start
                # otherwise consider as start the time t0 indicated by the dictionary with t_starts
                else:
                    t0 = float(t_start[tr])
                # same for t_stop
                if np.isscalar(t_stop):
                    t1 = t_stop
                else:
                    t1 = float(t_stop[tr])

            bin_event[ii] = bin_event[ii][(edge_tr > t0) * (edge_tr < t1)]
            ii += 1
        return bin_event

    def cut_continuous(self, continuous, edges, t_start=None, t_stop=None, select=None,idx0=None,idx1=None):

        bin_continous = {}
        if not select is None:
            edges_sel = np.arange(self.n_trials)[select]
        else:
            edges_sel = np.arange(self.n_trials)

        if idx0 is None:
            idx0 = 0

        # get the event time in the selected trial
        ii = 0
        i_idx1 = idx1
        for tr in edges_sel:
            if idx1 is None:
                i_idx1 = continuous[tr].shape[0]

            continuous_tr = continuous[tr][idx0:i_idx1]
            

            # edge_tr = edges[tr]
            # get the same numerosity as Kaushik
            edge_tr = np.array(edges[tr][1:-1],dtype=float)
            continuous_tr = continuous_tr[1:-1]

            # if t_start is set to None it means that edges must not be cut
            if t_start is None:
                t0 = -np.inf
                t1 = np.inf
            else:
                # if start is a scalar, always takes times greater than t_start
                if np.isscalar(t_start):
                    t0 = t_start
                # otherwise consider as start the time t0 indicated by the dictionary with t_starts
                else:
                    t0 = float(t_start[tr])
                # same for t_stop
                if np.isscalar(t_stop):
                    t1 = t_stop
                else:
                    t1 = float(t_stop[tr])

            bin_continous[ii] = continuous_tr[(edge_tr > t0) * (edge_tr < t1)]
            ii += 1
        return bin_continous

class load_trial_types(object):
    """
    Description
    ===========
        This function create an array with the trial types that can be used for filtering the required trial
    """
    def __init__(self,beh_stats,trials_behv,skip_not_ok=True):
        ttype = beh_stats['trialtype'].all()
        # extract trial number from a nested matlab struct
        n_trials = ttype['all'].all()['trlindx'].all().flatten().shape[0]
        dict_types = {
            'names':('all', 'reward', 'density', 'ptb', 'microstim', 'landmark', 'replay','controlgain','firefly_fullON'),
            'formats':(bool,int,float,int,int,int,int,float,int)
        }
        self.trial_type = np.zeros(n_trials,dtype=dict_types)

        self.trial_type['all'] = ttype['all'].all()['trlindx'].all().flatten()
        # self.trial_type['ptb'] = ttype['ptb'].all()['trlindx'].all().flatten()
        # self.trial_type['microstim'] = ttype['microstim'].all()['trlindx'].all().flatten()
        # self.trial_type['landmark'] = ttype['landmark'].all()['trlindx'].all().flatten()
        # self.trial_type['replay'] = ttype['replay'].all()['trlindx'].all().flatten()
        if 'microstim' in list(ttype.dtype.names):
            self.set_trial_type_microstim(trials_behv,skip_not_ok=skip_not_ok)
        if 'replay' in list(ttype.dtype.names):
            self.set_trial_type_replay(ttype)
            try:
                self.paired_trials = pair_replay_and_active(trials_behv)
            except:
                print('UNABLE TO PAIR REPLAY AND ACTIVE')
        if 'density' in list(ttype.dtype.names):
            self.set_trial_type_density(ttype,trials_behv,skip_not_ok=skip_not_ok)
        if 'reward' in list(ttype.dtype.names):
            self.set_trial_type_reward(ttype, trials_behv,skip_not_ok=skip_not_ok)
        if 'ptb' in list(ttype.dtype.names):
            self.set_trial_type_ptb(trials_behv,skip_not_ok=skip_not_ok)
        if 'landmark' in list(ttype.dtype.names):
            self.set_trial_type_landmark(ttype)
        if 'controlgain' in list(ttype.dtype.names):
            self.set_trial_type_controlgain(ttype,trials_behv,skip_not_ok=skip_not_ok)
        if 'firefly_fullON' in trials_behv['logical'][0].dtype.names:
            self.set_trial_type_firefly_fullON(trials_behv)

        self.dytpe_names = self.trial_type.dtype.names

    def set_trial_type_firefly_fullON(self, trials_behv):
        for k in range(trials_behv['logical'].shape[0]):
            self.trial_type['firefly_fullON'][k] = trials_behv['logical'][k]['firefly_fullON'][0,0][0,0]

    def set_trial_type_controlgain(self, ttype, trials_behv,skip_not_ok=True):
        self.trial_type['controlgain'] = 1.
        
        struc_array = ttype['controlgain'].all()
        # set unclassified trials as -1
        #self.trial_type['controlgain'] = -1
        for k in range(struc_array.shape[1]):
            descr = struc_array[0, k]['val'][0]
            if descr.startswith('gain ='):
                flag = float(descr.split('gain =')[1])
            else:
                raise ValueError('description must start with "gain ="')
            trialIndx = np.array(struc_array[0, k]['trlindx'].flatten(), dtype=bool)
            self.trial_type['controlgain'][trialIndx] = flag
        ok_trials = self.trial_type['all']
        if skip_not_ok:
            self.trial_type['controlgain'][~ok_trials] = -1
        

    def set_trial_type_landmark(self, ttype):
        struc_array = ttype['landmark'].all()
        # set unclassified trials as -1
        self.trial_type['landmark'] = 0
        for k in range(struc_array.shape[1]):
            descr = struc_array[0, k]['val'][0]
            if 'with landmark' in descr:
                flag = 1
            elif 'without landmark' in descr:
                flag = 0
            else:
                raise ValueError('description must be: without/with landmark')
            trialIndx = np.array(struc_array[0, k]['trlindx'].flatten(), dtype=bool)
            self.trial_type['landmark'][trialIndx] = flag
        # ok_trials = self.trial_type['all']
        # self.trial_type['landmark'][~ok_trials] = -1

    def set_trial_type_density(self,ttype, trials_behv, skip_not_ok=True):
        ok_trials = self.trial_type['all']
        struc_array = ttype['density'].all()
        self.trial_type['density'] = np.nan
        try:
            for k in range(self.trial_type.shape[0]):
                self.trial_type['density'][k] = float(np.squeeze(trials_behv['prs'][k]['floordensity'][0][0]))
        except:
            for k in range(struc_array.shape[1]):
                descr = struc_array[0,k]['val'][0]
                try:
                    density = float('0.'+descr.split('0.')[1])
                except IndexError:
                    density = float(descr.split('=')[1])
                # density = float(descr.split('=')[1].rstrip().lstrip())
                trialIndx = np.array(struc_array[0,k]['trlindx'].flatten(),dtype=bool)
                self.trial_type['density'][trialIndx] = density
        ok_trials = self.trial_type['all']

        if skip_not_ok:
            self.trial_type['density'][~ok_trials] = np.nan
            
        if np.prod(np.isnan(self.trial_type['density'])):
            raise Warning('some trial was not classified for the density of texture elements')

    def set_trial_type_microstim(self, trials_behv,skip_not_ok=True):
        self.trial_type['microstim'] = 0
        for k in range(self.trial_type.shape[0]):
            self.trial_type['microstim'][k] = float(np.squeeze(trials_behv['logical'][k]['microstim'][0][0]))
        ok_trials = self.trial_type['all']
        if skip_not_ok:
            self.trial_type['microstim'][~ok_trials] = -1
      


    def set_trial_type_ptb(self, trials_behv, skip_not_ok=True):
        self.trial_type['ptb'] = 0
        for k in range(self.trial_type.shape[0]):
            self.trial_type['ptb'][k] = float(np.squeeze(trials_behv['logical'][k]['ptb'][0][0]))
        ok_trials = self.trial_type['all']
        if skip_not_ok:
            self.trial_type['ptb'][~ok_trials] = -1
        



    def set_trial_type_reward(self,ttype, trials_behv,skip_not_ok=True):
        # struc_array = ttype['reward'].all()
        # set unclassified trials as -1
        self.trial_type['reward'] = 0
        tmp = np.zeros(self.trial_type.shape[0])
        
        for k in range(self.trial_type.shape[0]):
            tmp[k] = float(np.squeeze(trials_behv['logical'][k]['reward'][0][0]))


        self.trial_type['reward'] = tmp
        ok_trials = self.trial_type['all']
        if skip_not_ok:
            self.trial_type['reward'][~ok_trials] = -1

    def set_trial_type_replay(self,ttype):
        struc_array = ttype['replay'].all()
        # set unclassified trials as -1
        self.trial_type['replay'] = 0
        for k in range(struc_array.shape[1]):
            descr = struc_array[0,k]['val'][0]
            if 'active behaviour' in descr:
                flag = 0
            elif 'replay behaviour' in descr:
                flag = 1
            else:
                raise ValueError('description must be: active or passive behaviour')

            trialIndx = np.array(struc_array[0, k]['trlindx'].flatten(),dtype=bool)
            self.trial_type['replay'][trialIndx] = flag
        # DON'T check ok trials

    
    def get_controlgain(self, gain ,skip_not_ok=True):
    
            if np.isnan(gain):
                filter = np.isnan(self.trial_type['controlgain'])
            else:
                controlgain_levels = list(np.unique(self.trial_type['controlgain'][~np.isnan(self.trial_type['controlgain'])]))
                if not gain in controlgain_levels:
                    raise ValueError('Controlgain must be one of the following: ' + (controlgain_levels+[np.nan]).__repr__())
                filter = self.trial_type['controlgain'] == gain
                if skip_not_ok:
                    ok_trials = self.trial_type['all']
                    filter[~ok_trials] = False
            return filter
    
    def get_density(self,density,skip_not_ok=True):

        if np.isnan(density):
            filter = np.isnan(self.trial_type['density'])
        else:
            density_levels = list(np.unique(self.trial_type['density'][~np.isnan(self.trial_type['density'])]))
            if not density in density_levels:
                raise ValueError('Density must be one of the following: ' + (density_levels+[np.nan]).__repr__())
            filter = self.trial_type['density'] == density
            if skip_not_ok:
                ok_trials = self.trial_type['all']
                filter[~ok_trials] = False
        return filter

    def get_reward(self,reward,skip_not_ok=True):
        if not reward in [-1,1,0]:
            raise ValueError('reward must be 1 for rewarded, 0 for unrewarded, -1 otherwise')
        filter = self.trial_type['reward'] == reward
        if skip_not_ok:
            ok_trials = self.trial_type['all']
            filter[~ok_trials] = False
        return filter

    def get_ptb(self,perturbation,skip_not_ok=True):
        filter = self.trial_type['ptb'] == perturbation
        if skip_not_ok:
            ok_trials = self.trial_type['all']
            filter[~ok_trials] = False
        return filter

    def get_microstim(self,microstim,skip_not_ok=True):
        filter = self.trial_type['microstim'] == microstim
        if skip_not_ok:
            ok_trials = self.trial_type['all']
            filter[~ok_trials] = False
        return filter

    def get_landmark(self,landmark,skip_not_ok=True):
        filter = self.trial_type['landmark'] == landmark
        if skip_not_ok:
            ok_trials = self.trial_type['all']
            filter[~ok_trials] = False
        return filter

    def get_replay(self,replay,skip_not_ok=True):
        filter = self.trial_type['replay'] == replay
        if skip_not_ok:
            ok_trials = self.trial_type['all']
            filter[~ok_trials] = False
        return filter

    def get_all(self,all):
        filter = self.trial_type['all'] == all
        return filter

def pair_replay_and_active(trials_behv):
    # this based on the fact that active and replay are in blocks of equal size
    repl_bool = np.zeros(trials_behv.shape[0])
    iei = np.zeros((2,len(trials_behv)))
    for tr in range(trials_behv.shape[0]):
        repl_bool[tr] = trials_behv[tr]['logical']['replay'].all()[0][0]
        iei[0,tr] = trials_behv[tr]['events']['t_stop'].all()[0][0] - trials_behv[tr]['events']['t_move'].all()[0][0]
        iei[1,tr] = trials_behv[tr]['events']['t_move'].all()[0][0] - trials_behv[tr]['events']['t_flyON'].all()[0][0]

    num_blocks = (np.diff(repl_bool)!=0).sum() + 1
    if num_blocks < 2:
        print('unable to extract pairs, different blocks number')
        return
    block_switch = np.hstack((np.where(np.diff(repl_bool) != 0)[0],[repl_bool.shape[0]]))
    edge_0 = 0
    blocks_active = []
    blocks_replay = []

    for edge_1 in block_switch:
        if repl_bool[edge_0] == 0: # active
            blocks_active += [(edge_0,edge_1)]
        if repl_bool[edge_0] == 1: # replay
            blocks_replay += [(edge_0,edge_1)]
        edge_0 = edge_1 + 1

    # if num blocks differ cannot pair
    if len(blocks_active) != len(blocks_replay):
        print('unable to extract pairs, different blocks number')
        return

    # get the iei

    pair_trials = np.zeros(0,dtype={'names':('active','replay'),'formats':(float,float)})

    for k in range(len(blocks_replay)):
        e0_active,e1_active = blocks_active[k]
        e0_repl, e1_repl = blocks_replay[k]
        if (e1_active - e0_active) == (e1_repl - e0_repl):
            repl_idx = np.arange(e0_repl, e1_repl)
            active_idx = np.arange(e0_active, e1_active)
        else:
            M = np.argmax(((e1_active - e0_active) ,(e1_repl - e0_repl)))

            v0 = iei[0, blocks_replay[0][0]:blocks_replay[0][1]]
            v1 = iei[0, blocks_active[0][0]:blocks_active[0][1]]
            mxidx = np.argmax(np.correlate(v0,v1,mode='valid'))
            if M == 0:
                active_idx = np.arange(e0_active, e1_active)
                repl_idx = np.zeros(e1_active-e0_active)*np.nan
                repl_idx[mxidx: e1_repl - e0_repl + mxidx] = np.arange(e0_repl, e1_repl)
            else:
                active_idx = np.zeros(e1_repl - e0_repl) * np.nan
                repl_idx = np.arange(e0_repl, e1_repl)
                active_idx[mxidx: e1_active - e0_active + mxidx] = np.arange(e0_active, e1_active)


        non_nan = ~(np.isnan(repl_idx) | np.isnan(active_idx))
        if np.corrcoef(iei[0,np.array(repl_idx[non_nan],dtype=int)],iei[0,np.array(active_idx[non_nan],dtype=int)])[0,1] < 0.99:
            print('unable to extract pairs,uncorrelated inter event intervals')
            return

        pair = np.zeros(((e1_active - e0_active)),dtype={'names':('active','replay'),'formats':(float,float)})
        pair['active'] = active_idx
        pair['replay'] = repl_idx

        pair_trials = np.hstack((pair_trials, pair))
    return pair_trials

if __name__ == '__main__':
    from scipy.io import loadmat
    from spike_times_class import *
    from copy import deepcopy
    dat = loadmat('/Volumes/server/Data/Monkey2_newzdrive/Schro/Utah Array/Feb 20 2018/neural data/Pre-processing X E/m53s41.mat')
    # print(dat.keys())
    behav_stat_keys = 'behv_stats'
    lfps_key = 'lfps'
    units_key = 'units'
    behav_dat_key = 'trials_behv'

    beh_all = behavior_experiment(dat,behav_dat_key,behav_stat_keys)
    info = load_trial_types(dat[behav_stat_keys].flatten(),dat[behav_dat_key].flatten())
    # beh_stat = dat[behav_stat_keys].flatten()
    # trial_type = load_trial_types(beh_stat)
    # idxOther = trial_type.get_all(False)
    # idxUnclassRaw = trial_type.get_rewarded(-1)
    # idxUncDensity = trial_type.get_density(0.0001)
    units = dat[units_key].flatten()
    spk = spike_counts(dat,units_key)
    spk.bin_spikes(beh_all.time_stamps,0,beh_all.events.t_stop)
    histVec = deepcopy(spk.binned_spikes)
    select = np.array([7,8,17])
    spk.bin_spikes(beh_all.time_stamps,0,beh_all.events.t_stop,select)

    print(np.prod(histVec[0,17] == spk.binned_spikes[0,2]))
    # trials_behv = dat['trials_behv'].flatten()[1]
    # tb1 = trials_behv[1][0]['yre'].all().flatten()
