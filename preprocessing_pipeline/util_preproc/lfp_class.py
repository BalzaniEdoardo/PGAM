import numpy as np
from scipy.signal import hilbert


class lfp_class(object):
    def __init__(self,dat,lfp_key, binned=True,lfp_alpha=None,lfp_beta=None,lfp_theta=None,compute_phase=True):
        lfps = dat[lfp_key].flatten()

        # extract info about the cell channel and electrode from which the unit has been recorded
        self.channel_id = self.unpack_struct_N_times_1(lfps,'channel_id')
        self.channel_id = self.channel_id.flatten()

        self.electrode_id = self.unpack_struct_N_times_1(lfps, 'electrode_id')
        self.electrode_id = self.electrode_id.flatten()
        self.brain_area = self.unpack_struct_N_times_1(lfps, 'brain_area')
        self.compute_phase = compute_phase
        self.n_channels =lfps.shape[0]

        if not binned:
            self.lfp = None
            self.n_trials = None
        else:
            self.extract_lfp(lfps)
            self.n_trials = self.lfp.shape[1]

        if not lfp_beta is None:
            self.lfp_beta = self.extract_lfp_x_band(lfp_beta,'lfp_beta')

        if not lfp_alpha is None:
            self.lfp_alpha = self.extract_lfp_x_band(lfp_alpha,'lfp_alpha')


        if not lfp_theta is None:
            self.lfp_theta = self.extract_lfp_x_band(lfp_theta,'lfp_theta')



    def extract_lfp_x_band(self,lfp_band,band_label):
        num_trials = lfp_band[0,0]['trials'].shape[1]
        lfp_band_all = np.zeros((self.n_channels, num_trials), dtype=object)
        for chan in range(self.n_channels):
            for tr in range(num_trials):
                if self.compute_phase:
                    lfp_band_all[chan, tr] = np.array(np.angle(lfp_band[0, chan]['trials'][0, tr][band_label].flatten()),dtype=np.float32)
                else:
                    lfp_band_all[chan, tr] = lfp_band[0, chan]['trials'][0, tr][band_label].flatten()
        return lfp_band_all


    def extract_lfp(self,lfps):
        num_trials = lfps[0]['trials'].shape[1]
        self.lfp = np.zeros((self.n_channels,num_trials),dtype=object)
        for chan in range(self.n_channels):
            for tr in range(num_trials):
                self.lfp[chan,tr] = lfps[chan]['trials'][0,tr]['lfp'].flatten()

    def unpack_struct_N_times_1(self,lfps,key):
        length = lfps[key].shape[0]
        datatype = lfps[key][0].dtype.type

        shape = lfps[key][0].shape

        if len(shape) == 2:
            N = shape[0]
            if shape[1] != 1:
                raise ValueError('must unpack an array of N x 1 elements')
            unpacked = np.zeros((length, N), dtype=datatype)
            for k in range(length):
                unpacked[k,:] = lfps[key][k].reshape(N,)

        # case of a 1 d array of values
        elif len(shape) == 1 and datatype != np.str_:
            N = shape[0]
            unpacked = np.zeros((length, N), dtype=datatype)
            for k in range(length):
                unpacked[k] = lfps[key][k]

        # unpack an array of strings
        elif len(shape) == 1 and datatype == np.str_:
            set_dtype = 'U50'
            unpacked = np.zeros((length,), dtype=set_dtype)
            for k in range(length):
                unpacked[k] = lfps[key][k][0]

        else:
            raise ValueError("this function can't unpack tensors")
        return unpacked

    def bin_lfp(self,lfps,time_bins):
        print('Method not implemented yet!')
        return

    def extract_phase(self,filter,channel_id_x_unit,unit_brain_area):
        if self.lfp is None:
            return

        num_trials = np.sum(filter)
        num_units = channel_id_x_unit.shape[0]
        phase = np.zeros((num_units,num_trials),dtype=object)

        keep = np.arange(self.n_trials)[filter]

        for chan_un in range(num_units):
            ba_unit = unit_brain_area[chan_un]
            select = (self.channel_id == channel_id_x_unit[chan_un]) * (self.brain_area == ba_unit)
            assert(select.sum()==1)
            use_idx = np.where(select)[0][0]
            for idx in range(num_trials):
                tr = keep[idx]

                phase[chan_un,idx] = np.angle(hilbert(self.lfp[use_idx,tr]))

        return phase

    def extract_phase_x_unit(self,phase,filter,channel_id_x_unit,unit_brain_area):
        if self.lfp is None:
            return

        num_trials = np.sum(filter)
        num_units = channel_id_x_unit.shape[0]
        phase_unit = np.zeros((num_units,num_trials),dtype=object)

        keep = np.arange(self.n_trials)[filter]

        for chan_un in range(num_units):
            ba_unit = unit_brain_area[chan_un]
            select = (self.channel_id == channel_id_x_unit[chan_un]) * (self.brain_area == ba_unit)
            assert(select.sum()==1)
            use_idx = np.where(select)[0][0]
            for idx in range(num_trials):
                tr = keep[idx]

                phase_unit[chan_un,idx] = phase[use_idx,tr]#np.angle(hilbert(self.lfp[use_idx,tr]))

        return phase_unit

    def extract_phase_from_band(self,lfp_band,filter,channel_id_x_unit,unit_brain_area):


        num_trials = np.sum(filter)
        num_units = channel_id_x_unit.shape[0]
        phase = np.zeros((num_units,num_trials),dtype=object)

        keep = np.arange(self.n_trials)[filter]

        for chan_un in range(num_units):
            ba_unit = unit_brain_area[chan_un]
            select = (self.channel_id == channel_id_x_unit[chan_un]) * (self.brain_area == ba_unit)
            assert(select.sum()==1)
            use_idx = np.where(select)[0][0]
            for idx in range(num_trials):
                tr = keep[idx]

                phase[chan_un,idx] = np.angle(lfp_band[use_idx,tr])
        return phase


    def extract_phase_and_amplitude(self,filter,channel_id_x_unit,unit_brain_area):
        if self.lfp is None:
            return

        num_trials = np.sum(filter)
        num_units = channel_id_x_unit.shape[0]
        phase = np.zeros((num_units,num_trials),dtype=object)
        amplitude = np.zeros(phase.shape,dtype=object)
        keep = np.arange(self.n_trials)[filter]

        for chan_un in range(num_units):
            ba_unit = unit_brain_area[chan_un]
            select = (self.channel_id == channel_id_x_unit[chan_un]) * (self.brain_area == ba_unit)
            assert (select.sum() == 1)
            use_idx = np.where(select)[0][0]
            for idx in range(num_trials):
                tr = keep[idx]
                hilb_trans = hilbert(self.lfp[use_idx, tr])
                phase[chan_un,idx] = np.angle(hilb_trans)
                amplitude[chan_un,idx] = np.abs(hilb_trans)
        return phase,amplitude

    def cut_phase(self,phase, edges, t_start=None, t_stop=None, select=None, idx0=None, idx1=None):
        if select is None:
            cut_phase = np.zeros(phase.shape, dtype=object)
        else:
            cut_phase = np.zeros((phase.shape[0],np.sum(select)), dtype=object)
        for unit in range(phase.shape[0]):
            cut_phase[unit,:] = self.cut_continuous(phase[unit,:], edges, t_start=t_start,
                                        t_stop=t_stop, select=select, idx0=idx0, idx1=idx1)
        # np.save('/Users/edoardo/Work/Code/Angelaki-Savin/Kaushik/fire_fly_variable_selection/phase_stack.npy',cut_phase)
        return cut_phase


    def cut_continuous(self, continuous, edges, t_start=None, t_stop=None, select=None,idx0=None,idx1=None):

        if not select is None:
            edges_sel = np.arange(self.n_trials)[select]
        else:
            edges_sel = np.arange(self.n_trials)

        if idx0 is None:
            idx0 = 0
        bin_continous = np.zeros(edges_sel.shape[0], dtype=object)
        # get the event time in the selected trial
        ii = 0
        i_idx1 = idx1
        for tr in edges_sel:
            if idx1 is None:
                i_idx1 = continuous[tr].shape[0]

            continuous_tr = continuous[tr][idx0:i_idx1]

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


    # def cut_continuous(self, continuous, edges, t_start=None, t_stop=None, select=None, idx0=None, idx1=None):
    #
    #
    #     if not select is None:
    #         edges_sel = np.arange(self.n_trials)[select]
    #     else:
    #         edges_sel = np.arange(self.n_trials)
    #
    #     bin_continous = np.zeros(edges_sel.shape[0],dtype=object)
    #     if idx0 is None:
    #         idx0 = 0
    #
    #     # get the event time in the selected trial
    #     ii = 0
    #     i_idx1 = idx1
    #     for tr in edges_sel:
    #         if idx1 is None:
    #             i_idx1 = continuous[tr].shape[0]
    #         continuous_tr = continuous[tr][idx0:idx1]
    #
    #         edge_tr = edges[tr]
    #
    #         # if t_start is set to None it means that edges must not be cut
    #         if t_start is None:
    #             t0 = -np.inf
    #             t1 = np.inf
    #         else:
    #             # if start is a scalar, always takes times greater than t_start
    #             if np.isscalar(t_start):
    #                 t0 = t_start
    #             # otherwise consider as start the time t0 indicated by the dictionary with t_starts
    #             else:
    #                 t0 = t_start[tr]
    #             # same for t_stop
    #             if np.isscalar(t_stop):
    #                 t1 = t_stop
    #             else:
    #                 t1 = float(t_stop[tr])
    #
    #         bin_continous[ii] = continuous_tr[(edge_tr > t0) * (edge_tr < t1)]
    #         ii += 1
    #     return bin_continous

if __name__ == '__main__':

    from copy import deepcopy
    from scipy.io import loadmat
    from behav_class import *
    from spike_times_class import *
    import h5py
    dat = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/m53s127_new.mat')
    lfp_beta = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/lfp_beta_m53s127.mat')
    lfp_alpha = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/lfp_alpha_m53s127.mat')
    lfp_theta = loadmat('/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/MST/lfp_theta_m53s127.mat')

    # dat = loadmat('/Volumes/WD Edo/firefly_analysis/DATASET/PPC+PFC+MST/m53s84.mat')
    print(dat.keys())
    behav_stat_keys = 'behv_stats'
    lfps_key = 'lfps'
    units_key = 'units'
    behav_dat_key = 'trials_behv'

    beh_stat = dat[behav_stat_keys].flatten()
    units = dat[units_key].flatten()
    spk = spike_counts(dat, units_key)


    info = load_trial_types(beh_stat)
    lfp = lfp_class(dat,lfps_key,lfp_beta=lfp_beta['lfp_beta'],lfp_alpha=lfp_alpha['lfp_alpha'],lfp_theta=None,is_phase=True)
    filter = info.get_all(True)
    phase = lfp.extract_phase_from_band(lfp.lfp_beta,filter,spk.channel_id,spk.brain_area)

    # print(phase[0,0][:10])
