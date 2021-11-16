import numpy as np
import matplotlib.pylab as plt

class spike_counts(object):
    def __init__(self,dat,units_key, time_aligned_to_beh=True):
        units = dat[units_key].flatten()

        # flag to say if spike times are already aligned with the behav data
        # for now I don't have the raw data so everything is aligned
        # must predispose the class to the possibility of not being aligned
        # a structure above the spike counts should handle the possible misalignment
        # and than move the flag from False to True
        self.time_aligned_to_beh = time_aligned_to_beh

        # extract info about the cell channel and electrode from which the unit has been recorded
        self.brain_area = self.unpack_struct_N_times_1(units,'brain_area')
        self.channel_id = self.unpack_struct_N_times_1(units,'channel_id')
        self.electrode_id = self.unpack_struct_N_times_1(units, 'electrode_id')
        # single identifier for the unit
        self.cluster_id = self.unpack_struct_N_times_1(units, 'cluster_id')

        # spike stats (average waveform and width)
        self.waveform = self.unpack_struct_N_times_1(units, 'spkwf')
        self.amplitude_wf = spike_amplitude(self.waveform)
        self.spike_width = self.unpack_struct_N_times_1(units, 'spkwidth')

        # single or multi-unit label
        self.unit_type = self.unpack_struct_N_times_1(units,'type')

        self.num_units = self.cluster_id.shape[0]
        # extract the spike times and save it in a 2D array (units x trials)
        self.set_spike_array(units, 'trials', 'tspk')

        # extract the quality metrics indices
        self.uQ = self.unpack_struct_N_times_1(units,'uQ')
        self.isiV = self.unpack_struct_N_times_1(units,'isiV')
        self.cR = self.unpack_struct_N_times_1(units,'cR')


        self.n_trials = self.spike_times.shape[1]

        # bin spikes set to none
        self.binned_spikes = None

    def unpack_struct_N_times_1(self,units,key):
        length = units[key].shape[0]
        datatype = units[key][0].dtype.type
        if np.dtype(datatype).kind == 'i' or np.dtype(datatype).kind == 'u':
            datatype = np.int64

        if len(units[key])>1:
            shape = units[key][1].shape
        else:
            shape = units[key][0].shape

        if len(shape) == 2:

            if shape[1] != 1 and shape[0] != 1:
                unpacked = self.unpack_struct_N_times_M(units,key)
            else:
                if shape[0] == 1 and shape[1] != 1:
                    N = shape[1]
                else:
                    N = shape[0]
                unpacked = np.zeros((length, N), dtype=datatype)
                for k in range(length):
                    # print(k)
                    try:
                        unpacked[k,:] = units[key][k].reshape(N,)
                    except ValueError:
                        print('UNIT %d - could not assign %s'%(k,key))
                        try:
                            try:
                                unpacked = np.array(unpacked,dtype=float)
                                unpacked[k, :] = units[key][k].reshape(N,)
                            except:
                                raise ValueError
                        except ValueError:
                            unpacked[k, :] = np.nan
                        # raise ValueError('ValueError: cannot reshape array of size 0 into shape (1,)')

        # case of a 1 d array of values
        elif len(shape) == 1 and datatype != np.str_:
            N = shape[0]
            unpacked = np.zeros((length, N), dtype=datatype)
            for k in range(length):
                unpacked[k] = units[key][k]

        # unpack an array of strings
        elif len(shape) == 1 and datatype == np.str_:
            set_dtype = 'U50'
            unpacked = np.zeros((length,), dtype=set_dtype)
            for k in range(length):
                unpacked[k] = units[key][k][0]
        else:
            raise ValueError("this function can't unpack tensors")


        unpacked = np.squeeze(unpacked)
        return unpacked

    def unpack_struct_N_times_M(self,units,key):
        length = units[key].shape[0]

        datatype = units[key][0].dtype.type

        shape = units[key][0].shape

        if len(shape) == 2:
            N = shape[0]
            M = shape[1]
            unpacked = np.zeros((length, ), dtype=object)
            for k in range(length):

                unpacked[k] = units[key][k]

        else:
            raise ValueError('wrong number of dimensions ')


        unpacked = np.squeeze(unpacked)
        return unpacked

    def set_spike_array(self,units,key,field):
        ntrials = units[key][0].shape[1]
        self.spike_times = np.zeros((self.num_units,ntrials),dtype=object)
        for unt in range(self.num_units):
            for tr in range(ntrials):
                self.spike_times[unt,tr] = units[key][unt][0,tr][field].flatten()

    def bin_spikes(self,edges,t_start=None,t_stop=None,select=None,cutFirstLastTP=True):

        # extract freq in ms, use round to get the theoretical acquisition freq.
        dt = round(edges[0][1] - edges[0][0],3)

        if not select is None:
            spks_tmp = self.spike_times[:,select]
            edges_sel = np.arange(self.n_trials)[select]
        else:
            spks_tmp = self.spike_times
            edges_sel = np.arange(self.n_trials)

        self.binned_spikes = np.zeros((self.num_units,spks_tmp.shape[1]),dtype=object)
        bin_list = {}
        for unt in range(self.num_units):
            for idx in range(spks_tmp.shape[1]):
                tr = edges_sel[idx]
                # if tr == 1231:
                #     jumanji=10
                if cutFirstLastTP:
                    edges_tr = np.array(edges[tr][1:-1],dtype=float)
                else:
                    edges_tr = np.array(edges[tr],dtype=float)
                # if t_start is set to None it means that edges must not be cut
                if t_start is None:
                    bins = np.hstack((edges_tr,np.inf))
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
                    # this way we have the last bin that contains every spikes happened after t1
                    # edges_tr = np.hstack((edges[tr],edges[tr][-1] + dt))

                    # keep only the bins in the desired range
                    bins = edges_tr[(edges_tr >= t0) * (edges_tr <= t1)]

                    # check for empty bins
                    if len(bins) == 0:
                        self.binned_spikes[unt, idx] = np.array([])
                        if unt == 0:
                            bin_list[tr]  = np.array([])
                        continue
                    # add the last bin
                    bins = np.hstack((bins, bins[-1] + dt))
                self.binned_spikes[unt,idx],_ = np.histogram(self.spike_times[unt,tr],bins=bins)
                if unt == 0:
                    # might want to change to tr instead of idx...check this out
                    bin_list[tr] = 0.5*(bins[1:] + bins[:-1])
        return bin_list
                
    def select_spike_times(self,t_start,t_stop,select=None):

        # extract freq in ms, use round to get the theoretical acquisition freq.

        if not select is None:
            spks_tmp = self.spike_times[:,select]
            edges_sel = np.arange(self.n_trials)[select]
        else:
            spks_tmp = self.spike_times
            edges_sel = np.arange(self.n_trials)

        cut_spikes = np.zeros((self.num_units,spks_tmp.shape[1]),dtype=object)

        for unt in range(self.num_units):
            for idx in range(spks_tmp.shape[1]):

                tr = edges_sel[idx]
                
            
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
                sele_times = (self.spike_times[unt,tr] > t0) & (self.spike_times[unt,tr] < t1)
                cut_spikes[unt,idx] = self.spike_times[unt,tr][sele_times]
        return cut_spikes
                



    def clear_binned_spikes(self):
        self.binned_spikes = None


def spike_amplitude(waveform, plt_first=False):
    x = np.arange(waveform.shape[1])
    mn = np.argmin(waveform,axis=1)
    mx = np.argmax(waveform,axis=1)
    amplitude = waveform[:,mx] - waveform[:,mn]
    if plt_first:
        plt.plot(x, waveform[0,:])
        mn0 = mn[0]
        mx0 = mx[0]
        plt.plot(x[[mn0, mx0]], waveform[0,[mn0, mx0]], 'o')
    return amplitude

if __name__ == '__main__':

    from copy import deepcopy
    from scipy.io import loadmat
    from behav_class import *

    dat = loadmat('/Volumes/server/Data/Monkey2_newzdrive/Schro/Sim_recordings/Aug 10 2018/neural data/Pre-processing X E/m53s111.mat')
    print(dat.keys())
    behav_stat_keys = 'behv_stats'
    lfps_key = 'lfps'
    units_key = 'units'
    behav_dat_key = 'trials_behv'

    beh_all = behavior_experiment(dat,behav_dat_key,behav_stat_keys)
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

    print('select test: ',np.prod(histVec[0,17] == spk.binned_spikes[0,2]))
