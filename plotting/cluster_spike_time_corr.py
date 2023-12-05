#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:19:10 2020

@author: edoardo
"""


import numpy as np
import scipy.stats as sts
import sys,os,re
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
if 'eb162' in main_dir:
    main_dir = '/scratch/eb162/GAM_Repo'
print(main_dir)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline','util_preproc'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from path_class import get_paths_class
from scipy.io import loadmat
from data_handler import *
# from extract_presence_rate import *
from scipy.io import loadmat
from path_class import get_paths_class
path_user = get_paths_class()
import matplotlib.pylab as plt
from GAM_library import *
import pdb



load = lambda folder,filename: np.load(os.path.join(folder, filename))
def read_cluster_group_tsv(filename):

    """
    Reads a tab-separated cluster_group.tsv file from disk

    Inputs:
    -------
    filename : String
        Full path of file

    Outputs:
    --------
    IDs : list
        List of cluster IDs
    quality : list
        Quality ratings for each unit (same size as IDs)

    """

    info = np.genfromtxt(filename, dtype='str')
    cluster_ids = info[1:,0].astype('int')
    cluster_quality = info[1:,1]

    return cluster_ids, cluster_quality


def load_kilosort_data(folder, 
                       sample_rate = None, 
                       convert_to_seconds = True, 
                       use_master_clock = False, 
                       include_pcs = False,
                       template_zero_padding= 21):

    """
    Loads Kilosort output files from a directory

    Inputs:
    -------
    folder : String
        Location of Kilosort output directory
    sample_rate : float (optional)
        AP band sample rate in Hz
    convert_to_seconds : bool (optional)
        Flags whether to return spike times in seconds (requires sample_rate to be set)
    use_master_clock : bool (optional)
        Flags whether to load spike times that have been converted to the master clock timebase
    include_pcs : bool (optional)
        Flags whether to load spike principal components (large file)
    template_zero_padding : int (default = 21)
        Number of zeros added to the beginning of each template

    Outputs:
    --------
    spike_times : numpy.ndarray (N x 0)
        Times for N spikes
    spike_clusters : numpy.ndarray (N x 0)
        Cluster IDs for N spikes
    spike_templates : numpy.ndarray (N x 0)
        Template IDs for N spikes
    amplitudes : numpy.ndarray (N x 0)
        Amplitudes for N spikes
    unwhitened_temps : numpy.ndarray (M x samples x channels) 
        Templates for M units
    channel_map : numpy.ndarray
        Channels from original data file used for sorting
    cluster_ids : Python list
        Cluster IDs for M units
    cluster_quality : Python list
        Quality ratings from cluster_group.tsv file
    pc_features (optinal) : numpy.ndarray (N x channels x num_PCs)
        PC features for each spike
    pc_feature_ind (optional) : numpy.ndarray (M x channels)
        Channels used for PC calculation for each unit

    """

    if use_master_clock:
        spike_times = load(folder,'spike_times_master_clock.npy')
    else:
        spike_times = load(folder,'spike_times.npy')
        
    spike_clusters = load(folder,'spike_clusters.npy')
    spike_templates = load(folder, 'spike_templates.npy')
    amplitudes = load(folder,'amplitudes.npy')
    templates = load(folder,'templates.npy')
    unwhitening_mat = load(folder,'whitening_mat_inv.npy')
    channel_map = load(folder, 'channel_map.npy')

    if include_pcs:
        pc_features = load(folder, 'pc_features.npy')
        pc_feature_ind = load(folder, 'pc_feature_ind.npy') 
                
    templates = templates[:,template_zero_padding:,:] # remove zeros
    spike_clusters = np.squeeze(spike_clusters) # fix dimensions
    spike_times = np.squeeze(spike_times)# fix dimensions

    if convert_to_seconds and sample_rate is not None:
       spike_times = spike_times / sample_rate 
                    
    unwhitened_temps = np.zeros((templates.shape))
    
    for temp_idx in range(templates.shape[0]):
        
        unwhitened_temps[temp_idx,:,:] = np.dot(np.ascontiguousarray(templates[temp_idx,:,:]),np.ascontiguousarray(unwhitening_mat))
                    
    try:
        cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
    except OSError:
        cluster_ids = np.unique(spike_clusters)
        cluster_quality = ['unsorted'] * cluster_ids.size

    if not include_pcs:
        return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, cluster_ids, cluster_quality
    else:
        return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind

def merge_spike_times_x_electrode(exp_data, spike_times):
    elect_id = exp_data.spikes.electrode_id
    brain_area = exp_data.spikes.brain_area
    
    id_mix = np.zeros(brain_area.shape[0],dtype='U20')
    for kk in range(id_mix.shape[0]):
        id_mix[kk] = brain_area[kk] + '_%d'%elect_id[kk]
    
    unq_elect_num = np.unique(id_mix).shape[0]
    spike_times_x_electrode = np.zeros((unq_elect_num,spike_times.shape[1]),dtype=object)
    
    brain_ar = []
    elect_id = []
    ccele = 0
    for eleid in np.unique(id_mix):
        elect_id = np.hstack((elect_id,[int(eleid.split('_')[1])]))
        brain_ar = np.hstack((brain_ar,[eleid.split('_')[0]]))
        idx = np.where(id_mix == eleid)[0]
        for tr in range(spike_times.shape[1]):
        
            spk_time = np.array([])
            for unt in idx:
                spk_time = np.hstack((spk_time, spike_times[unt,tr]))
            spk_time.sort()
            spike_times_x_electrode[ccele, tr] = spk_time
        ccele += 1
    return spike_times_x_electrode, elect_id, brain_ar
    
    
    
def binning(tspk1, shift=0.002, numshift=100, mnTime=None, mxTime=None):
    """
        Description
        ===========
            This function is computing the CCG between 2 spike trains using
            pearson correlation of discretized spike bins. Spike times are in
            seconds, this function discretize the range of spike times by
            bins of duration "shift", and computes the pearson correlaiton
            between 1 spike train and the other shifted of multiples of shift
            from "-numshift * shit" to  "numshift * shift".
        Input
        =====
            - tspike1/2 = list or np.array of sipke times in sec
            - shift = float bin size in sec used for the analysis
            - numshift = number of lags that are used for computing ccg
        Output
        ======
            - corr = np.array, containing the values of CCG
            - edges = np.array containing the lags in sec corresponding to the
            CCG
    """
    # Transform into array eventual lists
    tspk1 = np.array(tspk1)

    # compute how many bins of shift size I must create ti duscretize spks
    nBin = int(np.ceil((mxTime - mnTime) / shift))

    # create two containers for discretized spikes
    discrtSpk1 = np.zeros(nBin)

    # compute the interval in which each spikes falls, set 1 if there's at
    # least one spike in the interval
    discrInd = np.array(np.floor((tspk1 - mnTime) / shift), dtype=int)
    for i in discrInd:
        if i == nBin:
            continue
        discrtSpk1[i] = discrtSpk1[i] + 1

    return discrtSpk1


def normalizeCountsCorrelate(counts, binSpk1, binSpk2):
    N = max(binSpk1.shape[0], binSpk2.shape[0])
    posSpk1 = binSpk1 > 0
    posSpk2 = binSpk2 > 0
    Nx = np.sum(binSpk1[posSpk1])
    Ny = np.sum(binSpk2[posSpk2])

    ii = np.dot(binSpk1[posSpk1], binSpk1[posSpk1])
    jj = np.dot(binSpk2[posSpk2], binSpk2[posSpk2])
    rho_xy = (counts - Nx * Ny / N) / \
             np.sqrt((ii - Nx ** 2. / N) * (jj - Ny ** 2. / N))

    return rho_xy


def ccgElephantEquivalent(binSpk1, binSpk2, window='full', cross_corr_coef=True, binary=False, border_correct=True):
    if binary:
        binSpk1 = np.array(binSpk1 > 0, dtype=int)
        binSpk2 = np.array(binSpk2 > 0, dtype=int)
    counts = np.correlate(binSpk2, binSpk1, mode=window)
    if border_correct:
        r = binSpk2.shape[0] - 1
        l = - binSpk1.shape[0] + 1
        max_num_bins = max(binSpk1.shape[0], binSpk2.shape[0])
        correction = float(max_num_bins + 1) / np.array(
                                                        max_num_bins + 1 - abs(
                                                                               np.arange(l, r + 1)), float)
        counts = counts * correction
    if cross_corr_coef:
        counts = normalizeCountsCorrelate(counts, binSpk1, binSpk2)

    return counts


def compute_ccg_tensor(spikes_cut,session,choer_folder,electrode_id, brain_area,
                       time_delta=0.002,window='full', cross_corr_coef=False, binary=False, border_correct=True):
    
    lfp_cho_mat = loadmat(os.path.join(choer_folder,'LFP_coherence_%s.mat'%session))


    dtype_dict = {'names':('session','brain_area','electrode_id'),
              'formats':('U30','U30',int)}
    lead_electrode_missing = np.zeros(0,dtype=dtype_dict)
    for area in lfp_cho_mat['lead_electrode']:
        ba = area[0][0]
        ele = area[1][0,0]
        sele = (electrode_id == ele) & (brain_area == ba)
        if not any(sele):
            tmp = np.zeros(1,dtype=dtype_dict)
            tmp['session'] = session
            tmp['electrode_id'] = ele
            tmp['brain_area'] = ba
            lead_electrode_missing = np.hstack((lead_electrode_missing,tmp))
            continue
        if sele.sum() != 1:
            raise ValueError('Multiple electrodes with the same id in the same area!')
        reference_idx = np.where(sele)[0][0]
        iterate = list(range(0,reference_idx)) + list(range(reference_idx+1,spikes_cut.shape[0]))
        # get the longest trial extremes
        tmax = -np.inf
        tmin = +np.inf
        for tr in range(spikes_cut.shape[1]):
            for k in range(spikes_cut.shape[0]):
                tspk2 = spikes_cut[k, tr]
                if tspk2.shape[0]:
                    tmax = max(tmax, tspk2[-1])
                    tmin = min(tmin, tspk2[0])
        
        # compute the num of time bins
        nBin_max = int(np.ceil((tmax - tmin) / time_delta))
        ccfedge_max = np.arange(- nBin_max + 1, nBin_max) 
        matrix_ccg = np.zeros((len(iterate), spikes_cut.shape[1], ccfedge_max.shape[0]))
            
        # cycle over spikes and trials
        for tr in range(spikes_cut.shape[1]):
            tspk1 = spikes_cut[reference_idx,tr]
            
            # select borders
            if tspk1.shape[0] == 0:
                continue
            tmax = tspk1[-1]
            tmin = tspk1[0]
            
           
            for k in iterate:
                tspk2 = spikes_cut[k, tr]
                if tspk2.shape[0]:
                    tmax = max(tmax, tspk2[-1])
                    tmin = min(tmin, tspk2[0])
            
            # compute the num of time bins
            nBin = int(np.ceil((tmax - tmin) / time_delta))
            ccfedge = np.arange(- nBin + 1, nBin) 
            idx_sel = (ccfedge_max >= ccfedge[0]) & (ccfedge_max <= ccfedge[-1])
            
            # bin and compute corr:
            bin1 = binning(tspk1, shift=time_delta, numshift=300, mnTime=tmin, mxTime=tmax)
            ccit = 0
            for k in iterate:
                tspk2 = spikes_cut[k, tr]
                bin2 =  binning(tspk2, shift=time_delta, numshift=300, mnTime=tmin, mxTime=tmax)
                matrix_ccg[ccit,tr,idx_sel] = ccgElephantEquivalent(bin1, bin2, window=window, cross_corr_coef=cross_corr_coef, binary=binary, border_correct=border_correct)
                ccit += 1
    return matrix_ccg, ccfedge_max*time_delta, lead_electrode_missing


def asses_mod_significance(ccg, nrm, consecEp):
    
    idx_first = np.where(np.diff(np.array(ccg[plot_sele] > nrm.ppf(0.99),dtype=int))==1)[0]
    idx_last = np.where(np.diff(np.array(ccg[plot_sele] > nrm.ppf(0.99),dtype=int))==-1)[0]

    if len(idx_last) == 0 and len(idx_first) == 1:
        idx_last = np.array([ccg.shape[0]])
        
    elif len(idx_last) != 0 and len(idx_first) != 0:
        idx_first = idx_first[idx_first < idx_last[-1]]
        idx_last = idx_last[idx_last  > idx_first[0]]
    
    if len(idx_last) != len(idx_first):
        print('something in the logic is odd')
        return False
    
    is_sign = any(idx_last - idx_first >= consecEp)
    
    
    idx_first = np.where(np.diff(np.array(ccg[plot_sele] < nrm.ppf(0.01),dtype=int))==1)[0]
    idx_last = np.where(np.diff(np.array(ccg[plot_sele] < nrm.ppf(0.01),dtype=int))==-1)[0]

    if len(idx_last) == 0 and len(idx_first) == 1:
        idx_last = np.array([ccg.shape[0]])
    
    if len(idx_last) == 1 and len(idx_first) == 0:
        idx_last = np.array([])
        
    elif len(idx_last) != 0 and len(idx_first) != 0:
        idx_first = idx_first[idx_first < idx_last[-1]]
        idx_last = idx_last[idx_last > idx_first[0]]
    
    if len(idx_last) != len(idx_first):
        print('something in the logic is odd')
        return(False)
        
    
    is_sign = any(idx_last - idx_first >= consecEp) or is_sign
    
    return is_sign
        

        
    
    


print('###################################################################')
print('#\tThis code assumes that PPC & PFC are recorded from Utah array\n#\tand MST from lineararray')
print('###################################################################')

user_paths = get_paths_class()
# list of session to be concatenated


concat_list = ['m53s98']

save = True
send = True
 
# destination folder
#sv_folder = '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/PPC+PFC+MST/'
# sv_folder = '/Users/edoardo/Work/Code/GAM_code/fitting/'

# path to files

# path to preproc mat files
#base_file = '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/PPC+PFC+MST/'
base_file =  '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/PPC+MST/'#user_paths.get_path('local_concat','m44s174')
if not os.path.exists(base_file):
    base_file = user_paths.get_path('data_hpc')

#result_fld = '/Volumes/WD Edo/firefly_analysis/LFP_band/results_radTarg/'

# list of session in which forcing the use of left eye posiiton
use_left_eye = ['m53s83','m53s84','m53s86','m53s90','m53s92','m53s133','m53s134','m53s35']


# keys in the mat file generated by the preprocessing script of  K.
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'

nsec = 1.


# presence rate params
occupancy_bin_sec = 60 # at least one spike per min
occupancy_rate_th = 0.1 #hz

linearprobe_sampling_fq = 20000
utah_array_sampling_fq = 30000

time_delta = 0.002

dtype_dict = {'names':('session','brain_area','electrode_id'),
              'formats':('U30','U30',int)}
lead_electrode_missing = np.zeros(0,dtype=dtype_dict)


if os.path.exists('/scratch/eb162/'):
    import pandas as pd
    
    JOB = int(sys.argv[1]) - 1
    lst_dir = os.listdir('/scratch/eb162/sorted')
    sess_id = []
    monk_id = []
    pattern_fh = '^m\d+s\d+$'
    for name in lst_dir:
        if re.match(pattern_fh,name):
            if not os.path.exists(os.path.join(base_file,name+'.mat')):
                continue
            sess_id += [int(name.split('s')[1])]
            monk_id += [ int(name.split('s')[0].split('m')[1])]
    sess_array = np.zeros((len(monk_id)),dtype={'names':('monk','sess'),'formats':(int,int)})
    sess_array['monk'] = monk_id
    sess_array['sess'] = sess_id
    sess_array = np.sort(sess_array,order=['monk','sess'])
    session = 'm%ds%d'%(sess_array['monk'][JOB],sess_array['sess'][JOB])
    print(pd.DataFrame(sess_array))
    
    
else:
    session = 'm44s182'


if session in use_left_eye:
    use_eye = 'left'
else:
    use_eye = 'right'




print('loading session %s...'%session)
pre_trial_dur = 0.2
post_trial_dur = 0.2

#pdb.set_trace()
dat = loadmat(os.path.join(base_file,'%s.mat'%(session)))


#lfp_beta = loadmat(os.path.join(base_file,'lfp_beta_%s.mat'%session))
#lfp_alpha = loadmat(os.path.join(base_file,'lfp_alpha_%s.mat'%session))
#lfp_theta = loadmat(os.path.join(base_file,'lfp_theta_%s.mat'%session))

# if 'is_phase' in lfp_beta.keys():
#     is_phase = lfp_beta['is_phase'][0,0]
# else:
#     is_phase = False

exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur,
                        post_trial_dur=post_trial_dur,
                        lfp_beta=None, lfp_alpha=None,lfp_theta=None,
                        use_eye=use_eye)

exp_data.set_filters('all', True)
# impose all replay trials
exp_data.filter = exp_data.filter + exp_data.info.get_replay(0,skip_not_ok=False)
# savemat('lfp_raw_%s.mat'%session,{'lfp':exp_data.lfp.lfp})

t_targ = dict_to_vec(exp_data.behav.events.t_targ)
t_move = dict_to_vec(exp_data.behav.events.t_move)

t_start = np.min(np.vstack((t_move, t_targ)), axis=0) - pre_trial_dur
t_stop = dict_to_vec(exp_data.behav.events.t_end) + post_trial_dur

spikes_cut = exp_data.spikes.select_spike_times(t_start,t_stop,select=exp_data.filter)
spikes_cut,electrode_id,brain_area = merge_spike_times_x_electrode(exp_data, spikes_cut)


# first extract utah array
sorted_fold = path_user.get_path('server_data',session)
if not os.path.exists(sorted_fold):
    sorted_fold = '/scratch/eb162/sorted/%s'%session
spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality = \
    load_kilosort_data(sorted_fold, \
                    utah_array_sampling_fq, \
                    use_master_clock=False,
                    include_pcs=False)
        

    

max_time = np.max(spike_times)
min_time = np.min(spike_times)

array_ba = ['PPC','PFC']

unq_cl_lab = np.unique(spike_clusters)
iii = 0
keep_cl= np.ones(unq_cl_lab.shape[0],dtype=bool)
for cl in unq_cl_lab:
    cond = (exp_data.spikes.cluster_id == cl) &\
        ((exp_data.spikes.brain_area == 'PPC') |
            (exp_data.spikes.brain_area == 'PFC'))
    if not any(cond):
        keep_cl[iii] = False
    iii+=1

unq_cl_lab = unq_cl_lab[keep_cl]
dict_dt = {
    'names':('session','reference electrode id', 'reference channel id', 'reference cluster id', 'referrence brain area',
             'other electrode id', 'other channel id', 'other cluster id','other brain area',
             'is significant'),
    'formats':('U30',int,int,int,'U30',int,int,int,'U30',bool)
    }

ccg_res = np.zeros((unq_cl_lab.shape[0]*(unq_cl_lab.shape[0]+1)//2,2000))
info_pair = np.zeros(unq_cl_lab.shape[0]*(unq_cl_lab.shape[0]+1)//2, dtype=dict_dt)
cnt = 0
for jj in range(unq_cl_lab.shape[0]):
    clust_id_ref = unq_cl_lab[jj]
    iidx_ref = (exp_data.spikes.cluster_id == clust_id_ref) &\
        ((exp_data.spikes.brain_area == 'PPC') |
            (exp_data.spikes.brain_area == 'PFC'))
    
    tspk1 = spike_times[spike_clusters == clust_id_ref]
    
    
    # other spike clust id in the same area
    
    

    brain_area_ref = exp_data.spikes.brain_area[iidx_ref]
    channel_id_ref = exp_data.spikes.channel_id[iidx_ref]
    electrode_id_ref = exp_data.spikes.electrode_id[iidx_ref]

    for kk in range(jj,unq_cl_lab.shape[0]):
        # if kk != 4:
        #     continue
        cluster_id_other = unq_cl_lab[kk]
        iidx = (exp_data.spikes.cluster_id == cluster_id_other) &\
        ((exp_data.spikes.brain_area == 'PPC') |
            (exp_data.spikes.brain_area == 'PFC'))
        
        
        
        brain_area_other = exp_data.spikes.brain_area[iidx]
        channel_id_other = exp_data.spikes.channel_id[iidx]
        electrode_id_other = exp_data.spikes.electrode_id[iidx]
        
        for numbins in (np.array([2000]) + 1):
            
            # extract histogram
            edges = np.linspace(-nsec, nsec, numbins)
            # sele another electrode
            tspk2 = spike_times[spike_clusters == cluster_id_other]
            
            # sele spikes after 1sec 
            tspk1 = tspk1[(tspk1>=nsec) & (tspk1<=spike_times[-1]-nsec)]
            hist_spks = np.zeros((edges.shape[0]-1,tspk1.shape[0]),dtype=float)
    
            k = 0
            for t0 in tspk1:
                sub_t2 = tspk2[(tspk2 >= t0-nsec) & (tspk2 < t0+nsec)] - t0
                hist_spks[:,k] = np.histogram(sub_t2,bins=edges)[0] 
                k += 1
           
            ccg=np.mean(hist_spks,axis=1)/(edges[1]-edges[0])
            
            ccg_res[cnt] = ccg
            
            if jj == kk:
                
                mu = ccg[edges[:-1]!=0].mean()
                std = ccg[edges[:-1]!=0].std()
            else:
                mu = ccg.mean()
                std = ccg.std()
            nrm = sts.norm(mu,std)
             # plt.subplot(1,1,cc+1)
            
            plot_sele = (edges[:-1] >= -0.1) * (edges[:-1] <= 0.1) 
            if kk ==jj:
                plot_sele[edges[:-1]==0]=False
            
            # plt.plot(edges[:-1][plot_sele],ccg[plot_sele],label='DT: %.2f ms'%((edges[1]-edges[0])*1000))
            # plt.plot(edges[:-1][plot_sele],[nrm.ppf(0.01)]*(plot_sele.sum()),'k')
            # plt.plot(edges[:-1][plot_sele],[nrm.ppf(0.99)]*(plot_sele.sum()),'k')
            # plt.legend()
            # plt.title('KK %d'%kk)
            # plt.xlabel('time [sec]')
            # plt.ylabel('counts')
            
            # select the significantly modulated
            is_sign = asses_mod_significance(ccg, nrm, 3)
            print(kk,)
            info_pair[cnt]['session'] = session
            info_pair[cnt]['reference electrode id'] = electrode_id_ref
            info_pair[cnt]['reference channel id'] = channel_id_ref
            info_pair[cnt]['reference cluster id'] = clust_id_ref
            info_pair[cnt]['referrence brain area'] = brain_area_ref[0]
            
            info_pair[cnt]['other electrode id'] = electrode_id_other
            info_pair[cnt]['other channel id'] = channel_id_other
            info_pair[cnt]['other cluster id'] = cluster_id_other
            info_pair[cnt]['other brain area'] = brain_area_other[0]
            info_pair[cnt]['is significant'] = is_sign
            
            cnt+=1
                     
np.savez('ccg_%s.npz'%session,info_pair=info_pair,ccg_res=ccg_res,edges=edges)
         
if not 'Utah' in sorted_fold:
    sorted_fold = path_user.get_path('cluster_array_data',session)
    if not os.path.exists(sorted_fold):
        sorted_fold = '/scratch/eb162/sorted/%s_PLEXON'%session
        lst = os.listdir('/scratch/eb162/sorted/%s_PLEXON'%session)
    else:
        lst = [1]
    if len(lst) != 0:
        
        spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality = \
        load_kilosort_data(sorted_fold, \
                        linearprobe_sampling_fq, \
                        use_master_clock=False,
                        include_pcs=False)
            
            
           
    
        max_time = np.max(spike_times)
        min_time = np.min(spike_times)
        
        # array_ba = ['PPC','PFC']
        
        unq_cl_lab = np.unique(spike_clusters)
        
        iii = 0
        keep_cl= np.ones(unq_cl_lab.shape[0],dtype=bool)
        for cl in unq_cl_lab:
            cond = (exp_data.spikes.cluster_id == clust_id_ref) &\
                ((exp_data.spikes.brain_area != 'PPC') &
                    (exp_data.spikes.brain_area != 'PFC'))
            if not any(cond):
                keep_cl[iii] = False
            iii+=1
    
        unq_cl_lab = unq_cl_lab[keep_cl]
    
        ccg_res_2 = np.zeros((unq_cl_lab.shape[0]*(unq_cl_lab.shape[0]+1)//2,2000))
        info_pair_2 = np.zeros(unq_cl_lab.shape[0]*(unq_cl_lab.shape[0]+1)//2, dtype=dict_dt)
        cnt = 0
        for jj in range(unq_cl_lab.shape[0]):
            clust_id_ref = unq_cl_lab[jj]
            iidx_ref = (exp_data.spikes.cluster_id == clust_id_ref) &\
                ((exp_data.spikes.brain_area != 'PPC') &
                    (exp_data.spikes.brain_area != 'PFC'))
            
            tspk1 = spike_times[spike_clusters == clust_id_ref]
            
            
            # other spike clust id in the same area
            
            
        
            brain_area_ref = exp_data.spikes.brain_area[iidx_ref]
            channel_id_ref = exp_data.spikes.channel_id[iidx_ref]
            electrode_id_ref = exp_data.spikes.electrode_id[iidx_ref]
        
            for kk in range(jj,unq_cl_lab.shape[0]):
                # if kk != 4:
                #     continue
                cluster_id_other = unq_cl_lab[kk]
                iidx = (exp_data.spikes.cluster_id == cluster_id_other) &\
                ((exp_data.spikes.brain_area == 'PPC') |
                    (exp_data.spikes.brain_area == 'PFC'))
                
                
                
                brain_area_other = exp_data.spikes.brain_area[iidx]
                channel_id_other = exp_data.spikes.channel_id[iidx]
                electrode_id_other = exp_data.spikes.electrode_id[iidx]
                
                for numbins in (np.array([2000]) + 1):
                    
                    # extract histogram
                    edges = np.linspace(-nsec, nsec, numbins)
                    # sele another electrode
                    tspk2 = spike_times[spike_clusters == cluster_id_other]
                    
                    # sele spikes after 1sec 
                    tspk1 = tspk1[(tspk1>=nsec) & (tspk1<=spike_times[-1]-nsec)]
                    hist_spks = np.zeros((edges.shape[0]-1,tspk1.shape[0]),dtype=float)
            
                    k = 0
                    for t0 in tspk1:
                        sub_t2 = tspk2[(tspk2 >= t0-nsec) & (tspk2 < t0+nsec)] - t0
                        hist_spks[:,k] = np.histogram(sub_t2,bins=edges)[0] 
                        k += 1
                   
                    ccg=np.mean(hist_spks,axis=1)/(edges[1]-edges[0])
                    
                    ccg_res_2[cnt] = ccg
                    
                    if jj == kk:
                        
                        mu = ccg[edges[:-1]!=0].mean()
                        std = ccg[edges[:-1]!=0].std()
                    else:
                        mu = ccg.mean()
                        std = ccg.std()
                    nrm = sts.norm(mu,std)
                     # plt.subplot(1,1,cc+1)
                    
                    plot_sele = (edges[:-1] >= -0.1) * (edges[:-1] <= 0.1) 
                    if kk ==jj:
                        plot_sele[edges[:-1]==0]=False
                    
                    # plt.plot(edges[:-1][plot_sele],ccg[plot_sele],label='DT: %.2f ms'%((edges[1]-edges[0])*1000))
                    # plt.plot(edges[:-1][plot_sele],[nrm.ppf(0.01)]*(plot_sele.sum()),'k')
                    # plt.plot(edges[:-1][plot_sele],[nrm.ppf(0.99)]*(plot_sele.sum()),'k')
                    # plt.legend()
                    # plt.title('KK %d'%kk)
                    # plt.xlabel('time [sec]')
                    # plt.ylabel('counts')
                    
                    # select the significantly modulated
                    is_sign = asses_mod_significance(ccg, nrm, 3)
                    print(kk,)
                    info_pair_2[cnt]['session'] = session
                    info_pair_2[cnt]['reference electrode id'] = electrode_id_ref
                    info_pair_2[cnt]['reference channel id'] = channel_id_ref
                    info_pair_2[cnt]['reference cluster id'] = clust_id_ref
                    info_pair_2[cnt]['referrence brain area'] = brain_area_ref[0]
                    
                    info_pair_2[cnt]['other electrode id'] = electrode_id_other
                    info_pair_2[cnt]['other channel id'] = channel_id_other
                    info_pair_2[cnt]['other cluster id'] = cluster_id_other
                    info_pair_2[cnt]['other brain area'] = brain_area_other[0]
                    info_pair_2[cnt]['is significant'] = is_sign
                    
                    cnt+=1
                
        info_pair =  np.hstack((info_pair,info_pair_2))
        ccg_res = np.vstack((ccg_res,ccg_res_2))
    
        np.savez('ccg_%s.npz'%session,info_pair=info_pair,ccg_res=ccg_res,edges=edges)
