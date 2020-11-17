#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:19:10 2020

@author: edoardo
"""


import numpy as np
import sys,os
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline','util_preproc'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from path_class import get_paths_class
from scipy.io import loadmat
from data_handler import *
from extract_presence_rate import *
from scipy.io import loadmat
from path_class import get_paths_class
path_user = get_paths_class()
import matplotlib.pylab as plt
from GAM_library import *

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
        raise ValueError('something in the logic is odd')
    
    is_sign = any(idx_last - idx_first >= consecEp)
    
    
    idx_first = np.where(np.diff(np.array(ccg[plot_sele] < nrm.ppf(0.01),dtype=int))==1)[0]
    idx_last = np.where(np.diff(np.array(ccg[plot_sele] < nrm.ppf(0.01),dtype=int))==-1)[0]

    if len(idx_last) == 0 and len(idx_first) == 1:
        idx_last = np.array([ccg.shape[0]])
        
    elif len(idx_last) != 0 and len(idx_first) != 0:
        idx_first = idx_first[idx_first < idx_last[-1]]
        idx_last = idx_last[idx_last > idx_first[0]]
    
    if len(idx_last) != len(idx_first):
        raise ValueError('something in the logic is odd')
    
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
base_file = user_paths.get_path('local_concat','m44s174')


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

for session in concat_list:


    if session in use_left_eye:
        use_eye = 'left'
    else:
        use_eye = 'right'


    

    print('loading session %s...'%session)
    pre_trial_dur = 0.2
    post_trial_dur = 0.2

    try:
        dat = loadmat(os.path.join(base_file,'%s.mat'%(session)))
    except:
        print('could not find', session)
        continue

    lfp_beta = loadmat(os.path.join(base_file,'lfp_beta_%s.mat'%session))
    lfp_alpha = loadmat(os.path.join(base_file,'lfp_alpha_%s.mat'%session))
    lfp_theta = loadmat(os.path.join(base_file,'lfp_theta_%s.mat'%session))

    if 'is_phase' in lfp_beta.keys():
        is_phase = lfp_beta['is_phase'][0,0]
    else:
        is_phase = False

    exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur,
                            post_trial_dur=post_trial_dur,
                            lfp_beta=lfp_beta['lfp_beta'], lfp_alpha=lfp_alpha['lfp_alpha'],lfp_theta=lfp_theta['lfp_theta'], extract_lfp_phase=(not is_phase),
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
    
    choer_folder = '/Volumes/WD Edo/firefly_analysis/LFP_band/processed_data/LFP_coherence/'
    # matrix_ccg,ccfedge,tmp = compute_ccg_tensor(spikes_cut,session,choer_folder,electrode_id, brain_area,
    #                    time_delta=0.002,window='full', cross_corr_coef=False, binary=False, border_correct=True)
    # lead_electrode_missing = np.hstack((lead_electrode_missing,tmp))
    
    
    
    
    lfp_cho_mat = loadmat(os.path.join(choer_folder,'LFP_coherence_%s.mat'%session))
    area = lfp_cho_mat['lead_electrode'][0]
    
    ba = area[0][0]
    ele = area[1][0,0]
    sele = (exp_data.spikes.electrode_id == ele) & (exp_data.spikes.brain_area == ba) 
    if not any(sele):
        tmp = np.zeros(1,dtype=dtype_dict)
        tmp['session'] = session
        tmp['electrode_id'] = ele
        tmp['brain_area'] = ba
        lead_electrode_missing = np.hstack((lead_electrode_missing,tmp))
    elif ba in ['PPC','PFC']: # tot time in sec
        # first extract utah array
        sorted_fold = path_user.get_path('cluster_data',session)
        spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality = \
            load_kilosort_data(sorted_fold, \
                           utah_array_sampling_fq, \
                           use_master_clock=False,
                           include_pcs=False)
        
        max_time = np.max(spike_times)
        min_time = np.min(spike_times)
        ba = area[0][0]
        ele = area[1][0,0]
        sele = (exp_data.spikes.electrode_id == ele) & (exp_data.spikes.brain_area == ba) 
        
        clust_id_ref = exp_data.spikes.cluster_id[sele]

        tspk1 = spike_times[spike_clusters == clust_id_ref[0]]
        
        
        # other spike clust id in the same area
        sele = (exp_data.spikes.brain_area == ba)  & (exp_data.spikes.cluster_id != clust_id_ref[0])
        other_cl_ids = exp_data.spikes.cluster_id[sele]
        other_ele_id = exp_data.spikes.electrode_id[sele]
        
        for kk in range(other_cl_ids.shape[0]):
            # if kk != 4:
            #     continue
            cc = 0
            plt.figure(figsize=(9,4))
            other_ele = other_ele_id[kk]
            clust_id_ele = other_cl_ids[kk]
            for numbins in (np.array([2000]) + 1):
                
                # extract histogram
                edges = np.linspace(-nsec, nsec, numbins)
                # sele another electrode
                tspk2 = spike_times[spike_clusters == other_cl_ids[kk]]
                
                # sele spikes after 1sec 
                tspk1 = tspk1[(tspk1>=nsec) & (tspk1<=spike_times[-1]-nsec)]
                hist_spks = np.zeros((edges.shape[0]-1,tspk1.shape[0]),dtype=float)
        
                k = 0
                for t0 in tspk1:
                    sub_t2 = tspk2[(tspk2 >= t0-nsec) & (tspk2 < t0+nsec)] - t0
                    hist_spks[:,k] = np.histogram(sub_t2,bins=edges)[0] 
                    k += 1
                plt.subplot(1,1,cc+1)
                cc+=1
                ccg=np.mean(hist_spks,axis=1)/(edges[1]-edges[0])
                mu = ccg.mean()
                std = ccg.std()
                nrm = sts.norm(mu,std)
                
                plot_sele = (edges[:-1] >= -0.1) * (edges[:-1] <= 0.1)
                plt.plot(edges[:-1][plot_sele],ccg[plot_sele],label='DT: %.2f ms'%((edges[1]-edges[0])*1000))
                plt.plot(edges[:-1][plot_sele],[nrm.ppf(0.01)]*(plot_sele.sum()),'k')
                plt.plot(edges[:-1][plot_sele],[nrm.ppf(0.99)]*(plot_sele.sum()),'k')
                plt.legend()
                plt.title('KK %d'%kk)
                plt.xlabel('time [sec]')
                plt.ylabel('counts')
                
                # select the significantly modulated
                print(kk,asses_mod_significance(ccg, nrm, 3))
                
            
            # plt.savefig('raw_clref_%d_clref_%d_fit_ref_%d_other_%d.png'%(clust_id_ref,clust_id_ele,ele,other_ele))

        #     sm_handler = smooths_handler()
        #     bin1 = binning(tspk1,shift=0.006,mnTime=spike_times[0],mxTime=spike_times[-1])
        #     bin2 = binning(tspk2,shift=0.006,mnTime=spike_times[0],mxTime=spike_times[-1])
        #     nrm = sts.norm(0,0.5)
        #     knots = nrm.ppf(np.linspace(0.005,0.995,20))
        #     # nrm = sts.uniform(0,1)
    
        #     # knots = nrm.ppf(np.linspace(0.005,0.995,15))
    
        #     knots = 505*knots / knots[-1]
        #     knots = np.hstack(([knots[0]]*3,knots,[knots[-1]]*3))
        #     sm_handler.add_smooth('ref_unit',[bin1],knots_num=None,perc_out_range=0,pre_trial_dur=0.,
        #                           trial_idx = np.ones(bin2.shape),kernel_length=501,
        #                           is_temporal_kernel=True,knots=[knots],time_bin=0.006,lam=50)
        #     sm_handler.add_smooth('spike_hist',[bin2],time_bin=0.006,trial_idx = np.ones(bin2.shape),
        #                           perc_out_range=0,pre_trial_dur=0.,is_temporal_kernel=True,
        #                           knots_num=15,kernel_length=165,kernel_direction=1,lam=50)
            
        #     link = deriv3_link(sm.genmod.families.links.log())
        #     poissFam = sm.genmod.families.family.Poisson(link=link)
        #     family = d2variance_family(poissFam)
    
        #     gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, bin2, poissFam,
        #                                    fisher_scoring=True)
    
    
        #     full, reduced = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001, method='L-BFGS-B', tol=1e-8,
        #                                                conv_criteria='gcv',
        #                                                max_iter=10000, gcv_sel_tol=10 ** -13, random_init=False,
        #                                                use_dgcv=True, initial_smooths_guess=False,
        #                                                fit_initial_beta=True, pseudoR2_per_variable=True,
        #                                                trial_num_vec=np.ones(bin2.shape,dtype=int), k_fold=False, fold_num=None,
        #                                                reducedAdaptive=False)

        #     cc_plot = 1
        #     gam_res = full
        #     plt.close('all')
        #     cc = 0
        #     for var in full.var_list:
        #         if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
        #             cc += 1
        #             continue
        #         print('plotting var', var)
    
        #         ax = plt.subplot(1, 2, cc_plot)
                
    
                
    
        #         dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
        #         knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
        #         ord_ = gam_res.smooth_info[var]['ord']
        #         # idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)
    
        #         impulse = np.zeros(dim_kern)
        #         impulse[(dim_kern - 1) // 2] = 1
        #         xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
        #         fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99, trial_idx=None)
        #         if var != 'spike_hist':
        #             xx = xx[1:-1]
        #             fX = fX[1:-1]
        #             fX_p_ci = fX_p_ci[1:-1]
        #             fX_m_ci = fX_m_ci[1:-1]
        #         else:
        #             xx = xx[:(-ord_ - 1)]
        #             fX = fX[:(-ord_ - 1)]
        #             fX_p_ci = fX_p_ci[:(-ord_ - 1)]
        #             fX_m_ci = fX_m_ci[:(-ord_ - 1)]
    
                
    
        #         if var == 'spike_hist':
        #             iend = xx.shape[0] // 2
    
        #             print('set spike_hist')
        #             fX = fX[iend+3 :][::-1]
        #             fX_p_ci = fX_p_ci[iend +3:][::-1]
        #             fX_m_ci = fX_m_ci[iend+3 :][::-1]
        #             plt.plot(xx[:iend-3], fX, ls='-', color='k', label=var)
        #             plt.fill_between(xx[:iend-3], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
        #         else:
        #             hh = (np.mean(hist_spks,axis=1)-np.mean(np.mean(hist_spks,axis=1)))/np.max(np.mean(hist_spks,axis=1)-np.mean(np.mean(hist_spks,axis=1)))*np.max(fX)
        #             hh = hh[(edges[:-1] >=xx[0])*(edges[:-1] <= xx[-1])]
        #             xxh = edges[(edges >=xx[0])*(edges <= xx[-1])]
        #             plt.plot(xxh,hh)
        #             plt.plot(xx, fX, ls='-', color='k', label=var)
        #             plt.fill_between(xx, fX_m_ci, fX_p_ci, color='k', alpha=0.4)
    
        #         ax.spines['top'].set_visible(False)
        #         ax.spines['right'].set_visible(False)
        #         plt.legend()
    
        #         cc += 1
        #         cc_plot += 1
        #     plt.savefig('clref_%d_clref_%d_fit_ref_%d_other_%d.png'%(clust_id_ref,clust_id_ele,ele,other_ele))
        # # ## method 2 collect all intevals
        # # max_time = np.max(spike_times)
        # # min_time = np.min(spike_times)
        # # ba = area[0][0]
        # # ele = area[1][0,0]
        # # sele = (exp_data.spikes.electrode_id == ele) & (exp_data.spikes.brain_area == ba) 
        
        # # clust_id_ref = exp_data.spikes.cluster_id[sele]

        # # tspk1 = spike_times[spike_clusters == clust_id_ref[0]]
        
        # # # other spike clust id in the same area
        # # sele = (exp_data.spikes.brain_area == ba)  & (exp_data.spikes.cluster_id != clust_id_ref[0])
        # # other_cl_ids = exp_data.spikes.cluster_id[sele]
        

        # # # sele another electrode
        # # tspk2 = spike_times[spike_clusters == other_cl_ids[0]]
        
        # # # sele spikes after 1sec 
        # # tspk1 = tspk1[(tspk1>=1) & (tspk1<=spike_times[-1]-1)]
        # # all_isi = []
      
        # # for t0 in tspk1:
        # #     sub_t2 = tspk2[(tspk2 >= t0-nsec) & (tspk2 < t0+nsec)] - t0
        # #     all_isi = np.hstack((all_isi,sub_t2))
   
        

        