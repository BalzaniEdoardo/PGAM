import numpy as np
from python_monkey_info import monkey_info_class
import os,sys
from path_class import get_paths_class
user_path_gen = get_paths_class()
list_all_dir = [x[0] for x in os.walk(user_path_gen.get_path('ecephys_spike_sorting'))]

for dir_name in list_all_dir:
    sys.path.append(dir_name)
# from utils import *
# from metrics import *
import zipfile
import json
import glob
import pandas as pd


def find_range(x,a,b,option='within'):
    
    """
    Find indices of data within or outside range [a,b]

    Inputs:
    -------
    x - numpy.ndarray
        Data to search
    a - float or int
        Minimum value
    b - float or int
        Maximum value
    option - String
        'within' or 'outside'

    Output:
    -------
    inds - numpy.ndarray
        Indices of x that fall within or outside specified range

    """

    if option=='within':
        return np.where(np.logical_and(x>=a, x<=b))[0]
    elif option=='outside':
        return np.where(np.logical_or(x < a, x > b))[0]
    else:
        raise ValueError('unrecognized option paramter: {}'.format(option))


def rms(data):

    """
    Computes root-mean-squared voltage of a signal

    Input:
    -----
    data - numpy.ndarray

    Output:
    ------
    rms_value - float
    
    """

    return np.power(np.mean(np.power(data.astype('float32'),2)),0.5)

def write_probe_json(output_file, channels, offset, scaling, mask, surface_channel, air_channel, vertical_pos, horizontal_pos):

    """
    Writes a json file containing information about one Neuropixels probe.

    Inputs:
    -------
    output_file : file path
        Location for writing the json file
    channels : numpy.ndarray (384 x 0)
        Probe channel numbers
    offset : numpy.ndarray (384 x 0)
        Offset of each channel from zero
    scaling : numpy.ndarray (384 x 0)
        Relative noise level on each channel
    mask : numpy.ndarray (384 x 0)
        1 if channel contains valid data, 0 otherwise
    surface_channel : Int
        Index of channel at brain surface
    air_channel : Int
        Index of channel at interface between saline/agar and air
    vertical_pos : numpy.ndarray (384 x 0)
        Distance (in microns) of each channel from the probe tip
    horizontal_pos : numpy.ndarray (384 x 0)
        Distance (in microns) of each channel from the probe edge

    Outputs:
    --------
    output_file.json (written to disk)

    """

    with open(output_file, 'w') as outfile:
        json.dump( 
                  {  
                        'channel' : channels.tolist(), 
                        'offset' : offset.tolist(), 
                        'scaling' : scaling.tolist(), 
                        'mask' : mask.tolist(), 
                        'surface_channel' : surface_channel, 
                        'air_channel' : air_channel,
                        'vertical_pos' : vertical_pos.tolist(),
                        'horizontal_pos' : horizontal_pos.tolist()
                   },
                 
                  outfile, 
                  indent = 4, separators = (',', ': ') 
                 ) 

def read_probe_json(input_file):

    """
    Reads a json file containing information about one Neuropixels probe.

    Inputs:
    -------
    input_file : file path
        Location of file to read

    Outputs:
    --------
    mask : numpy.ndarray (384 x 0)
        1 if channel contains valid data, 0 otherwise
    offset : numpy.ndarray (384 x 0)
        Offset of each channel from zero
    scaling : numpy.ndarray (384 x 0)
        Relative noise level on each channel
    surface_channel : Int
        Index of channel at brain surface
    air_channel : Int
        Index of channel at interface between saline/agar and air

    """
    
    with open(input_file) as data_file:
        data = json.load(data_file)
    
    scaling = np.array(data['scaling'])
    mask = np.array(data['mask'])
    offset = np.array(data['offset'])
    surface_channel = data['surface_channel']
    air_channel = data['air_channel']

    return mask, offset, scaling, surface_channel, air_channel


def write_cluster_group_tsv(IDs, quality, output_directory, filename = 'cluster_group.tsv'):

    """
    Writes a tab-separated cluster_group.tsv file

    Inputs:
    -------
    IDs : list
        List of cluster IDs
    quality : list
        Quality ratings for each unit (same size as IDs)
    output_directory : String
        Location to save the file

    Outputs:
    --------
    cluster_group.tsv (written to disk)

    """
       
    df = pd.DataFrame(data={'cluster_id' : IDs, 'group': quality})
    
    print('Saving data...')
    
    df.to_csv(os.path.join(output_directory, filename), sep='\t', index=False)


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


def load(folder, filename):

    """
    Loads a numpy file from a folder.

    Inputs:
    -------
    folder : String
        Directory containing the file to load
    filename : String
        Name of the numpy file

    Outputs:
    --------
    data : numpy.ndarray
        File contents

    """

    return np.load(os.path.join(folder, filename))


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


def saveCompressed(fh, **namedict):
     with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                          allowZip64=True) as zf:
         for k, v in namedict.items():
             with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                 np.lib.npyio.format.write_array(buf,
                                                 np.asanyarray(v),
                                                 allow_pickle=True)




def compute_amplitude_tc(ampl_spk,time_spk,bin_sec,tot_time):
    bin_num = int(np.ceil(tot_time / bin_sec))
    ampl_median = np.zeros(bin_num) * np.nan
    idx_spk = np.array(np.floor(time_spk /bin_sec),dtype=int)

    for ii in np.unique(idx_spk):
        ampl_median[ii] = np.median(ampl_spk[idx_spk==ii])
    return ampl_median

def extract_presecnce_rate_Uprobe(occupancy_bin_sec,occupancy_rate_th,unit_info,session,
                                  path_user,linearprobe_sampling_fq):
    sorted_fold = path_user.get_path('cluster_data', session)
    N = unit_info['brain_area'].shape[0]
    unit_info['presence_rate'] = np.zeros(N)
    # unit_info['dip_pval'] = np.zeros(unit_info['brain_area'].shape[0])



    # first extract utah array
    # sorted_fold = base_sorted_fold % monkey_info.get_folder(session)
    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality= \
        load_kilosort_data(sorted_fold, \
                           linearprobe_sampling_fq, \
                           use_master_clock=False,
                           include_pcs=False)
    # tot time in sec
    max_time = np.max(spike_times)
    min_time = np.min(spike_times)
    print('dur recording array: %f'%(max_time-min_time))
    num_bins_occ = int(np.floor((max_time - min_time) / occupancy_bin_sec))
    # index of the utah array in the stacked files
    # select_array = (unit_info['brain_area'] == 'PPC') + (unit_info['brain_area'] == 'PFC')



    for unit in np.unique(unit_info['cluster_id']):
        # extract the index of the unit in the stacked file
        idx_un = np.where(  (unit_info['cluster_id'] == unit))[0]
        if idx_un.shape[0] != 1:
            raise ValueError

        idx_un = idx_un[0]

        unit_bool = spike_clusters==unit

        h, b = np.histogram(spike_times[unit_bool], np.linspace(min_time, max_time, num_bins_occ))
        occupancy = np.sum(h>0)/num_bins_occ
        unit_info['presence_rate'][idx_un] = occupancy

    return unit_info




def extract_presecnce_rate(occupancy_bin_sec,occupancy_rate_th,unit_info,session,
                           path_user,utah_array_sappling_fq,linearprobe_sampling_fq):
    # monkey_info = monkey_info_class()

    N = unit_info['brain_area'].shape[0]
    unit_info['presence_rate'] = np.zeros(N)
    # unit_info['dip_pval'] = np.zeros(unit_info['brain_area'].shape[0])



    # first extract utah array
    sorted_fold = path_user.get_path('cluster_data',session)
    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality= \
        load_kilosort_data(sorted_fold, \
                           utah_array_sappling_fq, \
                           use_master_clock=False,
                           include_pcs=False)
    # tot time in sec
    max_time = np.max(spike_times)
    min_time = np.min(spike_times)
    print('dur recording array: %f'%(max_time-min_time))
    num_bins_occ = int(np.floor((max_time - min_time) / occupancy_bin_sec))
    # index of the utah array in the stacked files
    select_array = (unit_info['brain_area'] == 'PPC') + (unit_info['brain_area'] == 'PFC')



    for unit in np.unique(unit_info['cluster_id'][select_array]):
        # extract the index of the unit in the stacked file
        idx_un = np.where(select_array * (unit_info['cluster_id'] == unit))[0]
        if idx_un.shape[0] != 1:
            raise ValueError

        idx_un = idx_un[0]

        unit_bool = spike_clusters==unit

        h, b = np.histogram(spike_times[unit_bool], np.linspace(min_time, max_time, num_bins_occ))
        occupancy = np.sum(h>0)/num_bins_occ
        unit_info['presence_rate'][idx_un] = occupancy

    # second extract linear prove
    sorted_fold = path_user.get_path('cluster_array_data',session)
    if not 'Utah Array' in sorted_fold and not session.startswith('m51'):
        if session == 'm53s35':
            split_sortlfd = sorted_fold.split(os.path.sep)
            iidxx = np.where(np.array(split_sortlfd)=='Utah Array')[0][0]
            sorted_fold = os.path.sep.join(split_sortlfd[:iidxx+1]+['Feb 05 2018/neural data/Sorted/Sorted/'])
        try:
            spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality = \
                load_kilosort_data(sorted_fold, \
                                   linearprobe_sampling_fq, \
                                   use_master_clock=False,
                                   include_pcs=False)
            # tot time in sec
            max_time = np.max(spike_times)
            min_time = np.min(spike_times)
            num_bins_occ = int(np.floor((max_time - min_time) / occupancy_bin_sec))
            print('dur recording probe: %f' % (max_time - min_time))
            # index of the liear probe in the stacked files
            select_array = (unit_info['brain_area'] != 'PPC') * (unit_info['brain_area'] != 'PFC')
    
            for unit in np.unique(unit_info['cluster_id'][select_array]):
                # extract the index of the unit in the stacked file
                idx_un = np.where(select_array * (unit_info['cluster_id'] == unit))[0]
                if idx_un.shape[0] != 1:
                    raise ValueError
    
                idx_un = idx_un[0]
    
                unit_bool = spike_clusters==unit
    
                h, b = np.histogram(spike_times[unit_bool], np.linspace(min_time, max_time, num_bins_occ))
                occupancy = np.sum(h>occupancy_rate_th*occupancy_bin_sec)/num_bins_occ
                unit_info['presence_rate'][idx_un] = occupancy
        except:
            print('No array data...')
    return unit_info