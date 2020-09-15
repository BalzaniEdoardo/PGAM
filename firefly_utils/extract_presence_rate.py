import numpy as np
from python_monkey_info import monkey_info_class
import os,sys
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from path_class import get_paths_class
user_paths = get_paths_class()

list_all_dir = [x[0] for x in os.walk(user_paths.get_path('ecephys_spike_sorting'))]

for dir_name in list_all_dir:
    sys.path.append(dir_name)
from utils import *
from metrics import *
import zipfile

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




def extract_presecnce_rate(occupancy_bin_sec,occupancy_rate_th,unit_info,session,
                           base_sorted_fold,base_sorted_fold_array,utah_array_sappling_fq,linearprobe_sampling_fq):
    monkey_info = monkey_info_class()

    N = unit_info['brain_area'].shape[0]
    unit_info['presence_rate'] = np.zeros(N)
    # unit_info['dip_pval'] = np.zeros(unit_info['brain_area'].shape[0])



    # first extract utah array
    sorted_fold = base_sorted_fold % monkey_info.get_folder(session)
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
    sorted_fold = base_sorted_fold_array % monkey_info.get_folder(session)
    if not 'Utah Array' in sorted_fold and not session.startswith('m51'):

        spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality = \
            load_kilosort_data(sorted_fold, \
                               linearprobe_sampling_fq, \
                               use_master_clock=False,
                               include_pcs=False)
        # tot time in sec
        max_time = np.max(spike_times)
        min_time = np.min(spike_times)
        num_bins_occ = int(np.floor((max_time - min_time) / occupancy_bin_sec))
        print('dur recording array: %f' % (max_time - min_time))
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
    return unit_info