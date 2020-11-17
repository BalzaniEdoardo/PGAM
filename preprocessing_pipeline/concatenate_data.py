import numpy as np
import sys,os
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from path_class import get_paths_class
from scipy.io import loadmat
from data_handler import *
from extract_presence_rate import *

user_paths = get_paths_class()
# list of session to be concatenatenconcat_list = ['m51s121', 'm51s122'] 
concat_list = ['m53s110']


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



# presence rate params
occupancy_bin_sec = 60 # at least one spike per min
occupancy_rate_th = 0.1 #hz

linearprobe_sampling_fq = 20000
utah_array_sampling_fq = 30000


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



    var_names = ('rad_vel','ang_vel','rad_path','ang_path','rad_target','ang_target',
                 'lfp_beta','lfp_alpha','lfp_theta','t_move','t_flyOFF','t_stop','t_reward','eye_vert','eye_hori')
    try:
        y, X, trial_idx = exp_data.concatenate_inputs(*var_names, t_start=t_start, t_stop=t_stop)
    except Exception as ex:
        print('\n\ncould not open %s'%session)
        print(ex,'\n\n')
        continue

    res = {'data_concat':{},'var_names':var_names}
    res['data_concat']['Yt'] = y.T
    res['data_concat']['Xt'] = np.zeros((X[var_names[0]].shape[0],len(var_names)))
    res['data_concat']['lfp_beta'] = X['lfp_beta'].T
    res['data_concat']['lfp_alpha'] = X['lfp_alpha'].T
    res['data_concat']['lfp_theta'] = X['lfp_theta'].T
    res['data_concat']['trial_idx'] = trial_idx
    res['info_trial'] = exp_data.info
    res['pre_trial_dur'] = pre_trial_dur
    res['post_trial_dur'] = post_trial_dur
    res['time_bin'] = 0.006
    res['unit_info'] = {}
    res['unit_info']['unit_type'] = exp_data.spikes.unit_type
    res['unit_info']['spike_width'] = exp_data.spikes.spike_width
    res['unit_info']['waveform'] = exp_data.spikes.waveform
    res['unit_info']['amplitude_wf'] = exp_data.spikes.amplitude_wf
    res['unit_info']['cluster_id'] = exp_data.spikes.cluster_id
    res['unit_info']['electrode_id'] = exp_data.spikes.electrode_id
    res['unit_info']['channel_id'] = exp_data.spikes.channel_id
    res['unit_info']['brain_area'] = exp_data.spikes.brain_area
    res['unit_info']['uQ'] = exp_data.spikes.uQ
    res['unit_info']['isiV'] = exp_data.spikes.isiV
    res['unit_info']['cR'] = exp_data.spikes.cR
    res['unit_info']['date_exp'] = exp_data.date_exp

    print('num units',exp_data.spikes.unit_type.shape)
    cc = 0
    for var in var_names:
        if var in ['phase','lfp_beta','lfp_alpha','lfp_theta']:
            res['data_concat']['Xt'][:, cc] = np.nan
            cc += 1
            continue
        res['data_concat']['Xt'][:,cc] = X[var]
        cc += 1

    # compute additional quality metrics
    res['unit_info'] = extract_presecnce_rate(occupancy_bin_sec,occupancy_rate_th,res['unit_info'],session,
                           user_paths,utah_array_sampling_fq,linearprobe_sampling_fq)


    
    if save:
        print('saving variables...')
        sv_folder = user_paths.get_path('local_concat')
        if not os.path.exists(sv_folder):
            os.mkdir(sv_folder)

        saveCompressed(os.path.join(sv_folder,'%s.npz'%session),unit_info=res['unit_info'],info_trial=res['info_trial'],data_concat=res['data_concat'],
             var_names=np.array(res['var_names']),time_bin=res['time_bin'],post_trial_dur=res['post_trial_dur'],
             pre_trial_dur=res['pre_trial_dur'],force_zip64=True)

    if send:
        try:
            print('...sending %s.npz to server'%session)
            sendfrom = sv_folder.replace(' ','\ ')
            dest_folder = user_paths.get_path('data_hpc')
            os.system('sshpass -p "%s" scp %s jpn5@prince.hpc.nyu.edu:%s' % ('savin1234!', os.path.join(sendfrom,'%s.npz'%session),dest_folder))
        except Exception as e:
            print(e)
            print('could not send files to the HPC cluster')