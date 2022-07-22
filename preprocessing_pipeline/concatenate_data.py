import numpy as np
import sys,os,re
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

use_server = 'server'
# =============================================================================
# Here you should give the base directory that contains the .mat
# the code will walk through alll subdirectory, and if it will find a
# mat file with the correct name it will list it as a file to be concatenated
#
# =============================================================================
DIRECT = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'

print('The code assumes that the lfp_session.mat  files are in the same folder as the session.mat file!')
# list of session to be concatenated


concat_list = []
fld_list = []
pattern_fh = '^m\d+s\d+.mat$'
minSess = 12
for root, dirs, files in os.walk(DIRECT, topdown=False):
    for name in files:
        if re.match(pattern_fh,name):
            if not 'm72'  in name:
                continue
            sess_num = int(name.split('s')[1].split('.')[0])
            if sess_num < minSess:
                continue
            concat_list += [name.split('.mat')[0]]
            fld_list += [root]

           
# ii = np.where(np.array(concat_list)=='m53s96')[0][0]
# concat_list = concat_list[ii+1:]
# fld_list = fld_list[ii+1:]

concat_list =  ['m53s49']#['m53s38','m53s39', 'm53s48', 'm53s49', 'm53s50', 'm53s51']
# concat_list = ['m72s2']

sv_folder = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/concatenation_with_accel'

# concat_list = []

ptrn = '^m\d+s\d+.mat$'
ptrn = '^m\d+s\d+$'

# cc_list = ['m53s51']
# for name in cc_list: #os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'):
#     if  not re.match(ptrn,name):
#         continue
#     print(name)
#     svname = name.replace('.mat','.npz')
#     if not '.' in name:
#         svname = name + '.npz'
#     if not os.path.exists(os.path.join(sv_folder,svname)):
#         concat_list += [name.split('.')[0]]


fld_list = ['/Volumes/Balsip HD/dataset_firefly/']*len(concat_list)

save = True
send = True
# concat_list = ['m51s38']
# fld_list = ['Users/jean-paulnoel/Documents/Savin-Angelaki/saved']*len(concat_list)

# destination folder
#sv_folder = '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/PPC+PFC+MST/'
# sv_folder = '/Users/edoardo/Work/Code/GAM_code/fitting/'

# path to files

# path to preproc mat files
#base_file = '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/PPC+PFC+MST/'
#base_file = user_paths.get_path('local_concat','m44s174')

base_file = '/Volumes/Balsip HD/dataset_firefly/'
baseflld = os.path.dirname(base_file)

#result_fld = '/Volumes/WD Edo/firefly_analysis/LFP_band/results_radTarg/'

# list of session in which forcing the use of left eye posiiton
use_left_eye = ['53s48']


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

phase_precomputed = []
cnt_concat = 0


for session in concat_list:
    
    # if not session in ['m71s18']:
    #     continue
    if session in use_left_eye:
        use_eye = 'left'
    else:
        use_eye = 'right'

    if session in ['m51s43','m51s38','m51s41','m51s42','m51s40']:
        fhLFP = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/DATASET_accel/lfps_%s.mat'%session
    else:
        fhLFP = ''
    base_file = fld_list[cnt_concat]
    cnt_concat += 1

    

    print('loading session %s...'%session)
    pre_trial_dur = 0.2
    post_trial_dur = 0.2

    try:
        dat = loadmat(os.path.join(base_file,'%s.mat'%(session)))
    except:
        print('could not find', session)
        continue
    try:
        lfp_beta = loadmat(os.path.join(base_file,'lfp_beta_%s.mat'%session))
        lfp_alpha = loadmat(os.path.join(base_file,'lfp_alpha_%s.mat'%session))
        lfp_theta = loadmat(os.path.join(base_file,'lfp_theta_%s.mat'%session))
    except:
        print('could not find LFP', session)
        continue 
        
    

    if 'is_phase' in lfp_beta.keys():
        is_phase = lfp_beta['is_phase'][0,0]
    else:
        is_phase = False
    
    if is_phase:
        phase_precomputed += [session]
        continue
    # try:
    exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur,
                        post_trial_dur=post_trial_dur,
                        lfp_beta=lfp_beta['lfp_beta'], lfp_alpha=lfp_alpha['lfp_alpha'],lfp_theta=lfp_theta['lfp_theta'], extract_lfp_phase=(not is_phase),
                        use_eye=use_eye,fhLFP=fhLFP)
    # except Exception as e:
    #     print('unable to open', session,'\n',e)
    #     continue
    
##########################################################################
    #  SETTA IL REPLAY DI MODO DA INSERIRE SOLO I TRIAL CHE CORRISPONDONO ALL'ACTIVE PHASE
    #  SALVA IL TRIAL ID
    exp_data.set_filters('all', True)

    if any(exp_data.info.trial_type['replay'] == 0): # replay triial are available
        trial_all_id = exp_data.behav.trial_id[np.where(exp_data.filter)[0]]
        repl_tr = []
        for id in trial_all_id:
            pair = np.where(exp_data.behav.trial_id == id)[0]
            for tr in pair:
                if exp_data.info.trial_type['replay'][tr] == 0:
                    repl_tr += [tr]
        exp_data.filter[np.array(repl_tr)] = True

    # # impose all replay trials
    # is_replay = any(exp_data.info.get_replay(0,skip_not_ok=False))
    # exp_data.filter = exp_data.filter + exp_data.info.get_replay(0,skip_not_ok=False)
    #

    # savemat('lfp_raw_%s.mat'%session,{'lfp':exp_data.lfp.lfp})

    t_targ = dict_to_vec(exp_data.behav.events.t_targ)
    t_move = dict_to_vec(exp_data.behav.events.t_move)

    t_start = np.min(np.vstack((t_move, t_targ)), axis=0) - pre_trial_dur
    t_stop = dict_to_vec(exp_data.behav.events.t_end) + post_trial_dur



    var_names = ('rad_vel','ang_vel','rad_path','ang_path','rad_target','ang_target',
                 'lfp_beta','lfp_alpha','lfp_theta','t_move','t_flyOFF','t_stop','t_reward','eye_vert','eye_hori',
                 'hand_vel1','hand_vel2','rad_acc','ang_acc','rad_vel_diff','rad_vel_ptb',
                 'ang_vel_diff','ang_vel_ptb','t_ptb')
                 #'lfp_alpha_power',
                 #'lfp_theta_power','lfp_beta_power')
    try:
        y, X, trial_idx = exp_data.concatenate_inputs(*var_names, t_start=t_start, t_stop=t_stop)
    except MemoryError as ex:
        print('\n\ncould not open %s'%session)
        print(ex,'\n\n')
        continue
    
    var_Xt = []
    for var in var_names:
        if 'lfp' in var or var == 'phase':
            continue
        var_Xt += [var]
    
    res = {'data_concat':{},'var_names':var_Xt}
    res['data_concat']['Yt'] = y.T
    res['data_concat']['Xt'] = np.zeros((X[var_names[0]].shape[0],len(var_Xt)))
    res['data_concat']['lfp_beta'] = X['lfp_beta'].T
    res['data_concat']['lfp_alpha'] = X['lfp_alpha'].T
    res['data_concat']['lfp_theta'] = X['lfp_theta'].T
    res['data_concat']['trial_idx'] = trial_idx
    # res['lfp_alpha_power'] = X['lfp_alpha_power'].T
    # res['lfp_theta_power'] = X['lfp_theta_power'].T
    # res['lfp_beta_power'] = X['lfp_beta_power'].T
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
    for var in var_Xt:
        # if 'lfp' in var or var == 'phase':
        #     continue
        # if var in ['phase','lfp_beta','lfp_alpha','lfp_theta'] or 'lfp' in var:
        #     res['data_concat']['Xt'][:, cc] = np.nan
        #     cc += 1
        #     continue
        res['data_concat']['Xt'][:,cc] = X[var]
        cc += 1

    # compute additional quality metrics
    try:
        if ('m72' in session) or ('m73' in session):
            res['unit_info'] = extract_presecnce_rate_Uprobe(occupancy_bin_sec,occupancy_rate_th,res['unit_info'],session,
                                                             user_paths,linearprobe_sampling_fq,use_server=False)
        else:
            res['unit_info'] = extract_presecnce_rate(occupancy_bin_sec,occupancy_rate_th,res['unit_info'],session,
                           user_paths,utah_array_sampling_fq,linearprobe_sampling_fq,use_server=False)
    except Exception as e:
        print(e)
        print('skip %s'%session)


    
    if save:
        print('saving variables...')
        # sv_folder = base_file#user_paths.get_path('local_concat')
        if not os.path.exists(sv_folder):
            os.mkdir(sv_folder)

        saveCompressed(os.path.join(sv_folder,'%s.npz'%session),unit_info=res['unit_info'],info_trial=res['info_trial'],data_concat=res['data_concat'],
             var_names=np.array(res['var_names']),time_bin=res['time_bin'],post_trial_dur=res['post_trial_dur'],
             pre_trial_dur=res['pre_trial_dur'], force_zip64=True)#,lfp_alpha_power=res['lfp_alpha_power'],
             #lfp_beta_power=res['lfp_beta_power'],lfp_theta_power=res['lfp_theta_power'],
             #force_zip64=True)

    if send:
        try:
            print('...sending %s.npz to server'%session)
            sendfrom = sv_folder.replace(' ','\ ')
            dest_folder = user_paths.get_path('data_hpc')
            if 'jpn5' in dest_folder:
                dest_folder=dest_folder.replace('jpn5','eb162')
            # os.system('sshpass -p "%s" scp %s eb162@prince.hpc.nyu.edu:%s' % ('', os.path.join(sendfrom,'%s.npz'%session),dest_folder))
            os.system('sshpass -p "%s" scp %s eb162@greene.hpc.nyu.edu:%s' % ('savin12345!', os.path.join(sendfrom,'%s.npz'%session),dest_folder))
        except Exception as e:
            print(e)
            print('could not send files to the HPC cluster')