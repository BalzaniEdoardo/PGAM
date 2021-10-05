# load  data (requires a pull from the repo to get the pre processing code going)
import sys,os
import numpy as np
from scipy.io import loadmat
import dill
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
from data_handler import data_handler
from GAM_library import general_additive_model,GAM_result
from behav_class import load_eye_pos

# save directory
DIRECT = ''

session = 'm53s113'
# path to the .mat dataset
base_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel/'
dat = loadmat(os.path.join(base_file,'%s.mat'%(session)))



use_left_eye = 'right'
exp_data = data_handler(dat, 'trials_behv', 'units', None, 'behv_stats',
                        use_eye=use_left_eye,extract_fly_and_monkey_xy=True)
trials_behv = dat['trials_behv'].flatten()
exp_data.set_filters('all', True)
exp_data.filter = exp_data.filter + exp_data.info.get_replay(0,skip_not_ok=False)

time_pts, rate, sm_traj, raw_traj, fly_pos, cov_dict = exp_data.GPFA_YU_preprocessing([('t_targ','t_targ_off',15),('t_targ_off','t_stop',50),('t_stop','t_reward',15)],
                                                                                          var_list=['eye_vert','eye_hori','rad_vel','ang_vel','rad_target','ang_target'])

