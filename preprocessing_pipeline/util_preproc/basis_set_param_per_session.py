import numpy as np
from copy import deepcopy


basis_info = {}

########################################################################################################################
# Session m53s95
########################################################################################################################
session = 'm53s95'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 40, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 30, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}
basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 10, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0,
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0 # acausal filter
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # (very few time points for the filter, use step functions)
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': None, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1, # causal filter (past history only affects future spikes)
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}



########################################################################################################################
# Session m53s83
########################################################################################################################
session = 'm53s83'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 11, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s84
########################################################################################################################
session = 'm53s84'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 30, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 11, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s86
########################################################################################################################
session = 'm53s86'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.8, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 11, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 11, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s91
########################################################################################################################
session = 'm53s91'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 18, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 60, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.8, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 7, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}



########################################################################################################################
# Session m53s97
########################################################################################################################
session = 'm53s97'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 18, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 7, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s98
########################################################################################################################
session = 'm53s98'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 18, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 60, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 7, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s100
########################################################################################################################
session = 'm53s100'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.4, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 9, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 50, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s100
########################################################################################################################
session = 'm53s105'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s106
########################################################################################################################
session = 'm53s106'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}



########################################################################################################################
# Session m53s107
########################################################################################################################
session = 'm53s107'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s109
########################################################################################################################
session = 'm53s109'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 9, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s108
########################################################################################################################
session = 'm53s108'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


# ########################################################################################################################
# # Session m53s109
# ########################################################################################################################
# session = 'm53s109'
# basis_info[session] = {}
# basis_info[session]['rad_vel'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None,
# }
#
# basis_info[session]['ang_vel'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':98, # max knot value as a percentile of the input vector of velocities,
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
# basis_info[session]['rad_path'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':99, # max knot value as a percentile of the input vector of velocities
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
#
# basis_info[session]['ang_path'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':98, # max knot value as a percentile of the input vector of velocities
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
#
# basis_info[session]['rad_target'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':99, # max knot value as a percentile of the input vector of velocities
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
#
#
# basis_info[session]['ang_target'] = {
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
#     'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
#     'init_knots_num': 17, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':False,
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None,
# }
#
# basis_info[session]['eye_vert'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
#     'dkn_max': 9, # max distance between knots (range is between -20 and 20...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':98, # max knot value as a percentile of the input vector of velocities
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
#
# basis_info[session]['eye_hori'] = {
#     'is_cyclic':False,
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
#     'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':99, # max knot value as a percentile of the input vector of velocities
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
#
# basis_info[session]['lfp_beta'] = {
# 'knot_0':-np.pi,
# 'knot_end':np.pi,
#     'knots_type':'fix_range', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 8, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':True,
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
# basis_info[session]['lfp_alpha'] = {
# 'knot_0':-np.pi,
# 'knot_end':np.pi,
#     'knots_type':'fix_range', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 8, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':True,
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
#
# basis_info[session]['lfp_theta'] = {
# 'knot_0':-np.pi,
# 'knot_end':np.pi,
#     'knots_type':'fix_range', # can be equi_spaced or equi_freq
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 8, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':True,
#     'kernel_length': None,  # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
# basis_info[session]['phase'] = {
#     'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
#     'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
#     'init_knots_num': 8, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':True,
#     'kernel_length': None, # total duration of the temporal kernel
#     'kernel_direction': None
# }
#
# basis_info[session]['t_move'] = {
#     'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'kernel_length': 161, # total duration of the temporal kernel
#     'init_knots_num': 8, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':False,
#     'kernel_direction':0
# }
#
# basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
# basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])
#
# basis_info[session]['t_flyOFF'] = {
#     'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
#     'order': 4, # for all except spike_hist
#     'penalty_type':'der', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'kernel_length': 322, # total duration of the temporal kernel
#     'init_knots_num': 15, # initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':False,
#     'kernel_direction':0
# }
#
# basis_info[session]['spike_hist'] = {
#     'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
#     'order':1, # for all except spike_hist
#     'penalty_type':'EqSpaced', # for all except spike_hist
#     'der': 2, # always penalize energy (2nd der) if order is 4
#     'kernel_length': 11, # total duration of the temporal kernel
#     'kernel_direction': 1,
#     'init_knots_num': 5,# initial knots number
#     'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
#     'perc_high':100, # max knot value as a percentile of the input vector of velocities
#     'is_cyclic':False
# }

########################################################################################################################
# Session m53s110
########################################################################################################################
session = 'm53s110'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 9, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s111
########################################################################################################################
session = 'm53s111'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 100, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s113
########################################################################################################################
session = 'm53s113'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1., # min distance between knots (range is between -20 and 20...)
    'dkn_max': 9, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}



########################################################################################################################
# Session m53s114
########################################################################################################################
session = 'm53s114'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 6, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s115
########################################################################################################################
session = 'm53s115'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 19, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s128
########################################################################################################################
session = 'm53s128'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s132
########################################################################################################################
session = 'm53s132'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':95, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m53s116
########################################################################################################################
session = 'm53s116'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1., # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 18, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s133
########################################################################################################################
session = 'm53s133'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.8, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.5, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s134
########################################################################################################################
session = 'm53s134'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m53s136
########################################################################################################################
session = 'm53s136'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 12, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 17, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s120
########################################################################################################################
session = 'm53s120'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s123
########################################################################################################################
session = 'm53s123'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s124
########################################################################################################################
session = 'm53s124'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}



#######################################################################################################################
# Session m53s123
########################################################################################################################
session = 'm53s123'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s124
########################################################################################################################
session = 'm53s124'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s92
########################################################################################################################
session = 'm53s92'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s93
########################################################################################################################
session = 'm53s93'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s90
########################################################################################################################
session = 'm53s90'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':95, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 3, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s93
########################################################################################################################
session = 'm53s93'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1.5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 20, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s122
########################################################################################################################
session = 'm53s122'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 21, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s126
########################################################################################################################
session = 'm53s126'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s130
########################################################################################################################
session = 'm53s130'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s47
########################################################################################################################
session = 'm53s47'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':97., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


#######################################################################################################################
# Session m53s46
########################################################################################################################
session = 'm53s46'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 1.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s35
########################################################################################################################
session = 'm53s35'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2., # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s34
########################################################################################################################
session = 'm53s34'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2., # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

#######################################################################################################################
# Session m53s33
########################################################################################################################
session = 'm53s33'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 12, # initial knots number
    'perc_low': 6, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99.9, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 5, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 30, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 300, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':93, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 25, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 100, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}



basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 80, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2., # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99., # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 2.5, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 20, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}


########################################################################################################################
# Session m51s120
########################################################################################################################
session = 'm51s120'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 40, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 30, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}
basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 10, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0,
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': None, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m51s121
########################################################################################################################
session = 'm51s121'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 2, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 40, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 5, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 30, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}
basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 10, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0,
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': None, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

########################################################################################################################
# Session m51s122
########################################################################################################################
session = 'm51s122'
basis_info[session] = {}
basis_info[session]['rad_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 20, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['ang_vel'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 10, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 50, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 3, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['rad_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_path'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 15, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 40, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['rad_target'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 40, # min distance between knots (range is between 0 and 200...)
    'dkn_max': 80, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['ang_target'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 8, # min distance between knots (range is approx between -40 and 200...)
    'dkn_max': 30, # max distance between knots (range is between -40 and 40...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None,
}

basis_info[session]['lfp_beta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}
basis_info[session]['eye_vert'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 10, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':98, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['eye_hori'] = {
    'is_cyclic':False,
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': 4, # min distance between knots (range is between -20 and 20...)
    'dkn_max': 8, # max distance between knots (range is between -20 and 20...)
    'init_knots_num': 15, # initial knots number
    'perc_low': 1, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':99, # max knot value as a percentile of the input vector of velocities
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['lfp_alpha'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}


basis_info[session]['lfp_theta'] = {
'knot_0':-np.pi,
'knot_end':np.pi,
    'knots_type':'fix_range', # can be equi_spaced or equi_freq
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None,  # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['phase'] = {
    'knots_type':'equi_freq', # can be equi_spaced or equi_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'dkn_min': np.pi/6, # min distance between knots (range is between 0 and 200...)
    'dkn_max': np.pi/2, # max distance between knots (range is between 0 and 200...)
    'init_knots_num': 8, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':True,
    'kernel_length': None, # total duration of the temporal kernel
    'kernel_direction': None
}

basis_info[session]['t_move'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 161, # total duration of the temporal kernel
    'init_knots_num': 10, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0,
}

basis_info[session]['t_stop'] = deepcopy(basis_info[session]['t_move'])
basis_info[session]['t_reward'] = deepcopy(basis_info[session]['t_move'])

basis_info[session]['t_flyOFF'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order': 4, # for all except spike_hist
    'penalty_type':'der', # for all except spike_hist
    'der': 2, # always penalize energy (2nd der) if order is 4
    'kernel_length': 322, # total duration of the temporal kernel
    'init_knots_num': 15, # initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False,
    'kernel_direction':0
}

basis_info[session]['spike_hist'] = {
    'knots_type':'temporal', # can be equi_spaced, eq_freq or temporal
    'order':1, # for all except spike_hist
    'penalty_type':'EqSpaced', # for all except spike_hist
    'der': None, # always penalize energy (2nd der) if order is 4
    'kernel_length': 11, # total duration of the temporal kernel
    'kernel_direction': 1,
    'init_knots_num': 5,# initial knots number
    'perc_low': 0, # min knot value as percentile of input vector (a couple of outliers with neg velocity)
    'perc_high':100, # max knot value as a percentile of the input vector of velocities
    'is_cyclic':False
}

del session