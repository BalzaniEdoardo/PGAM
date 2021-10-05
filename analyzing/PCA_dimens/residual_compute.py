import sys,os
import numpy as np
import matplotlib.pylab as plt
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')

from GAM_library import *
import dill
from utils_loading import unpack_preproc_data
from knots_constructor import *


neuron = 13
session = 'm53s113'
area = 'PPC'

cond_type = 'all'
cond_value = True




fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
        'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
        'unit_type','channel_id','electrode_id','cluster_id']

(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
  trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
  cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
  channel_id,electrode_id,cluster_id) = unpack_preproc_data(fhName, par_list)


# get the unit to include as input covariates
cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate_filter > 0.9
isi_v_filter = isi_v_filter < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

# unit number according to matlab indexing
neuron_keep = np.arange(1, yt.shape[1] + 1)[combine_filter]


cond_knots = None

neuron_fit = neuron_keep[brain_area[combine_filter]==area]
Xt = Xt[:1000]
yt = yt[:1000]
lfp_theta = lfp_theta[:1000]
lfp_beta = lfp_theta[:1000]
lfp_gamma = lfp_theta[:1000]
trial_idx = trial_idx[:1000]


train_trials = np.where(trial_type[cond_type] == cond_value)[0]


# take the train trials
keep = []
for ii in train_trials:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

print(' condition', cond_type, cond_value)

keep = np.array(keep, dtype=int)
trial_idx_train = trial_idx[keep]


# fit with coupling
hand_vel_temp = True
sm_handler = smooths_handler()
dict_xlims = {}

for var in var_names:
    # for now skip
    # if var !='spike_hist':
    #     continue
    if var == 'hand_vel1' or var == 'hand_vel2':
        continue




    if var == 'lfp_theta':
        x = lfp_theta[keep, neuron - 1]


    elif var == 'lfp_beta':
        x = lfp_beta[keep, neuron - 1]


    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, neuron - 1]


    elif var == 'spike_hist':
        tmpy = yt[keep, neuron - 1]
        x = tmpy


    else:
        cc = np.where(var_names == var)[0][0]
        x = Xt[keep, cc]


    knots, x_trans, include_var, is_cyclic, order, \
    kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
        knots_cerate(x, var, session, hand_vel_temp=hand_vel_temp, hist_filt_dur='short',
                     exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                     condition=cond_knots)



    if not var.startswith('t_') and var != 'spike_hist':
        if 'lfp' in var:
            dict_xlims[var] = (-np.pi, np.pi)
        else:
            if not knots is None:
                xx0 = max(np.nanpercentile(x_trans, 0), knots[0])
                xx1 = min(np.nanpercentile(x_trans, 100), knots[-1])
            else:
                xx0 = None
                xx1 = None
            dict_xlims[var] = (xx0, xx1)
    else:
        dict_xlims[var] = None

    # print(np.nanmax(np.abs(x_trans)),np.nanmax(np.abs(x_test)))
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                              knots_num=None, perc_out_range=None,
                              is_cyclic=[is_cyclic], lam=50,
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx_train, time_bin=time_bin,
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len,
                              kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                              repeat_extreme_knots=False)



for other in neuron_keep:
    # # break
    # if other == neuron:
    #     continue
    print('adding unit: %d' % other)
    if area == brain_area[other - 1]:
        filt_len = 'short'
    else:
        filt_len = 'long'

    tmpy = yt[keep, other - 1]
    x = tmpy

    knots, x_trans, include_var, is_cyclic, order, \
    kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
        knots_cerate(x, 'spike_hist', session, hand_vel_temp=hand_vel_temp, hist_filt_dur=filt_len,
                     exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'])


    var = 'neu_%d' % other
    if include_var:
        if var in sm_handler.smooths_dict.keys():
            sm_handler.smooths_dict.pop(var)
            sm_handler.smooths_var.remove(var)

        sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                              knots_num=None, perc_out_range=None,
                              is_cyclic=[is_cyclic], lam=50,
                              penalty_type=penalty_type,
                              der=der,
                              trial_idx=trial_idx_train, time_bin=time_bin,
                              is_temporal_kernel=is_temporal_kernel,
                              kernel_length=kernel_len, kernel_direction=kernel_direction, ord_AD=3, ad_knots=4)

for neuron in neuron_fit:
    print('neuron ',neuron)
    for var in ['lfp_beta', 'lfp_alpha', 'lfp_theta', 'spike_hist']:
        if var != 'spike_hist':
            is_cyclic = True
        else:
            is_cyclic = False


        if var == 'lfp_theta':
            x = lfp_theta[keep, neuron - 1]


        elif var == 'lfp_beta':
            x = lfp_beta[keep, neuron - 1]


        elif var == 'lfp_alpha':
            x = lfp_alpha[keep, neuron - 1]


        elif var == 'spike_hist':
            tmpy = yt[keep, neuron - 1]
            x = tmpy


        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep, cc]


        knots, x_trans, include_var, is_cyclic, order, \
        kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
            knots_cerate(x, var, session, hand_vel_temp=hand_vel_temp, hist_filt_dur='short',
                         exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                         condition=cond_knots)



        if not var.startswith('t_') and var != 'spike_hist':
            if 'lfp' in var:
                dict_xlims[var] = (-np.pi, np.pi)
            else:
                if not knots is None:
                    xx0 = max(np.nanpercentile(x_trans, 0), knots[0])
                    xx1 = min(np.nanpercentile(x_trans, 100), knots[-1])
                else:
                    xx0 = None
                    xx1 = None
                dict_xlims[var] = (xx0, xx1)
        else:
            dict_xlims[var] = None

        # print(np.nanmax(np.abs(x_trans)),np.nanmax(np.abs(x_test)))
        if include_var:
            if var in sm_handler.smooths_dict.keys():
                sm_handler.smooths_dict.pop(var)
                sm_handler.smooths_var.remove(var)

            sm_handler.add_smooth(var, [x_trans], ord=order, knots=[knots],
                                  knots_num=None, perc_out_range=None,
                                  is_cyclic=[is_cyclic], lam=50,
                                  penalty_type=penalty_type,
                                  der=der,
                                  trial_idx=trial_idx_train, time_bin=time_bin,
                                  is_temporal_kernel=is_temporal_kernel,
                                  kernel_length=kernel_len,
                                  kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                                  repeat_extreme_knots=False)


    fullX, idx_dict = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)

    full = dill.load(open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_all_1.0000.dill'%(session,session,neuron),'rb'))['full']

    # predict rate
    FFX = np.ones((fullX.shape[0], full.beta.shape[0]))
    for var in full.index_dict.keys():
        FFX[:, full.index_dict[var]] = fullX[:,idx_dict[var]]
    del fullX,idx_dict
    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)
    eta = np.dot(FFX, full.beta)
    mu = family.link.inverse(eta)


