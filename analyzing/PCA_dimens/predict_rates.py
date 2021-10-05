import sys,os
import numpy as np
import matplotlib.pylab as plt
import re
if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc')

    sess_list = ['m53s113']
    JOB = 0
    clust = False
    area = 'PFC'
else:
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')

    sess_list = []
    for fhn in os.listdir('/scratch/jpn5/dataset_firefly'):
        if re.match('^m\d+s\d+.npz$',fhn):
            sess_list += [fhn.split('.')[0]]
    clust = True
    area = sys.argv[2]
    JOB = int(sys.argv[1]) - 1

from GAM_library import *
import dill
from utils_loading import unpack_preproc_data
from knots_constructor import *




session = sess_list[JOB]

cond_type = 'all'
cond_value = True
if not clust:
    fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
else:
    fhName = '/scratch/jpn5/dataset_firefly/%s.npz' % session



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

# Xt = Xt[:1000]
# yt = yt[:1000]
# lfp_theta = lfp_theta[:1000]
# lfp_beta = lfp_theta[:1000]
# lfp_gamma = lfp_theta[:1000]
# trial_idx = trial_idx[:1000]


# create the results to be saved
meanFr = np.zeros((Xt.shape[0],neuron_fit.shape[0]),dtype=np.float32)
meanFr_noHist = np.zeros((Xt.shape[0],neuron_fit.shape[0]),dtype=np.float32)
meanFr_noCP = np.zeros((Xt.shape[0],neuron_fit.shape[0]),dtype=np.float32)
meanFr_noInt = np.zeros((Xt.shape[0],neuron_fit.shape[0]),dtype=np.float32)

pseudo_r2 = np.zeros(neuron_fit.shape[0])



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
        knots_cerate(x, var, session, hand_vel_temp=hand_vel_temp, hist_filt_dur='long',
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
        filt_len = 'long'
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


print('changing to full')
for var in sm_handler.smooths_var:
    sm_handler[var].X = sm_handler[var].X.toarray()
print('done!')




FFX_all, idx_dict_all = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)

cnt_neu = 0
rm_neuron = []
for neuron in neuron_fit:
    if not clust:
        fit_res_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/fit_longFilters/gam_%s/fit_results_%s_c%d_all_1.0000.dill'%(session,session,neuron)
    else:
        fit_res_path = '/scratch/jpn5/fit_longFilters/gam_%s/fit_results_%s_c%d_all_1.0000.dill'%(session,session,neuron)

    if not os.path.exists(fit_res_path):
        rm_neuron += [cnt_neu]
        cnt_neu += 1
        continue

    print('neuron ',neuron)
    for var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
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
            knots_cerate(x, var, session, hand_vel_temp=hand_vel_temp, hist_filt_dur='long',
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


    fullX, idx_dict = sm_handler.get_exog_mat_fast(['lfp_beta', 'lfp_alpha', 'lfp_theta'])


    dd = dill.load(open(fit_res_path,'rb'))
    full = dd['full']
    # predict rate
    FFX = np.ones((fullX.shape[0], full.beta.shape[0]))
    for var in full.index_dict.keys():
        if var == 'spike_hist':
            FFX[:, full.index_dict[var]] = FFX_all[:, idx_dict_all['neu_%d'%neuron]]
        elif 'lfp_' in var:
            FFX[:, full.index_dict[var]] = fullX[:, idx_dict[var]]
        else:
            FFX[:, full.index_dict[var]] = FFX_all[:, idx_dict_all[var]]

    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)
    eta = np.dot(FFX, full.beta)
    mu = family.link.inverse(eta)

    # no coupling
    shape = 1
    for var in full.index_dict.keys():
        if 'neu_' in var:
            continue
        shape += len(full.index_dict[var])

    FFX = np.ones((fullX.shape[0], shape))
    bbeta = np.zeros(shape)
    bbeta[0] = full.beta[0]
    cc = 1
    for var in full.index_dict.keys():
        if 'neu_' in var:
            continue
        ii = len(full.index_dict[var])
        if var == 'spike_hist':
            FFX[:, cc:cc+ii] = FFX_all[:, idx_dict_all['neu_%d'%neuron]]
            bbeta[cc:cc+ii] = full.beta[full.index_dict[var]]
        elif 'lfp_' in var:
            FFX[:, cc:cc + ii] = fullX[:, idx_dict[var]]
            bbeta[cc:cc + ii] = full.beta[full.index_dict[var]]
        else:
            FFX[:, cc:cc+ii] = FFX_all[:,idx_dict_all[var]]
            bbeta[cc:cc + ii] = full.beta[full.index_dict[var]]
        cc += ii

    eta = np.dot(FFX, bbeta)
    mu_nocoup = family.link.inverse(eta)

    # no coupling / nohist
    shape = 1
    for var in full.index_dict.keys():
        if 'neu_' in var:
            continue
        if 'spike_hist' in var:
            continue

        shape += len(full.index_dict[var])

    FFX = np.ones((fullX.shape[0], shape))
    bbeta = np.zeros(shape)
    bbeta[0] = full.beta[0]
    cc = 1
    for var in full.index_dict.keys():
        if 'neu_' in var:
            continue
        if 'spike_hist' in var:
            continue

        ii = len(full.index_dict[var])
        if var == 'spike_hist':
            FFX[:, cc:cc+ii] = FFX_all[:, idx_dict_all['neu_%d'%neuron]]
            bbeta[cc:cc+ii] = full.beta[full.index_dict[var]]
        elif 'lfp_' in var:
            FFX[:, cc:cc + ii] = fullX[:, idx_dict[var]]
            bbeta[cc:cc + ii] = full.beta[full.index_dict[var]]
        else:
            FFX[:, cc:cc+ii] = FFX_all[:,idx_dict_all[var]]
            bbeta[cc:cc + ii] = full.beta[full.index_dict[var]]
        cc += ii

    eta = np.dot(FFX, bbeta)
    mu_nohist = family.link.inverse(eta)


    # no  coupling / nohist / latent
    shape = 1
    for var in full.index_dict.keys():
        if 'neu_' in var:
            continue
        if 'spike_hist' in var:
            continue
        if var == 'rad_target' or var == 'ang_target' or var == 'rad_path' or var == 'ang_path':
            continue

        shape += len(full.index_dict[var])

    FFX = np.ones((fullX.shape[0], shape))
    bbeta = np.zeros(shape)
    bbeta[0] = full.beta[0]
    cc = 1
    for var in full.index_dict.keys():
        if 'neu_' in var:
            continue
        if 'spike_hist' in var:
            continue
        if var == 'rad_target' or var == 'ang_target' or var == 'rad_path' or var == 'ang_path':
            continue

        ii = len(full.index_dict[var])
        if var == 'spike_hist':
            FFX[:, cc:cc+ii] = FFX_all[:, idx_dict_all['neu_%d'%neuron]]
            bbeta[cc:cc+ii] = full.beta[full.index_dict[var]]
        elif 'lfp_' in var:
            FFX[:, cc:cc + ii] = fullX[:, idx_dict[var]]
            bbeta[cc:cc + ii] = full.beta[full.index_dict[var]]
        else:
            FFX[:, cc:cc+ii] = FFX_all[:,idx_dict_all[var]]
            bbeta[cc:cc + ii] = full.beta[full.index_dict[var]]
        cc += ii

    eta = np.dot(FFX, bbeta)
    mu_nointernal = family.link.inverse(eta)

    meanFr[:,cnt_neu] = mu
    meanFr_noInt[:,cnt_neu] = mu_nointernal
    meanFr_noHist[:,cnt_neu] = mu_nohist
    meanFr_noCP[:,cnt_neu] = mu_nocoup
    pseudo_r2[cnt_neu] = dd['p_r2_coupling_full']
    del fullX, idx_dict
    cnt_neu += 1

rm_neuron = np.array(rm_neuron,dtype=int)
keep_neu = np.ones(neuron_fit.shape[0],dtype=bool)
keep_neu[rm_neuron] = False
meanFr = meanFr[:,keep_neu]
pseudo_r2 = pseudo_r2[keep_neu]
neuron_fit = neuron_fit[keep_neu]

np.savez('meanFR_%s_%s.npz'%(session,area),meanFr=meanFr,meanFr_noHist=meanFr_noHist,meanFr_noCP=meanFr_noCP,meanFr_noInt=meanFr_noInt,pseudo_r2=pseudo_r2,neuron_fit=neuron_fit)


