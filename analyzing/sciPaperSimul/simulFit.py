import numpy as np
import matplotlib.pylab as plt
import dill
import os,re,sys
from copy import deepcopy
import pandas as pd
import seaborn as sbn
import statsmodels.api as sm

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append(os.path.join('/Users/edoardo/Work/Code/GAM_code/', 'preprocessing_pipeline/util_preproc'))

from GAM_library import *
from utils_loading import unpack_preproc_data
from knots_constructor import *


fld_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info/'
lst_done = os.listdir(fld_file)
# mutual_info_and_tunHz_m53s42.dill

first = True
for fh in lst_done:
    if not re.match('^mutual_info_and_tunHz_m\d+s\d+.dill$',fh):
        continue

    with open(os.path.join(fld_file,fh),'rb') as fh:
        res = dill.load(fh)
        mi = res['mutual_info']
        tun = res['tuning_Hz']
    if first:
        mutual_info = deepcopy(mi)
        tuning = deepcopy(tun)
        first = False
    else:
        mutual_info = np.hstack((mutual_info,mi))
        tuning = np.hstack((tuning, tun))

# filter only density manip
keep_sess = np.unique(mutual_info['session'][mutual_info['manipulation_type']=='density'])
filt_sess = np.zeros(mutual_info.shape,dtype=bool)
for sess in keep_sess:
    filt_sess[mutual_info['session']==sess] = True

dprime_vec = np.zeros(tuning.shape)
cc = 0
for tun in tuning:
    dprime_vec[cc] = np.mean(tun['y_raw'] - tun['y_model'])/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))
    cc += 1


filter = (np.abs(dprime_vec)<0.1) & (mutual_info['manipulation_type'] == 'all') & (mutual_info['pseudo-r2'] > 0.005) &\
         filt_sess & (~np.isnan(mutual_info['mutual_info']))
mutual_info = mutual_info[filter]
tuning = tuning[filter]
dprime_vec = dprime_vec[filter]

sel = (mutual_info['brain_area']=='MST') & (mutual_info['significance']) & (mutual_info['variable']=='rad_target')
mutual_info = mutual_info[sel]
tuning = tuning[sel]
dprime_vec = dprime_vec[sel]
srt_idx = np.argsort(mutual_info, order='mutual_info')

# open a fit



for k in range(9):

    session = mutual_info['session'][srt_idx[-k-1]]
    neuron = mutual_info['neuron'][srt_idx[-k-1]]
    res = dill.load(open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_all_1.0000.dill'%(session,session,neuron),'rb'))

    reduced = res['reduced']
    dist_targ = np.linspace(0,400,100)
    fX,fXp,fXm = reduced.smooth_compute([dist_targ],'rad_target')

    plt.subplot(3,3,k+1)
    plt.title('%s - c%d'%(session,neuron))
    plt.plot(dist_targ,fX)
    plt.fill_between(dist_targ,fXm,fXp,alpha=0.5)
plt.tight_layout()

list_use = [('m53s114',5),('m53s93',1),('m53s93',7),('m44s206',13),('m53s115',10),('m44s212',6),
            ('m44s185',6)]

tunRes = np.zeros((len(list_use),3,100))
tunResWO = np.zeros((len(list_use),3,100))
groundTruth = np.zeros((len(list_use),100))
dtype_dict={'names':('session','neuron','variable','significant','fit_type'),
            'formats':('U20',int,'U20',bool,'U20')}
fit_full_table = np.zeros(0,dtype=dtype_dict)
xx_dist = np.linspace(0,400,100)
cc_fit = 0
cc_fit2 = 0

for session,neuron in list_use:

    res = dill.load(open(
        '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_all_1.0000.dill' % (
        session, session, neuron), 'rb'))

    reduced = res['reduced']

    fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session
    par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
                'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
                'unit_type', 'channel_id', 'electrode_id', 'cluster_id']
    (Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
     trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
     cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type,
     channel_id, electrode_id, cluster_id) = unpack_preproc_data(fhName, par_list)

    dict_xlims = {}

    # get the unit to include as input covariates
    cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
    presence_rate_filter = presence_rate_filter > 0.9
    isi_v_filter = isi_v_filter < 0.2
    combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

    idx_subselect = np.where(trial_type['all'] == True)[0]
    keep = []
    for ii in idx_subselect:
        keep = np.hstack((keep, np.where(trial_idx == ii)[0]))
    keep = np.array(keep,dtype=int)

    sm_handler = smooths_handler()

    for var in reduced.var_list:

        if var == 'hand_vel1' or var == 'hand_vel2':
            continue

        if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
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

        elif var.startswith('neu_'):
            other = int(var.split('_')[1])
            tmpy = yt[keep, other - 1]
            x = tmpy

            if brain_area[neuron - 1] == brain_area[other - 1]:
                filt_len = 'short'
            else:
                filt_len = 'long'


        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep, cc]

        if var.startswith('neu_'):

            knots, x_trans, include_var, is_cyclic, order, \
            kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
                knots_cerate(x, 'spike_hist', session, hand_vel_temp=False, hist_filt_dur=filt_len,
                             exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                             condition=None)
        else:

            knots, x_trans, include_var, is_cyclic, order, \
            kernel_len, kernel_direction, is_temporal_kernel, penalty_type, der = \
                knots_cerate(x, var, session, hand_vel_temp=False, hist_filt_dur='short',
                             exclude_eye_position=['m44s213', 'm53s133', 'm53s134', 'm53s105', 'm53s182'],
                             condition=None)


        if not var.startswith('t_') and var != 'spike_hist' and (not var.startswith('neu_')):
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
                                  trial_idx=trial_idx, time_bin=time_bin,
                                  is_temporal_kernel=is_temporal_kernel,
                                  kernel_length=kernel_len,
                                  kernel_direction=kernel_direction, ord_AD=3, ad_knots=4,
                                  repeat_extreme_knots=False)


    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)

    modelX,idx_dict = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)
    mu = np.exp(np.dot(modelX,reduced.beta))
    spk = np.random.poisson(mu)

    gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, spk, poissFam,
                                       fisher_scoring=False)

    full_coupling, reduced_coupling = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001,
                                                                     method='L-BFGS-B', tol=1e-8,
                                                                     conv_criteria='gcv',
                                                                     max_iter=1000, gcv_sel_tol=10 ** -13,
                                                                     random_init=False,
                                                                     use_dgcv=True, initial_smooths_guess=False,
                                                                     fit_initial_beta=True, pseudoR2_per_variable=True,
                                                                     trial_num_vec=trial_idx, k_fold=False,
                                                                     fold_num=None,
                                                                     reducedAdaptive=False, compute_MI=False,
                                                                     perform_PQL=True)

    # xx = np.linspace(0,400,100)
    fX,_,_ = reduced.smooth_compute([xx_dist],'rad_target')
    fX2, fXm, fXp = full_coupling.smooth_compute([xx_dist], 'rad_target')


    tunRes[cc_fit,0] = fXm
    tunRes[cc_fit, 1] = fX2
    tunRes[cc_fit, 2] = fXp

    groundTruth[cc_fit] = fX

    tmp = np.zeros(len(full_coupling.var_list), dtype=dtype_dict)
    fit_tp = 'all_vars'
    cc_tmp = 0
    for var in full_coupling.var_list:
        sign = var in reduced_coupling.var_list
        tmp['neuron'][cc_tmp] = neuron
        tmp['session'][cc_tmp] = session
        tmp['variable'][cc_tmp] = var
        tmp['significant'][cc_tmp] = sign
        tmp['fit_type'][cc_tmp] = fit_tp
        cc_tmp +=1

    fit_full_table = np.hstack((fit_full_table,tmp))


    iidx = [0]
    for var in full_coupling.var_list:
        if var == 'rad_target':
            continue
        iidx = np.hstack((iidx,full_coupling.index_dict[var]))
    iidx =  np.array(iidx,dtype=int)
    mu = np.exp(np.dot(modelX[:,iidx], reduced.beta[iidx]))
    spk = np.random.poisson(mu)

    gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, spk, poissFam,
                                       fisher_scoring=False)

    full_coupling, reduced_coupling = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001,
                                                                     method='L-BFGS-B', tol=1e-8,
                                                                     conv_criteria='gcv',
                                                                     max_iter=1000, gcv_sel_tol=10 ** -13,
                                                                     random_init=False,
                                                                     use_dgcv=True, initial_smooths_guess=False,
                                                                     fit_initial_beta=True, pseudoR2_per_variable=True,
                                                                     trial_num_vec=trial_idx, k_fold=False,
                                                                     fold_num=None,
                                                                     reducedAdaptive=False, compute_MI=False,
                                                                     perform_PQL=True)


    fX2, fXm, fXp = full_coupling.smooth_compute([xx_dist], 'rad_target')

    tunResWO[cc_fit2, 0] = fXm
    tunResWO[cc_fit2, 1] = fX2
    tunResWO[cc_fit2, 2] = fXp


    tmp = np.zeros(len(full_coupling.var_list), dtype=dtype_dict)
    fit_tp = 'wo_rad_target'
    cc_tmp = 0
    for var in full_coupling.var_list:
        sign = var in reduced_coupling.var_list
        tmp['neuron'][cc_tmp] = neuron
        tmp['session'][cc_tmp] = session
        tmp['variable'][cc_tmp] = var
        tmp['significant'][cc_tmp] = sign
        tmp['fit_type'][cc_tmp] = fit_tp
        cc_tmp += 1

    fit_full_table = np.hstack((fit_full_table, tmp))

    cc_fit+=1
    cc_fit2+=1

np.savez('res_simul.npz',table=fit_full_table,tunRes=tunRes,tunResWO=tunResWO,groundTruth=groundTruth)

cc_plt = 1
plt.figure(figsize=(13.21,  3.4 ))
plt.suptitle('simulation MST units')
for k in range(7):
    plt.subplot(2,7,cc_plt)
    if cc_plt==1:
        plt.ylabel('rad dist included')

    plt.plot(groundTruth[k],'k')
    p,=plt.plot(tunRes[k, 1])
    plt.fill_between(np.arange(0,tunRes.shape[2]),tunRes[k, 0],tunRes[k, 2],color=p.get_color(),alpha=0.4)
    ylim = plt.ylim()
    plt.subplot(2,7, cc_plt+7)

    if cc_plt==1:
        plt.ylabel('rad dist not included')

    plt.plot([0]*tunRes.shape[2], 'k')
    p, = plt.plot(tunResWO[k, 1])
    plt.fill_between(np.arange(0, tunResWO.shape[2]), tunResWO[k, 0], tunResWO[k, 2], color=p.get_color(),alpha=0.4)
    plt.ylim(ylim)

    cc_plt+=1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
