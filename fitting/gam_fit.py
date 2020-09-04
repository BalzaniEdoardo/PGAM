## script to control that kernel are not forced to zero by the algorithm
import numpy as np
import sys, os, dill
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
from utils_loading import unpack_preproc_data, add_smooth
from GAM_library import *
from time import perf_counter
import statsmodels.api as sm
from basis_set_param_per_session import *
from knots_util import *
from path_class import get_paths_class

user_paths = get_paths_class()


tot_fits = 1
plot_res = False
fit_fully_coupled = True
use_k_fold = True
reducedAdaptive = False
num_folds = 10

print('folder name')
print(folder_name)
print(' ')

use_fisher_scoring = False

# load the data Kaushik passed me
try:
    folder_name = user_paths.get_path('data_hpc')
    sv_folder_base = user_paths.get_path('code_hpc')
    fhName = os.path.join(folder_name, sys.argv[2])
    dat = np.load(os.path.join(folder_name, fhName), allow_pickle=True)
except:
    print('EXCEPTION RAISED')
    folder_name = ''
    sv_folder_base = ''
    fhName = os.path.join(user_paths.get_path('local_concat'),'m53s91.npz')
    # fhName = '/Users/edoardo/Downloads/PPC+PFC+MST/m53s109.npz'
    if fhName.endswith('.mat'):
        dat = loadmat(fhName)
    elif fhName.endswith('.npy'):
        dat = np.load(fhName, allow_pickle=True).all()
    elif fhName.endswith('.npz'):
        dat = np.load(fhName, allow_pickle=True)

print('loaded ', fhName)

par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
            'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin', 'cR', 'presence_rate', 'isiV',
            'unit_type']

(Xt, yt, lfp_beta, lfp_alpha, lfp_theta, var_names, trial_type,
 trial_idx, brain_area, pre_trial_dur, pre_trial_dur, time_bin,
 cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type) = unpack_preproc_data(fhName, par_list)

# get the unit to include as input covariates
cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate_filter > 0.9
isi_v_filter = isi_v_filter < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

# unit number according to matlab indexing
neuron_keep = np.arange(1, yt.shape[1] + 1)[combine_filter]

# extract the condition to be filtered and the triL
session = os.path.basename(fhName).split('.')[0]

try:  # IF CLUSTER JOB IS RUNNING
    JOB = int(sys.argv[1]) - 1
    list_condition = np.load(os.path.join(user_paths.get_path('code_hpc'),'condiiton_list_%s.npy' % session))
    neuron_list = list_condition[JOB:JOB + tot_fits]['neuron']
    cond_type_list = list_condition[JOB:JOB + tot_fits]['condition']
    cond_value_list = list_condition[JOB:JOB + tot_fits]['value']
    pop_size_max = yt.shape[1]
except Exception as ex:
    JOB = 1
    list_condition = np.load(os.path.join(os.path.join(main_dir,'preprocessing_pipeline'),
        'condiiton_list_%s.npy' % session))
    neuron_list = list_condition[JOB:JOB + tot_fits]['neuron']
    cond_type_list = list_condition[JOB:JOB + tot_fits]['condition']
    cond_value_list = list_condition[JOB:JOB + tot_fits]['value']
    pop_size_max = yt.shape[1]


print('list condition', cond_type_list, cond_value_list)

cont_names = np.array(
    ['rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target', 'phase', 'lfp_beta', 'lfp_theta',
     'lfp_alpha', 'eye_vert', 'eye_hori'])
event_names = np.array(['t_move', 't_flyOFF', 't_stop', 't_reward', 'spike_hist'])

##  truncate ang dist
ang_idx = np.where(np.array(var_names) == 'ang_target')[0][0]
Xt[np.abs(Xt[:, ang_idx]) > 50, ang_idx] = np.nan

# cycle over all fits
for idx_subjob in range(len(neuron_list)):
    # select the proper condition and neuron to be fitted
    neuron = neuron_list[idx_subjob]
    cond_type = cond_type_list[idx_subjob]
    cond_value = cond_value_list[idx_subjob]

    # filter condition
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
    if cond_type != 'replay':
        keep = []
        for ii in idx_subselect:
            keep = np.hstack((keep, np.where(trial_idx == ii)[0]))
            if np.sum(trial_idx == ii) == 0:
                raise ValueError
    else:
        keep = []
        for ii in idx_subselect:
            keep = np.hstack((keep, np.where(trial_idx == ii)[0]))

    print(' condition', cond_type, cond_value)

    keep = np.array(keep, dtype=int)
    trial_idx = trial_idx[keep]
    print('FITTING NEURON %d\n' % neuron)

    # create the smooth handler object
    sm_handler = smooths_handler()
    var_use = []
    count_additional_pen = 0
    for var in np.hstack((var_names, ['spike_hist'])):
        # for now skip
        if var in ['hand_vel1', 'hand_vel2']:
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
            x = np.hstack(([0], tmpy[:-1]))

        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep, cc]

        knots = knots_by_session(x, session, var, basis_info)
        sm_handler = add_smooth(sm_handler, x, var, knots, session, trial_idx, time_bin=time_bin, lam=50)


    # add coupling
    if fit_fully_coupled:
        for neu_num in range(pop_size_max):
            # continue if same unit or if unit is of poor quality
            if neu_num == (neuron - 1):
                continue
            if not any(neu_num == (neuron_keep - 1)):
                continue

            tmpy = yt[keep, neu_num]
            x = np.hstack(([0], tmpy[:-1]))

            print('adding neuron %d' % (neu_num + 1))
            knots = knots_by_session(x, session, 'spike_hist', basis_info)
            sm_handler = add_smooth(sm_handler, x, 'neu_%d' % (neu_num + 1), knots, session, trial_idx, time_bin=time_bin, lam=50)


    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)

    gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, yt[keep, neuron - 1], poissFam,
                                       fisher_scoring=use_fisher_scoring)

    t0 = perf_counter()
    full, reduced = gam_model.fit_full_and_reduced(sm_handler.smooths_var, th_pval=0.001, method='L-BFGS-B', tol=1e-8,
                                                   conv_criteria='gcv',
                                                   max_iter=10000, gcv_sel_tol=10 ** -13, random_init=False,
                                                   use_dgcv=True, initial_smooths_guess=False,
                                                   fit_initial_beta=True, pseudoR2_per_variable=True,
                                                   trial_num_vec=trial_idx, k_fold=use_k_fold, fold_num=num_folds,
                                                   reducedAdaptive=reducedAdaptive)
    t1 = perf_counter()
    print('\n\n=========================\nFIT TIME: %f sec\n=========================\n\n' % (t1 - t0))
    gam_res = {}
    gam_res['full'] = full
    gam_res['reduced'] = reduced

    sv_folder = os.path.join(sv_folder_base, 'gam_%s/' % session)
    if not os.path.exists(sv_folder):
        os.mkdir(sv_folder)

    with open(os.path.join(sv_folder, 'gam_fit_%s_c%d_%s_%.4f.dill' % (session, neuron, cond_type, cond_value)),
              "wb") as dill_file:
        dill_file.write(dill.dumps(gam_res))

    if plot_res:
        dict_xlim = {'rad_vel': (0., 200),
                     'ang_vel': (-90, 90),
                     'rad_path': (0, 400),
                     'ang_path': (-90, 90),
                     'hand_vel1': (-100., 100),
                     'hand_vel2': (-100, 100),
                     'phase': (-np.pi, np.pi),
                     't_move': (-0.36, 0.36),
                     't_flyOFF': (-0.36, 0.36),
                     't_stop': (-0.36, 0.36),
                     't_reward': (-0.36, 0.36)}

        gam_res = reduced
        FLOAT_EPS = np.finfo(float).eps
        import matplotlib.pylab as plt

        var_list = gam_res.var_list

        pvals = np.clip(gam_res.covariate_significance['p-val'], FLOAT_EPS, np.inf)
        dropvar = np.log(pvals) > np.mean(np.log(pvals)) + 1.5 * np.std(np.log(pvals))
        dropvar = pvals > 0.001
        drop_names = gam_res.covariate_significance['covariate'][dropvar]
        fig = plt.figure(figsize=(14, 8))
        plt.suptitle('%s - neuron %d' % (session, neuron))
        cc = 0
        cc_plot = 1
        for var in np.hstack((var_names, ['spike_hist'])):
            if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
                cc += 1
                continue
            print('plotting var', var)

            ax = plt.subplot(5, 4, cc_plot)

            if var == 'spike_hist':
                pass
            else:
                # max_x, min_x = X[var].max(), X[var].min()
                try:
                    min_x, max_x = dict_xlim[var]
                except:
                    min_x = np.nanpercentile(Xt[:, cc], 5)
                    max_x = np.nanpercentile(Xt[:, cc], 95)

            if gam_res.smooth_info[var]['is_temporal_kernel']:

                dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
                knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
                ord_ = gam_res.smooth_info[var]['ord']
                idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

                impulse = np.zeros(dim_kern)
                impulse[(dim_kern - 1) // 2] = 1
                xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
                fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99, trial_idx=None)
                if var != 'spike_hist':
                    xx = xx[idx_select][1:-1]
                    fX = fX[idx_select][1:-1]
                    fX_p_ci = fX_p_ci[idx_select][1:-1]
                    fX_m_ci = fX_m_ci[idx_select][1:-1]
                else:
                    xx = xx[:(-ord_ - 1)]
                    fX = fX[:(-ord_ - 1)]
                    fX_p_ci = fX_p_ci[:(-ord_ - 1)]
                    fX_m_ci = fX_m_ci[:(-ord_ - 1)]

            else:
                knots = gam_res.smooth_info[var]['knots']
                knots_sort = np.unique(knots[0])
                knots_sort.sort()
                xx = (knots_sort[1:] + knots_sort[:-1]) * 0.5

                fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)
            if np.sum(drop_names == var):
                label = var
            else:
                label = var

            if var == 'spike_hist':
                iend = xx.shape[0] // 2

                print('set spike_hist')
                fX = fX[iend + 1:][::-1]
                fX_p_ci = fX_p_ci[iend + 1:][::-1]
                fX_m_ci = fX_m_ci[iend + 1:][::-1]
                plt.plot(xx[:iend], fX, ls='-', color='k', label=label)
                plt.fill_between(xx[:iend], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
            else:
                plt.plot(xx, fX, ls='-', color='k', label=label)
                plt.fill_between(xx, fX_m_ci, fX_p_ci, color='k', alpha=0.4)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend()

            cc += 1
            cc_plot += 1
