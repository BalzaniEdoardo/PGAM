## script to control that kernel are not forced to zero by the algorithm
import sys,os
# sys.path.append('/Users/jean-paulnoel/Documents/Savin-Angelaki/GAM_Repo/GAM_library')
# sys.path.append( '/Users/jean-paulnoel/Documents/Savin-Angelaki/GAM_Repo/preprocessing_pipeline/util_preproc/')
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from copy import deepcopy
from time import perf_counter
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm

# from get_knots_per_cond_continuous import *
from basis_set_param_per_session import *
from knots_util import *
from utils_loading import *
from path_class import get_paths_class

user_paths = get_paths_class()

plt.close('all')
all_var_fit = False
reload = True
k_fold = False
plot_res = True
fit_neuron = False
session = 'm91s2'
analyze_unit = 3
<<<<<<< HEAD
sbfld = 'PPC+PFC+MST'
var = 'ang_target'
skip_var = ''
WLS_solver = 'negative_weights'
send = False

folder_name = os.path.dirname(os.path.realpath(__file__))
print('folder name')
print(folder_name)
print(' ')

# use
use_fisher_scoring = False

fhName = os.path.join(user_paths.get_path('local_concat'),'%s.npz'%(session))
cont_names = np.array(['rad_vel', 'ang_vel', 'rad_path', 'ang_path', 'rad_target', 'ang_target', 'phase','lfp_beta','lfp_theta','lfp_alpha', 'eye_vert', 'eye_hori'])
event_names = np.array(['t_move', 't_flyOFF', 't_stop', 't_reward','spike_hist'])
if reload:
    par_list = ['Xt', 'Yt', 'lfp_beta', 'lfp_alpha', 'lfp_theta', 'var_names', 'info_trial',
                'trial_idx', 'brain_area', 'pre_trial_dur', 'post_trial_dur', 'time_bin']

    (Xt,yt,lfp_beta,lfp_alpha,lfp_theta,var_names,trial_type,
            trial_idx,brain_area,pre_trial_dur,pre_trial_dur,time_bin) = unpack_preproc_data(fhName,par_list)
    cond_type = 'all'
    cond_value = 1

    ##  truncate ang dist
    ang_idx = np.where(np.array(var_names) == 'ang_target')[0][0]
    Xt[np.abs(Xt[:, ang_idx]) > 50, ang_idx] = np.nan
    del par_list



# filter condition
idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]
keep = []
for ii in idx_subselect:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))
    if np.sum(trial_idx == ii) == 0:
        raise ValueError

keep = np.array(keep, dtype=int)
trial_idx = trial_idx[keep]



# generate basis spline for given knots
fit_var = deepcopy(var)
if not var in ['lfp_beta','phase','lfp_alpha','lfp_theta','spike_hist']:
    cc = np.where(var_names == var)[0][0]
    x = Xt[keep,cc]
else:
    if var == 'lfp_beta':
        x = lfp_beta[keep,analyze_unit-1]
    elif var == 'lfp_alpha':
        x = lfp_alpha[keep, analyze_unit - 1]
    elif var == 'lfp_theta':
        x = lfp_theta[keep, analyze_unit - 1]
    elif var == 'phase':
        x = lfp_phase[keep, analyze_unit - 1]
    elif var == 'spike_hist':
        tmpy = yt[keep, analyze_unit - 1]
        x = np.hstack(([0], tmpy[:-1]))

# compute knots for b-sline
knots = knots_by_session(x,session,var,basis_info)
sm_handler = smooths_handler()
sm_handler = add_smooth(sm_handler, x, var, knots, session,trial_idx, time_bin=time_bin)


# plot knots location
plt.figure()
plt.hist(x,bins=100,range=(np.nanpercentile(x,1),np.nanpercentile(x,99.)))
plt.plot(knots,[0]*knots.shape[0],'-or')
plt.title(var)
plt.ylabel('counts')

# plot basis set
xx = np.linspace(knots[0],knots[-1],100)
fX = basisAndPenalty([xx], [knots], is_cyclic=[basis_info[session][var]['is_cyclic']], ord=4,
                     penalty_type='der', xmin=[knots[0]], xmax=[knots[-1]], der=2)[0]
fX = fX.toarray()
plt.figure()
for ii in range(fX.shape[1]):
    plt.plot(xx,fX[:,ii])
plt.plot(knots,[0]*knots.shape[0],'-or')


if fit_neuron:
    # filter condition
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]

    print('FITTING NEURON %d\n'%analyze_unit)

    var_use = []
    count_additional_pen = 0
    if all_var_fit:
        list_fit = np.hstack((var_names,['spike_hist']))
    else:
        list_fit = np.hstack((fit_var, ['spike_hist']))

    for var in np.unique(list_fit):
        if var == skip_var:
            continue
        if var in sm_handler.smooths_dict.keys():
            var_use += [var]
            count_additional_pen =count_additional_pen + (basis_info[session][var]['penalty_type'] == 'der')
            continue
        if var in ['hand_vel1','hand_vel2']:
            continue

        # if count_additional_pen > 4:
        #     break

        if var in ['phase','lfp_beta','lfp_alpha','lfp_theta']:
            is_cyclic = True
        else:
            is_cyclic = False



        if var == 'lfp_theta':
            x = lfp_theta[keep,analyze_unit-1]

        elif var == 'lfp_beta':
            x = lfp_beta[keep,analyze_unit-1]

        elif var == 'lfp_alpha':
            x = lfp_alpha[keep,analyze_unit-1]

        elif var == 'spike_hist':
            tmpy = yt[keep,analyze_unit-1]
            x = np.hstack(([0],tmpy[:-1]))

        else:
            cc = np.where(var_names == var)[0][0]
            x = Xt[keep,cc]

        knots = knots_by_session(x,session,var,basis_info)
        sm_handler = add_smooth(sm_handler, x, var, knots, session,trial_idx, time_bin=time_bin)
        var_use += [var]

    # create model
    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)

    gam_model = general_additive_model(sm_handler, var_use, yt[keep, analyze_unit-1], poissFam, fisher_scoring=use_fisher_scoring)

    t0 = perf_counter()
    full,reduced = gam_model.fit_full_and_reduced(var_use,  method='L-BFGS-B', tol=1e-8, conv_criteria='gcv',
                         max_iter=10000,gcv_sel_tol=10**-13,random_init=False,
                         use_dgcv=True,initial_smooths_guess=False,
                         fit_initial_beta=True,th_pval=0.001,
                         trial_num_vec=trial_idx,k_fold = k_fold,fold_num=5)
    if plot_res:
        dict_xlim = {'rad_vel': (0., 200),
                     'ang_vel': (-90, 90),
                     'rad_path': (0, 400),
                     'ang_path': (-90, 90),
                     'hand_vel1': (-100., 100),
                     'hand_vel2': (-100, 100),
                     'phase': (-np.pi, np.pi),
                     'lfp_beta': (-np.pi, np.pi),
                     'lfp_alpha': (-np.pi, np.pi),
                     'lfp_theta':(-np.pi, np.pi),
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
        fig = plt.figure(figsize=(14,8))
        plt.suptitle('%s - neuron %d - %s'%(session,analyze_unit,brain_area[analyze_unit-1]))
        cc = 0
        cc_plot =1
        for var in np.hstack((var_names,['spike_hist'])):
            if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
                cc += 1
                continue
            if var == 'spike_hist' and not var in gam_res.var_list:
                continue
            print('plotting var',var)

            ax = plt.subplot(4,4,cc_plot)

            if var == 'spike_hist':
                pass

            else:
                try:
                    min_x, max_x = dict_xlim[var]
                except:
                    min_x = np.nanpercentile(Xt[:,cc],5)
                    max_x = np.nanpercentile(Xt[:, cc], 95)


            if gam_res.smooth_info[var]['is_temporal_kernel']:

                dim_kern = gam_res.smooth_info[var]['basis_kernel'].shape[0]
                knots_num = gam_res.smooth_info[var]['knots'][0].shape[0]
                ord_ =gam_res.smooth_info[var]['ord']
                idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

                impulse = np.zeros(dim_kern)
                impulse[(dim_kern - 1) // 2] = 1
                xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
                fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([impulse], var, perc=0.99,trial_idx=None)
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
                if var in ['rad_path','rad_target']:
                    xx = np.linspace(knots[0][0], knots[0][-1], 100)
                else:
                    xx = (knots_sort[1:] + knots_sort[:-1]) * 0.5
                    xx = np.linspace(xx[0],xx[-1],100)

                fX, fX_p_ci, fX_m_ci = gam_res.smooth_compute([xx], var, perc=0.99)
            if np.sum(drop_names==var):
                label = var
            else:
                label = var

            if var == 'spike_hist':
                iend = xx.shape[0]//2
                print('set spike_hist')
                if iend == (fX.shape[0] - 1)//2:

                    fX = fX[iend + 1:][::-1]
                    fX_p_ci = fX_p_ci[iend + 1:][::-1]
                    fX_m_ci = fX_m_ci[iend + 1:][::-1]
                    plt.plot(xx[:iend], fX, ls='-', color='k', label=label)
                    plt.fill_between(xx[:iend], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
                else:
                    fX = fX[iend + 1:][::-1]
                    fX_p_ci = fX_p_ci[iend + 1:][::-1]
                    fX_m_ci = fX_m_ci[iend + 1:][::-1]
                    plt.plot(xx[:iend-1], fX, ls='-', color='k', label=label)
                    plt.fill_between(xx[:iend-1], fX_m_ci, fX_p_ci, color='k', alpha=0.4)
            else:
                plt.plot(xx, fX , ls='-', color='k', label=label)
                plt.fill_between(xx,fX_m_ci,fX_p_ci,color='k',alpha=0.4)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend()



            cc+=1
            cc_plot+=1

if send:
    print('sending to server...')
    send_script = os.path.join(user_paths.get_path('basis_info_local'),'basis_set_param_per_session.py')
    dest_folder = user_paths.get_path('basis_info_cluster')
    os.system('sshpass -p "%s" scp %s jpn5@prince.hpc.nyu.edu:%s' % ('savin123!', send_script,dest_folder))
