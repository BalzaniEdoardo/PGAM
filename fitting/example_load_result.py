import numpy as np
import sys, os, dill
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
from utils_loading import unpack_preproc_data, add_smooth
from GAM_library import *
import pandas as pd


plot_reduced = True

# load the full and reduced model
with open(os.path.join(folder_name,'gam_m51s120/gam_fit_m51s120_c4_replay_0.0000.dill'),'rb') as fh:
    dict_res = dill.load(fh)

reduced_model = dict_res['reduced']
full_model = dict_res['full']
session = 'm51s120'

# sigificance of variables
# I am using pandas only for a nicer print of the table
# I consider significant a response with a p-val below the threshold of 0.001
print(pd.DataFrame(reduced_model.covariate_significance))

# this is a dictionary containing a lot of information about the smooth resp. functions
smooth_info = reduced_model.smooth_info
# for example spline interp knots
knots_eye = smooth_info['eye_hori']['knots']

# plot full or reduced

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

if not plot_reduced:
    gam_res = full_model
else:
    gam_res = reduced_model
FLOAT_EPS = np.finfo(float).eps
import matplotlib.pylab as plt

var_list = gam_res.var_list

pvals = np.clip(gam_res.covariate_significance['p-val'], FLOAT_EPS, np.inf)
dropvar = np.log(pvals) > np.mean(np.log(pvals)) + 1.5 * np.std(np.log(pvals))
dropvar = pvals > 0.001
drop_names = gam_res.covariate_significance['covariate'][dropvar]
fig = plt.figure(figsize=(14, 8))
cc = 0
cc_plot = 1
for var in gam_res.var_list:
    if not np.sum(np.array(gam_res.var_list) == var) and var != 'spike_hist':
        cc += 1
        continue
    print('plotting var', var)

    ax = plt.subplot(5, 5, cc_plot)

    if var == 'spike_hist':
        pass
    else:
        # max_x, min_x = X[var].max(), X[var].min()
        try:
            min_x, max_x = dict_xlim[var]
        except:
            min_x,max_x = gam_res.smooth_info[var]['knots'][0][0],gam_res.smooth_info[var]['knots'][0][-1]

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

    if (var == 'spike_hist') or (var.startswith('neu_')):
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
plt.tight_layout()