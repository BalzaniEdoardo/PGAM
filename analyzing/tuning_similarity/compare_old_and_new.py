#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:40:41 2021

@author: edoardo
"""
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import matplotlib.pylab as plt
import dill,sys,os
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from seaborn import *
from GAM_library import *
from scipy.integrate import simps
from spectral_clustering import *
from basis_set_param_per_session import *
from spline_basis_toolbox import *
from scipy.cluster.hierarchy import linkage,dendrogram

range_dict = {'rad_vel': (-0.00886941459029913, 178.91336059570312),
 'ang_vel': (-26.109495162963867, 39.09432571411133),
 'rad_path': (0.0, 327.19013946533204),
 'ang_path': (-52.81943088531493, 22.953210849761962),
 'rad_target': (25.766419427171442, 372.1482421866442),
 'ang_target': (-38.32807377216164, 43.287638230488355),
 'eye_vert': (-2, 2),
 'eye_hori': (-2,2),
 't_move': (-165.0, 165.0),
 't_flyOFF': (-327.0, 327.0),
 't_stop': (-165.0, 165.0),
 't_reward': (-165.0, 165.0),
 'spike_hist': (1e-06, 5.0),
 'lfp_beta': (-3.141592653589793, 3.141592653589793),
 'lfp_theta': (-3.141592653589793, 3.141592653589793),
 'lfp_alpha': (-3.141592653589793, 3.141592653589793),
  'rad_acc':(-800,800),
  'ang_acc':(-100,100)}


old_dat = np.load('/Users/edoardo/Work/Code/Angelaki-Savin/NIPS_Analysis/coupling_x_similarity/pairwise_L2_dist.npz',allow_pickle=True)
info_dict_old = old_dat['info_dict'].all()
beta_dict_old = old_dat['beta_dict'].all()
npz_old = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/oldConcatExample/m53s98.npz',allow_pickle=True)
pairwise_dist_old = old_dat['pairwise_dist'].all()
var_list_old = old_dat['var_list']


dat = np.load('pairwise_L2_dist.npz',allow_pickle=True)
info_dict = dat['info_dict'].all()
beta_dict = dat['beta_dict'].all()
npz_new = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s98.npz',allow_pickle=True)
var_list = dat['var_list']

pairwise_dist = dat['pairwise_dist'].all()


pairwise_dist = pairwise_dist['m53s98','m53s98']
pairwise_dist_old = pairwise_dist_old['m53s98','m53s98']

info = info_dict['m53s98']
info_old = info_dict_old['m53s98']

sort_ind = []
for row in info:
    cluster_id = row['cluster_id']
    electrode_id = row['electrode_id']
    brain_area = row['brain_area']
    
    sel = (info_old['cluster_id']==cluster_id)*(info_old['electrode_id']==electrode_id)*(info_old['brain_area']==brain_area)
    if sel.sum():
        sort_ind += [np.where(sel)[0][0]]
        

pairwise_dist_old = pairwise_dist_old[sort_ind,:]
pairwise_dist_old = pairwise_dist_old[:,sort_ind]
info_old = info_old[sort_ind]

session_i = 'm53s98'
session_j = 'm53s98'
variable = 'ang_acc'

idx = np.where(var_list == variable)[0][0]
dist_l2 = pairwise_dist[:, :, idx]

# pairwise_dist_old
tmp = np.triu(dist_l2, 1)
tmp[np.tril_indices(tmp.shape[0])] = 2
idx_sort = np.argsort(tmp.flatten())
row, col = np.indices(pairwise_dist[:, :, idx].shape)


row = row.flatten()
col = col.flatten()

sort_i = row[idx_sort]
sort_j = col[idx_sort]

idx0 = np.where(pairwise_dist[sort_i, sort_j, idx] > 0.01)[0][0]

neu_vector_i = info['neuron']
neu_vector_j = info['neuron']

cc = 1
plt.figure(figsize=[14, 10])
plt.suptitle('L2 distance: sorted by similarity - %s' % variable)
for j in range(idx0, idx0 + 25):
    plt.subplot(5, 5, cc)
    neu_1 = neu_vector_i[row[idx_sort[j]]]
    neu_2 = neu_vector_j[col[idx_sort[j]]]
    
    idx_neu_1 = np.where(info['neuron'] == neu_1)[0][0]
    idx_neu_2 = np.where(info['neuron'] == neu_2)[0][0]
    
    
    
    # plt.title('%d - %d: %.4f' % (neu_1, neu_2, pairwise_dist_old[(session_i, session_j)][idx_neu_1, idx_neu_2, 0]), fontsize=10)

    index_1 = row[idx_sort[j]]
    index_2 = col[idx_sort[j]]


    beta_1 = beta_dict[variable][session_i][index_1,:]
    beta_2 = beta_dict[variable][session_j][index_2,:]
    # beta_1 = beta_dict[variable][session_i][idx_neu_1,:]
    # beta_2 = beta_dict[variable][session_j][idx_neu_2,:]

    # load tuining 1
    folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/' % ( session_i)
    fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
    with open(folder + fhName, "rb") as dill_file:
        gam_res_dict = dill.load(dill_file)
    gam_model = gam_res_dict['full']
    knots = gam_model.smooth_info[variable]['knots'][0]
    order = gam_model.smooth_info[variable]['ord']
    is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
    exp_bspline_1 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

    beta_zeropad = np.hstack((beta_1, [0]))
    tuning_raw_1 = tuning_function(exp_bspline_1, beta_zeropad, subtract_integral_mean=False)


    # load tuining 2
    folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/' % ( session_j)
    fhName = 'fit_results_%s_c%d_%s_%.4f.dill' % (session_j, neu_2, 'all', 1)
    with open(folder + fhName, "rb") as dill_file:
        gam_res_dict = dill.load(dill_file)
    gam_model = gam_res_dict['full']
    knots = gam_model.smooth_info[variable]['knots'][0]
    order = gam_model.smooth_info[variable]['ord']
    is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
    exp_bspline_2 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

    beta_zeropad = np.hstack((beta_2, [0]))
    tuning_raw_2 = tuning_function(exp_bspline_2, beta_zeropad, subtract_integral_mean=False)

    # integral subtract
    c1 = tuning_raw_1.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
    func_1 = lambda x: tuning_raw_1(x) - c1
    c2 = tuning_raw_2.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
    func_2 = lambda x: tuning_raw_2(x) - c2

    # normalization constant
    
    x0 = max(knots[0],range_dict[variable][0])
    x1 = min(knots[-1],range_dict[variable][1])

    xx = np.linspace(x0, x1 - 10 ** -6, 10 ** 4)
    norm_1 = np.sqrt(simps(func_1(xx) ** 2, dx=xx[1] - xx[0]))
    norm_2 = np.sqrt(simps(func_2(xx) ** 2, dx=xx[1] - xx[0]))

    xx = np.linspace(x0, x1 - 10 ** -6 - 10 ** -6, 10 ** 3)

    plt.plot(xx, func_1(xx) / norm_1,'k')
    plt.plot(xx, func_2(xx) / norm_2, '--k')




    plt.xticks([])
    plt.yticks([])
    cc+=1
    



# idx = np.where(var_list_old == variable)[0][0]
# dist_l2 = pairwise_dist_old[:, :, idx]

# # pairwise_dist_old
# tmp = np.triu(dist_l2, 1)
# tmp[np.tril_indices(tmp.shape[0])] = 2
# idx_sort = np.argsort(tmp.flatten())
# row, col = np.indices(pairwise_dist_old[:, :, idx].shape)


# row = row.flatten()
# col = col.flatten()

# sort_i = row[idx_sort]
# sort_j = col[idx_sort]

# idx0 = np.where(pairwise_dist_old[sort_i, sort_j, idx] > 0.01)[0][0]

# neu_vector_i = info_old['neuron']
# neu_vector_j = info_old['neuron']

# cc = 1
# plt.figure(figsize=[14, 10])
# plt.suptitle('L2 distance: sorted by similarity - %s' % variable)
# for j in range(idx0, idx0 + 25):
#     plt.subplot(5, 5, cc)
#     neu_1 = neu_vector_i[row[idx_sort[j]]]
#     neu_2 = neu_vector_j[col[idx_sort[j]]]
    
#     idx_neu_1 = np.where(info_old['neuron'] == neu_1)[0][0]
#     idx_neu_2 = np.where(info_old['neuron'] == neu_2)[0][0]
    
    
    
#     # plt.title('%d - %d: %.4f' % (neu_1, neu_2, pairwise_dist_old[(session_i, session_j)][idx_neu_1, idx_neu_2, 0]), fontsize=10)

#     index_1 = row[idx_sort[j]]
#     index_2 = col[idx_sort[j]]


#     beta_1 = beta_dict_old[variable][session_i]
#     beta_1 = beta_1[sort_ind][index_1,:]
#     beta_2 = beta_dict_old[variable][session_j]
#     beta_2 = beta_2[sort_ind][index_2,:]
#     # beta_1 = beta_dict[variable][session_i][idx_neu_1,:]
#     # beta_2 = beta_dict[variable][session_j][idx_neu_2,:]

#     # load tuining 1
#     folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/%s_gam_fit_without_acceleration/gam_%s/' % ('cubic', session_j)
#     fhName = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session_i, neu_1, 'all', 1)
#     with open(folder + fhName, "rb") as dill_file:
#         gam_res_dict = dill.load(dill_file)
#     gam_model = gam_res_dict['full']
#     knots = gam_model.smooth_info[variable]['knots'][0]
#     order = gam_model.smooth_info[variable]['ord']
#     is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
#     exp_bspline_1 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

#     beta_zeropad = np.hstack((beta_1, [0]))
#     tuning_raw_1 = tuning_function(exp_bspline_1, beta_zeropad, subtract_integral_mean=False)


#     # load tuining 2
#     folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/%s_gam_fit_without_acceleration/gam_%s/' % ('cubic', session_j)
#     fhName = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session_i, neu_2, 'all', 1)
#     with open(folder + fhName, "rb") as dill_file:
#         gam_res_dict = dill.load(dill_file)
#     gam_model = gam_res_dict['full']
#     knots = gam_model.smooth_info[variable]['knots'][0]
#     order = gam_model.smooth_info[variable]['ord']
#     is_cyclic = gam_model.smooth_info[variable]['is_cyclic'][0]
#     exp_bspline_2 = spline_basis(knots, order, is_cyclic=is_cyclic, subtract_integral=False)

#     beta_zeropad = np.hstack((beta_2, [0]))
#     tuning_raw_2 = tuning_function(exp_bspline_2, beta_zeropad, subtract_integral_mean=False)

#     # integral subtract
#     c1 = tuning_raw_1.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
#     func_1 = lambda x: tuning_raw_1(x) - c1
#     c2 = tuning_raw_2.integrate(knots[0], knots[-1]) / (knots[-1] - knots[0])
#     func_2 = lambda x: tuning_raw_2(x) - c2

#     # normalization constant
    
#     x0 = max(knots[0],range_dict[variable][0])
#     x1 = min(knots[-1],range_dict[variable][1])

#     xx = np.linspace(x0, x1 - 10 ** -6, 10 ** 4)
#     norm_1 = np.sqrt(simps(func_1(xx) ** 2, dx=xx[1] - xx[0]))
#     norm_2 = np.sqrt(simps(func_2(xx) ** 2, dx=xx[1] - xx[0]))

#     xx = np.linspace(x0, x1 - 10 ** -6 - 10 ** -6, 10 ** 3)

#     plt.plot(xx, func_1(xx) / norm_1,'k')
#     plt.plot(xx, func_2(xx) / norm_2, '--k')




#     plt.xticks([])
#     plt.yticks([])
#     cc+=1