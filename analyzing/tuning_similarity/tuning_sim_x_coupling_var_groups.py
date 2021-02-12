import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import matplotlib.pylab as plt
import dill,sys,os
sys.path.append('/Users/edoardo/Work/Code/Angelaki-Savin/GAM_library')
from seaborn import *
from GAM_library import *
from scipy.integrate import simps
from spectral_clustering import *
from basis_set_param_per_session import *
from spline_basis_toolbox import *
from scipy.cluster.hierarchy import linkage,dendrogram
import scipy.stats as sts
import statsmodels.api as sm

filter_area = 'MST'
check_pair_matrix_rows = True
filter_nonzero_coup = True

dat = np.load('pairwise_L2_dist.npz',allow_pickle=True)
info_dict = dat['info_dict'].all()
beta_dict = dat['beta_dict'].all()
pairwise_dist = dat['pairwise_dist'].all()
var_list = dat['var_list']


coupling_dict = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/pairwise_coupling.npz',allow_pickle=True)['coupling_dict'].all()



filter_nonzero_coup = False

sensory = ['ang_vel','rad_vel','t_flyOFF']
internal = ['rad_target','ang_target','ang_path','rad_path']
reward = ['t_reward']
motor = ['t_stop','t_move']
lfp = ['lfp_beta']
label = ['sensory variables','internal variables','motor variables','reward','lfp']

color_list = [(158/255.,42/255.,155/255.),(244/255.,90/255.,42/255.),
              (3/255.,181/255.,149/255.),(125/255.,)*3,(125/255.,)*3]

plt.figure(figsize=(10,8))
cnt_var = 0
for var_list_this in [sensory,internal,motor,reward,lfp]:

    idx_var = np.zeros(var_list.shape[0],dtype=bool)
    for var in var_list_this:
        idx_var[var_list==var] = True

    pair_dist = []
    coupling = []
    for key in coupling_dict.keys():
        if filter_area == 'all':
            filt_bool = np.ones(info_dict[key[0]]['brain_area'].shape[0], dtype=bool)
        else:
            filt_bool = info_dict[key[0]]['brain_area'] == filter_area
        coup_vec = coupling_dict[key][filt_bool, :]
        coup_vec = coup_vec[:,filt_bool]
        coup_vec = coup_vec.flatten()

        pair_dist_vec = np.nanmean(pairwise_dist[key][:, :, idx_var],axis=2)
        pair_dist_vec = pair_dist_vec[filt_bool, :]
        pair_dist_vec = pair_dist_vec[:, filt_bool]
        pair_dist_vec = pair_dist_vec.flatten()


        keep = (pair_dist_vec > 0)
        coupling = np.hstack((coupling,coup_vec[keep]))
        pair_dist = np.hstack((pair_dist,pair_dist_vec[keep]))

    tuning_sim = 2 - pair_dist
    print(var, 'correlation:',sts.pearsonr(coupling,tuning_sim)[0])

    ax = plt.subplot(2,3,cnt_var+1)

    edges = np.linspace(0,1,5)
    coupling_list = []
    counts = []
    for k in range(edges.shape[0]-1):
        idx = ((tuning_sim/2) > edges[k]) * ((tuning_sim/2 ) <= edges[k+1])

        coupling_list += [(coupling[idx]>0).sum() / idx.sum()]
        counts += [idx.sum()]
    print(counts)
    ax.bar(edges[:-1],coupling_list,width =(edges[1]-edges[0])*0.5,color=color_list[cnt_var])
    ax.set_title(label[cnt_var])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel('coupling probability')
    plt.xlabel('tuning similarity')
    ax.set_xlim(-0.25, 1)
    ax.set_ylim(0,0.6)

    cnt_var += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


if filter_area == 'all':
    plt.savefig('coupling_prob_grouped.pdf')

else:
    plt.savefig('coupling_prob_grouped_%s.pdf'%filter_area)


## Extract pairwise firing rate
# plt.figure(figsize=(10,8))
# cnt_var = 0
# for var_list_this in [sensory,internal,motor,reward,lfp]:

#     idx_var = np.zeros(var_list.shape[0],dtype=bool)
#     for var in var_list_this:
#         idx_var[var_list==var] = True

#     pair_dist = []
#     coupling = []
#     firing = []
#     for key in coupling_dict.keys():
#         if filter_area == 'all':
#             filt_bool = np.ones(info_dict[key[0]]['brain_area'].shape[0], dtype=bool)
#         else:
#             filt_bool = info_dict[key[0]]['brain_area'] == filter_area
#         coup_vec = coupling_dict[key][filt_bool, :]
#         coup_vec = coup_vec[:,filt_bool]
#         coup_vec = coup_vec.flatten()

#         pair_dist_vec = np.nanmean(pairwise_dist[key][:, :, idx_var],axis=2)
#         pair_dist_vec = pair_dist_vec[filt_bool, :]
#         pair_dist_vec = pair_dist_vec[:, filt_bool]
#         shape_pairs = pair_dist_vec.shape
#         pair_dist_vec = pair_dist_vec.flatten()

#         firing_hz = info_dict[key[0]]['firing_rate_hz'][filt_bool]
#         paired_mean_firing = np.zeros(shape_pairs)
#         for i in range(paired_mean_firing.shape[0]):
#             for j in range(i+1,paired_mean_firing.shape[0]):
#                 paired_mean_firing[i,j] = np.mean([firing_hz[i],firing_hz[j]])
#                 paired_mean_firing[j, i] = paired_mean_firing[i,j]

#         paired_mean_firing = paired_mean_firing.flatten()
#         keep = (pair_dist_vec > 0)
#         coupling = np.hstack((coupling, coup_vec[keep]))
#         pair_dist = np.hstack((pair_dist, pair_dist_vec[keep]))
#         firing = np.hstack((firing,paired_mean_firing[keep]))
#     tuning_sim = 2 - pair_dist
#     ax = plt.subplot(2, 3, cnt_var + 1)

#     edges = np.linspace(0, 1, 5)
#     coupling_list = []
#     counts = []
#     firing_list = []
#     for k in range(edges.shape[0] - 1):
#         idx = ((tuning_sim / 2) > edges[k]) * ((tuning_sim / 2) <= edges[k + 1])

#         coupling_list += [(coupling[idx] > 0).sum() / idx.sum()]
#         firing_list += [firing[idx].mean()]
#         counts += [idx.sum()]
#     print(counts)
#     ax.bar(edges[:-1], firing_list, width=(edges[1] - edges[0]) * 0.5, color=color_list[cnt_var])
#     ax.set_title(label[cnt_var])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.ylabel('firing rate [Hz]')
#     plt.xlabel('tuning similarity')
#     ax.set_xlim(-0.25, 1)
#     # ax.set_ylim(0, 0.32)
#     cnt_var += 1
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# if filter_area == 'all':
#     plt.savefig('firing_rate_grouped.pdf')

# else:
#     plt.savefig('firing_rate_grouped_%s.pdf'%filter_area)