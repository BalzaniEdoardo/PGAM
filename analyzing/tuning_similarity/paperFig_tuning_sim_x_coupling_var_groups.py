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
from copy import  deepcopy
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.family':'Arial'})


monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno','m72':'Marco','m73':'Jimmy'}
monkey = 'all'

blue = np.array([40.,20.,204.])/255.
green = np.array([0.,176.,80.])/255.
red = np.array([255.,0.,0.])/255.


delta_edge = 0.2
edge_mst = np.array([0,1,2,3])
edge_ppc = edge_mst+delta_edge
edge_pfc = edge_ppc+delta_edge

# height_ppc = [0.05,0.07,0.08,0.1]
# height_pfc = [0.05,0.05,0.05,0.05]
# height_mst = [0.05,0.07,0.25,0.65]
#
# plt.bar(edge_mst,height_mst,width=delta_edge,color=green)
# plt.bar(edge_ppc,height_ppc,width=delta_edge,color=blue)
# plt.bar(edge_pfc,height_pfc,width=delta_edge,color=red)

for key in monkey_dict.keys():
    if monkey_dict[key] == monkey:
        monkey_id = key
        break
    else:
        monkey_id='all'

monkey_dict = {}

check_pair_matrix_rows = True
filter_nonzero_coup = True

dat = np.load('pairwise_L2_dist.npz',allow_pickle=True)
info_dict = dat['info_dict'].all()
beta_dict = dat['beta_dict'].all()
pairwise_dist = dat['pairwise_dist'].all()
var_list = dat['var_list']


coupling_dict = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/pairwise_coupling.npz',allow_pickle=True)['coupling_dict'].all()



filter_nonzero_coup = False

# sensory = ['ang_vel','rad_vel','t_flyOFF']
# # internal = ['rad_target','ang_target','ang_path','rad_path']
# # reward = ['t_reward']
# # motor = ['t_stop','t_move']
# # lfp = ['lfp_beta']
# # label = ['sensory variables','internal variables','motor variables','reward','lfp']
# #
# # color_list = [(158/255.,42/255.,155/255.),(244/255.,90/255.,42/255.),
# #               (3/255.,181/255.,149/255.),(125/255.,)*3,(125/255.,)*3]

sensory = ['ang_vel','rad_vel','t_flyOFF','t_stop','t_move','rad_acc','ang_acc']
internal = ['rad_target','ang_target','ang_path','rad_path']
reward = ['t_reward']
lfp = ['eye_vert','eye_hori']#['lfp_beta']
label = ['sensorimotor variables','internal variables','reward','eye position']

color_list = [(158/255.,42/255.,155/255.),(244/255.,90/255.,42/255.),
              (125/255.,)*3,(125/255.,)*3]


plt.figure(figsize=(10,8))


height_mst = {}
height_ppc = {}
height_pfc = {}

for filter_area in ['MST','PPC','PFC']:
    cnt_var = 0
    for var_list_this in [sensory,internal,reward,lfp]:



        idx_var = np.zeros(var_list.shape[0],dtype=bool)
        for var in var_list_this:
            idx_var[var_list==var] = True

        pair_dist = []
        coupling = []
        for key in coupling_dict.keys():
            if (not  monkey_id in key[0]) and (not monkey_id == 'all'):
                continue
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


        # print(var, 'correlation:',sts.pearsonr(coupling,tuning_sim)[0])
        #
        # ax = plt.subplot(2,3,cnt_var+1)
        #
        edges = np.linspace(0,1,5)
        coupling_list = []
        counts = []
        for k in range(edges.shape[0]-1):
            idx = ((tuning_sim/2) > edges[k]) * ((tuning_sim/2 ) <= edges[k+1])

            coupling_list += [(coupling[idx]>0).sum() / idx.sum()]
            counts += [idx.sum()]

        if filter_area == 'MST':
            height_mst[label[cnt_var]] = deepcopy(coupling_list)
        if filter_area == 'PPC':
            height_ppc[label[cnt_var]] = deepcopy(coupling_list)
        if filter_area == 'PFC':
            height_pfc[label[cnt_var]] = deepcopy(coupling_list)
        print(filter_area,label[cnt_var],counts)
        # print(counts)
        # ax.bar(edges[:-1],coupling_list,width =(edges[1]-edges[0])*0.5,color=color_list[cnt_var])
        # ax.set_title(label[cnt_var])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.ylabel('coupling probability')
        # plt.xlabel('tuning similarity')
        # ax.set_xlim(-0.25, 1)
        # ax.set_ylim(0,0.8)

        cnt_var += 1

plt.figure()

for k in range(4):
    ax = plt.subplot(2,2,k+1)
    plt.title(label[k],fontsize=12)

    plt.bar(edge_mst,height_mst[label[k]],width=delta_edge,color=green)
    plt.bar(edge_ppc,height_ppc[label[k]],width=delta_edge,color=blue)
    plt.bar(edge_pfc,height_pfc[label[k]],width=delta_edge,color=red)
    plt.ylim(0,0.75)
    plt.xticks([edge_ppc[0],edge_ppc[-1]],[0,1])
    if k % 2 == 0:
        plt.ylabel('coupling prob.',fontsize=12)
    if k >= 2:
        plt.xlabel('tuning similarity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('paper_tuning_sim_cp.pdf')

plt.figure()

for k in range(4):
    ax = plt.subplot(2,2,k+1)
    ax.yaxis.grid(linestyle='--',alpha=0.5)
    plt.title(label[k],fontsize=12)

    plt.bar(edge_mst,height_mst[label[k]]/height_mst[label[k]][0],width=delta_edge,color=green)
    plt.bar(edge_ppc,height_ppc[label[k]]/height_ppc[label[k]][0],width=delta_edge,color=blue)
    plt.bar(edge_pfc,height_pfc[label[k]]/height_pfc[label[k]][0],width=delta_edge,color=red)
    # plt.ylim(0,0.75)
    plt.xticks([edge_ppc[0],edge_ppc[-1]],[0,1])
    if k % 2 == 0:
        plt.ylabel('c.p. ratio',fontsize=12)
    if k >= 2:
        plt.xlabel('tuning similarity',fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('ratio_paper_tuning_sim_cp.pdf')
# if filter_area == 'all':
#     plt.savefig('monk_%s_coupling_prob_grouped.pdf'%monkey)
#
# else:
#     plt.savefig('monk_%s_coupling_prob_grouped_%s.pdf'%(monkey,filter_area))
