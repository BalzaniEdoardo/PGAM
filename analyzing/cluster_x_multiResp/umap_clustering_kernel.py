import os, inspect, sys, re

print(inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import dill
import pandas as pd
import scipy.stats as sts
import scipy.linalg as linalg
from time import perf_counter
from seaborn import heatmap
from path_class import get_paths_class
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import loadmat
import hdbscan
import umap
from scipy.spatial.distance import pdist, squareform
import seaborn as sbn
from scipy.io import savemat

def plot_tuning(data, idx, index_list, var_list, ax_dict={}, newFig=False, sign_dict=None):
    if newFig:
        ax_dict = {}
        plt.figure(figsize=(10, 8))

    kk = 1
    for var in var_list:
        if not var in ax_dict.keys():
            ax_dict[var] = plt.subplot(3, len(var_list) // 3 + 1, kk)
        if sign_dict is None:
            color = 'r'
        else:
            if sign_dict[var]:
                color = 'r'
            else:
                color = [0.5] * 3
        sele = index_list == var
        dat_var = data[:, sele]
        ax = ax_dict[var]
        ax.plot(dat_var[idx, :], color=color)
        ax.set_title(var)
        kk += 1
    return ax_dict


def get_significance(idx, info, var_list,
                     bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/'):
    session = info[idx]['session']
    cond = info[idx]['manipulation type']
    val = info[idx]['manipulation value']
    unit = info[idx]['unit id']
    if os.path.exists(os.path.join(bsfld, 'gam_%s' % session)):
        bsfld = os.path.join(bsfld, 'gam_%s' % session)
    fhname = 'gam_fit_%s_c%d_%s_%.4f.dill' % (session, unit, cond, val)
    dict_significance = {}
    for var in var_list:
        dict_significance[var] = False

    for root, dirs, names in os.walk(bsfld):
        if fhname in names:
            with open(os.path.join(root, fhname), 'rb') as fh:
                gam_res = dill.load(fh)
                reduced = gam_res['reduced']
                if reduced is None:
                    continue
                for var in var_list:
                    bl = reduced.covariate_significance['covariate'] == var
                    if np.sum(bl) == 0:
                        continue
                    else:
                        if reduced.covariate_significance['p-val'][bl] < 0.001:
                            dict_significance[var] = True
            break

    return dict_significance

np.random.seed(4)

# plt.close('all')

dat = np.load('eval_matrix_and_info.npz')
eval_matrix = dat['eval_matrix']
info = dat['info']

keep = (info['manipulation type'] == 'all') * (info['monkey'] != 'Ody') * (info['pseudo_r2'] > 0.01)
data = eval_matrix[keep]

brain_area = info['brain area'][keep]
info = info[keep]

# remvove sessions with no eye position?
unit_sel = ~np.isnan(data.sum(axis=1))
data = data[unit_sel]
brain_area = brain_area[unit_sel]

pca_model = PCA(n_components=data.shape[1])
pca_model.fit(data)

plot_comp = data.shape[1]

n_perm = 20
data_perm = np.zeros(data.shape)
expl_var_perm = np.zeros((n_perm, pca_model.explained_variance_ratio_.shape[0]))
for k in range(n_perm):
    for jj in range(data.shape[0]):
        data_perm[jj, :] = data[jj, np.random.permutation(data.shape[1])]

    pca_perm = PCA(n_components=data.shape[1])
    pca_perm.fit(data_perm)
    expl_var_perm[k, :] = pca_perm.explained_variance_ratio_

plt.plot(np.arange(1, 1 + plot_comp), pca_model.explained_variance_ratio_[:plot_comp], '-og', label='explained by PCA')
plt.errorbar(np.arange(1, 1 + plot_comp), expl_var_perm.mean(axis=0)[:plot_comp],
             yerr=expl_var_perm.std(axis=0)[:plot_comp], color='r', label='explained by chance')
plt.legend()

pval = (expl_var_perm >= pca_model.explained_variance_ratio_).mean(axis=0)
optNum = np.where(pval > 0.05)[0][0]
optPerp = np.round(np.sqrt(data.shape[0]))
pca_dat = pca_model.transform(data)[:, :optNum]

kk = 1
plt.figure()
for n_neighbors in [20, 40, 50, 100]:
    fit = umap.UMAP(n_neighbors=n_neighbors)
    umap_res = fit.fit_transform(pca_dat)
    # tsne_res = umap(n_components=2, perplexity=optPerp, early_exaggeration=12.0, learning_rate=200.0, n_iter=maxiter).fit_transform(pca_dat)

    plt.subplot(2, 2, kk)
    plt.title('%d neighbors'%n_neighbors)
    plt.scatter(umap_res[brain_area == 'PPC', 0],
                umap_res[brain_area == 'PPC', 1], s=10, c='b', alpha=0.4)

    plt.scatter(umap_res[brain_area == 'PFC', 0],
                umap_res[brain_area == 'PFC', 1], s=10, c='r', alpha=0.4)

    plt.scatter(umap_res[brain_area == 'MST', 0],
                umap_res[brain_area == 'MST', 1], s=10, c='g', alpha=0.4)

    kk += 1

fit = umap.UMAP(n_neighbors=40)

umap_res = fit.fit_transform(pca_dat)

clusterer = hdbscan.HDBSCAN(min_cluster_size=int(80),cluster_selection_epsilon=0.0).fit(umap_res)

plt.figure()
plt.suptitle('UMAP results')
plt.subplot(121, aspect='equal')
plt.scatter(umap_res[brain_area == 'PPC', 0],
            umap_res[brain_area == 'PPC', 1], s=10, c='b', alpha=0.4)
plt.scatter(umap_res[brain_area == 'PFC', 0],
            umap_res[brain_area == 'PFC', 1], s=10, c='r', alpha=0.4)

plt.scatter(umap_res[brain_area == 'MST', 0],
            umap_res[brain_area == 'MST', 1], s=10, c='g', alpha=0.4)
plt.scatter(umap_res[brain_area == 'VIP', 0],
            umap_res[brain_area == 'VIP', 1], s=10, c='k', alpha=0.9)

plt.title('Brain Area')

plt.subplot(122, aspect='equal')
col_label = {}
for label in np.unique(clusterer.labels_):

    cl_proj = umap_res[clusterer.labels_ == label]
    centroid = cl_proj.mean(axis=0)
    i0 = np.argmin(np.linalg.norm(cl_proj - centroid, axis=1))

    # D = pdist(cl_proj)

    # M = squareform(D)
    # M[np.diag_indices(M.shape[0])] = 100

    # i0,i1 = np.unravel_index(np.argmin(M), M.shape, order='C')

    if label != -1:
        sct = plt.scatter(umap_res[clusterer.labels_ == label, 0],
                          umap_res[clusterer.labels_ == label, 1], s=10, alpha=0.4)
        col_label[label] = sct.get_facecolor()[0][:3]
        plt.scatter([umap_res[clusterer.labels_ == label, 0][i0]],
                    [umap_res[clusterer.labels_ == label, 1][i0]], s=80, c='y', alpha=0.9)
    else:
        sct = plt.scatter(umap_res[clusterer.labels_ == label, 0],
                          umap_res[clusterer.labels_ == label, 1], s=10, alpha=0.4, c=[0.5, 0.5, 0.5])
        col_label[label] = sct.get_facecolor()[0][:3]

plt.title('HDBScan results')
plt.savefig('umap_dim_reduction.png')
index_list = dat['index_list']
print(info.shape,data.shape,umap_res.shape,brain_area.shape)


mdict = {'kernel_matrix':data, 'info':info[unit_sel], 'umap_proj':umap_res, 'cluster_label':clusterer.labels_,
         'brain_area':brain_area, 'variable_label':index_list}
savemat('hdbscan_umap_kernel.mat',mdict=mdict)
np.save('hdbscan_umap_kernel.npy',mdict)



#

#
# # denoised tuning
#
# X = data - data.mean(axis=0)
# denoised_data = np.dot(np.dot(X, pca_model.components_[:optNum, :].T), pca_model.components_[:optNum, :]) + data.mean(
#     axis=0)
# plt.figure()
# kk = 1
# for label in np.unique(clusterer.labels_):
#     if label == -1:
#         continue
#     plt.subplot(3, 2, kk)
#     cl_proj = umap_res[clusterer.labels_ == label]
#
#     # centroid = cl_proj[i0]
#     centroid = cl_proj.mean(axis=0)
#     i0 = np.argmin(np.linalg.norm(cl_proj - centroid, axis=1))
#
#     srt_index = np.argsort(np.linalg.norm(cl_proj - centroid, axis=1))
#
#     for k in range(4):
#         plt.plot(data[clusterer.labels_ == label][srt_index[k]], color=col_label[label])
#     col_label[label]
#     kk += 1
#
#
# index_list = dat['index_list']
#



# var_list = ['rad_vel', 'ang_vel', 'rad_target', 'ang_target',
#             't_move', 't_stop', 't_flyOFF', 't_reward', 'lfp_alpha', 'lfp_beta', 'lfp_theta']
# idx = np.argmax(umap_res[:, 0])
#
# sign_dict = get_significance(idx, info, var_list,
#                              bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
# print(idx, sign_dict['rad_vel'])
#
# ax_dict = plot_tuning(data, idx, index_list, var_list, newFig=True, sign_dict=sign_dict)
# plt.suptitle('right-most responses')
#
# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx, :], axis=1))[1:5]
# for idx in srt_idx:
#     sign_dict = get_significance(idx, info, var_list,
#                                  bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#     print(idx, sign_dict['rad_vel'])
#     plot_tuning(data, idx, index_list, var_list, ax_dict=ax_dict, sign_dict=sign_dict)
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('right_resp.png')
#
# idx = np.argmin(umap_res[:, 0])
#
# sign_dict = get_significance(idx, info, var_list,
#                              bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
# print(idx, sign_dict['t_stop'])
#
# ax_dict = plot_tuning(data, idx, index_list, var_list, newFig=True, sign_dict=sign_dict)
#
# plt.suptitle('left-most responses')
#
# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx, :], axis=1))[1:5]
# for idx in srt_idx:
#     sign_dict = get_significance(idx, info, var_list,
#                                  bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#     print(idx, sign_dict['t_stop'])
#
#     plot_tuning(data, idx, index_list, var_list, ax_dict=ax_dict, sign_dict=sign_dict)
#
# # ax_dict = plot_tuning(data,idx,index_list,var_list,ax_dict=ax_dict)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('left_resp.png')
#
# idx = np.argmax(umap_res[:, 1])
# sign_dict = get_significance(idx, info, var_list,
#                              bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#
# ax_dict = plot_tuning(data, idx, index_list, var_list, newFig=True, sign_dict=sign_dict)
# plt.suptitle('up-most responses')
#
# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx, :], axis=1))[1:5]
# for idx in srt_idx:
#     sign_dict = get_significance(idx, info, var_list,
#                                  bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#
#     plot_tuning(data, idx, index_list, var_list, ax_dict=ax_dict, sign_dict=sign_dict)
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('up_resp.png')
#
# idx = np.argmin(umap_res[:, 1])
# sign_dict = get_significance(idx, info, var_list,
#                              bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#
# ax_dict = plot_tuning(data, idx, index_list, var_list, newFig=True, sign_dict=sign_dict)
# plt.suptitle('down-most responses')
#
# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx, :], axis=1))[1:5]
# for idx in srt_idx:
#     sign_dict = get_significance(idx, info, var_list,
#                                  bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#
#     plot_tuning(data, idx, index_list, var_list, ax_dict=ax_dict, sign_dict=sign_dict)
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
# plt.savefig('down_resp.png')

# # PFC MST bottom of clusterr
# idx=np.argmin(np.linalg.norm(umap_res - np.array([13.62,3.84]),axis=1))
# ax_dict = plot_tuning(data,idx,index_list,var_list,newFig=True)
# plt.suptitle('MST-PFC bottom ')

# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx,:],axis=1))[1:5]
# for idx in srt_idx:
#     plot_tuning(data,idx,index_list,var_list,ax_dict=ax_dict)


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # PFC MST bottom of clusterr
# idx=np.argmin(np.linalg.norm(umap_res - np.array([13.15,7.]),axis=1))
# ax_dict = plot_tuning(data,idx,index_list,var_list,newFig=True)
# plt.suptitle('MST-PFC top ')

# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx,:],axis=1))[1:5]
# for idx in srt_idx:
#     plot_tuning(data,idx,index_list,var_list,ax_dict=ax_dict)


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#
# # PFC MST bottom of cluster
# idx = np.argmin(np.linalg.norm(umap_res - np.array([11.68, 9.86]), axis=1))
#
# sign_dict = get_significance(idx, info, var_list,
#                              bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#
# ax_dict = plot_tuning(data, idx, index_list, var_list, newFig=True, sign_dict=sign_dict)
# plt.suptitle('MST-PFC bottom right ')
#
# srt_idx = np.argsort(np.linalg.norm(umap_res - umap_res[idx, :], axis=1))[1:5]
# for idx in srt_idx:
#     sign_dict = get_significance(idx, info, var_list,
#                                  bsfld='/Volumes/WD_Edo/firefly_analysis/LFP_band/cubic_gam_fit_with_coupling/')
#
#     plot_tuning(data, idx, index_list, var_list, ax_dict=ax_dict, sign_dict=sign_dict)
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])



