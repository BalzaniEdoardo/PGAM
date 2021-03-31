import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import scipy.stats as sts
from scipy.io import savemat
res_mat = np.load('between_area_coupling_filters.npy',allow_pickle=True)



mat_filters = np.zeros((res_mat.shape[0],res_mat[0]['y'].shape[0]))

cc = 0
for row in res_mat:
    mat_filters[cc,:] = row['y']

    cc+=1

# mat_filters = sts.zscore(mat_filters,axis=1)

pca = PCA()
res_pca = pca.fit(mat_filters)

cumsum_var_expl = np.cumsum(res_pca.explained_variance_ratio_)
plt.figure()
plt.plot(cumsum_var_expl,'-ok')
keepNum = (cumsum_var_expl < 0.95).sum()

plt.figure()
proj = res_pca.transform(mat_filters)
plt.scatter(proj[:,0], proj[:,1])
tsne = TSNE()
transf = tsne.fit_transform(proj[:,:keepNum])

dbscan = DBSCAN(eps=3.2, min_samples=10)
dbscan.fit(transf)

np.savez('noZscore_dbscan_fit.npz',dbscan_results=dbscan, tsne_results=tsne,tsne_transf=transf,raw_filt=mat_filters)

recipr = []
# check reciprocal couplings
skip = ['m53s46','m53s124','m53s128','m53s31','m53s47']
for session in np.unique(res_mat['session']):
    if session in  skip:
        continue
    sub = res_mat[res_mat['session']==session]
    # if 'MST' in sub['sender brain area']
    for row in sub:
        send = row['sender unit id']
        rec = row['receiver unit id']
        if send > rec:
            continue
        if any((sub['sender unit id'] == rec) & (sub['receiver unit id'] == send)):
            recipr += [(session,send,rec,row['sender brain area'],row['receiver brain area'])]
    # sender_rec  = np.array([sub['sender unit id']

# dat = np.load('dbscan_fit.npz',allow_pickle=True)
# dbscan = dat['dbscan_results'].all()
# tsne = dat['tsne_results'].all()
# transf = dat['tsne_transf']
# mat_filters = dat['raw_filt']

fig = plt.figure(figsize=[6., 6.  ])
grid = plt.GridSpec(8, 3, wspace=0.4, hspace=0.3)
ax1 = plt.subplot(grid[:2, :])
row = 2
col = 0
ax_dict = {}
for kk in range(16):
    ax_dict[kk] = plt.subplot(grid[row, col])
    col += 1
    if col % 3 == 0:
        row += 1
    col = col % 3

kk = 0
for cl in np.unique(dbscan.labels_):
    cl_trans = transf[dbscan.labels_ == cl]
    ax2 = ax_dict[kk]
    if cl == -1:
        p = ax1.scatter(cl_trans[:, 0], cl_trans[:, 1], color=(0.5,) * 3, alpha=0.4, s=2)
    elif (dbscan.labels_ == cl).sum() < 20:
        continue
    else:
        p = ax1.scatter(cl_trans[:, 0], cl_trans[:, 1])
        col = p.get_facecolor()[0][:3]
        xx = np.arange(mat_filters.shape[1])*0.006

        ax2.plot(xx,mat_filters[dbscan.labels_ == cl].mean(axis=0), color=col)
        ax2.fill_between(xx, mat_filters[dbscan.labels_ == cl].mean(axis=0) - mat_filters[dbscan.labels_ == cl].std(
            axis=0),
                         mat_filters[dbscan.labels_ == cl].mean(axis=0) + mat_filters[dbscan.labels_ == cl].std(
                             axis=0), color=col, alpha=0.4)
    kk+=1

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
grid.tight_layout(fig,rect=[0, 0.03, 1, 0.95])
plt.suptitle('tsne_and_clustering: between area coupling filters')
# plt.savefig('tsne_cluster.pdf',bbox_inches='tight')

mat_dict = {'cluster_labels':dbscan.labels_,'raw_filt':mat_filters,
            'raw_filters_and_info':res_mat,'tsne_projection':transf}

# savemat('raw_cluster_filters.mat',mdict=mat_dict)