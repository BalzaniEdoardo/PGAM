#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:28:32 2018

@author: edoardo
"""

import numpy as np
from sklearn import neighbors,cluster
from seaborn import heatmap
from scipy.spatial.distance import pdist,squareform
from scipy import sparse
from scipy.linalg import eig
import matplotlib.pylab as plt
from seaborn import heatmap as snsheatmap

def KNN_Graph(S, k):
    """
        Input:
        =====
            S = nxn similarity matrix
            k = number of neighbours
        Output:
        =======
            W = nxn ajdacency matrix
            D = nxn element degree matrix
    """
    
    knn = neighbors.NearestNeighbors(n_neighbors=k,metric='precomputed')
    knn.fit(S)
    W = knn.radius_neighbors_graph(mode='distance')
    D = sparse.csr_matrix(np.diag(np.sum(W.toarray(),axis=0)))
    return knn,W,D

#def laplacian_RW(W,D):
#    Dinv = np.linalg.pinv(D.toarray())
#    L_rw = np.eye(Dinv.shape[0]) - np.dot(Dinv,W.toarray())
#    return L_rw

def normalized_spectClust(S,clust_num, graph_method='KKN_graph', neigh_num=None):
    if graph_method == 'KKN_graph':
        _,W,D = KNN_Graph(S, neigh_num)
        W = W.toarray()
        D = D.toarray()
    if graph_method == 'full':
        W = S
        D = np.diag(np.sum(W,axis=1))
    L = D - W
    eigVal,eigVec = eig(L,b=D,right=False,left=True)
    # the eig are real since L and D are positive semidef
    eigVal = np.real(eigVal)
    eigVec = np.real(eigVec)
    sortIdx = np.argsort(eigVal)
    eigVal = eigVal[sortIdx]
    eigVec = eigVec[:,sortIdx]
    eigVal = eigVal[:clust_num]
    eigVec = eigVec[:,:clust_num]
    kmeans = cluster.KMeans(n_clusters=clust_num,n_init=40)
    kmeans.fit(eigVec)
    return kmeans, eigVal,eigVec

if __name__ == '__main__':
    plt.close('all')
    monkeyID = 'm44s183'
    np.random.seed(3)
    clust_num = 8
    X1 = np.random.normal(loc=0,scale=0.5,size=(250,2))
    X2 = np.random.normal(loc=1.5,scale=0.5,size=(250,2))
    X = np.vstack((X1,X2))
    perm = np.random.permutation(range(500))
    # X = X[perm,:]
    # S = pdist(X)
    # S = np.exp(-squareform(S))
    dat = np.load('/Volumes/WD Edo/firefly_analysis/tuning_clustering/neu_beta_%s.npz'%monkeyID,allow_pickle=True)
    brain_area = dat['brain_area']
    index_dict = dat['index_dict'].all()
    num_neu = dat['num_neu']
    beta_label = dat['beta_label']
    S = dat['dist_eucl']
    beta_matrix = dat['beta_matrix']

    S = squareform(S)
    S = np.exp(-(S)/np.max(S))
    heatmap(S)
    #
    ## compute adjacency and 
    #knn,W,D = KNN_Graph(S,k)
    #S_sparse = knn.radius_neighbors_graph(mode='distance')
    #plt.figure()
    #heatmap(S_sparse.toarray())
    #
    ## compute L_rw
    #L_rw = laplacian_RW(W,D)
    kmeans, eigVal,eigVec = normalized_spectClust(S,clust_num, graph_method='KKN_graph', neigh_num=20)
    
    labels = kmeans.predict(eigVec)
    Ssort = np.zeros(S.shape)
    cc=0
    sort_idx = []
    for ii in np.unique(labels):
        sort_idx = np.hstack((sort_idx,np.where(labels==ii)[0]))
        print(ii,'clust num',np.sum(labels==ii))
    sort_idx = np.array(sort_idx,dtype=int)
    Ssort = S[sort_idx,:]
    Ssort = Ssort[:,sort_idx]


    plt.figure()
    heatmap(Ssort)

    clust_num = 30
    _, eigValCheck, _ = normalized_spectClust(S, clust_num, graph_method='KKN_graph', neigh_num=20)
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(2, eigValCheck.shape[0] + 1), eigValCheck[1:], '-ob')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Order')
    plt.title('Eig Decomposition Results')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.figure()
    plt.title('Percent Neuron per cluster')

    cc = 0
    x_tick_pos = []
    x_tick_lab = []
    clust_color = {}
    for clu_num in np.unique(labels):
        bar_x = []
        bar_h = []
        label_x_position = []
        for ba in np.unique(brain_area):
            count_ba = np.sum(brain_area[labels == clu_num] == ba)
            count_tot = np.sum(labels == clu_num)
            bar_h += [100.*(float(count_ba)/count_tot)]
            bar_x += [cc]
            label_x_position += [ba]
            plt.text(bar_x[-1], bar_h[-1] + 3, '%d' % count_ba, horizontalalignment='center')
            cc += 1

        p = plt.bar(bar_x, bar_h, align='center')#, color=clust_color[clu_num])

        clust_color[clu_num] = p[0].get_facecolor()
        x_tick_lab += label_x_position
        x_tick_pos += bar_x
        cc += 2

    plt.xticks(x_tick_pos, x_tick_lab, rotation=90)
    plt.ylim(0, 110)


    brain_area_avail = np.unique(brain_area)

    for clu_num in np.unique(labels):
        plt.figure(figsize=[11., 6.83])

        # if clu_num != 9:
        #     continue
        select_neu = labels == clu_num

        # get the centroid
        clu_beta = beta_matrix[select_neu, :]
        mean_beta = clu_beta.mean(axis=0)
        std_beta = clu_beta.std(axis=0)
        plt.suptitle('Cluster %d - tot cells %d' % (clu_num,np.sum(select_neu)))
        # split and plot centroid
        cc = 1
        for variable in index_dict.keys():
            array_var = index_dict[variable] - 1
            centr_tuning = mean_beta[array_var]
            label = np.unique(beta_label[1 + array_var])
            std_tuning = std_beta[array_var]
            if len(label) > 1:
                raise ValueError
            else:
                label = label[0]
            plt.subplot(3, 4, cc)
            plt.title(label)
            p, = plt.plot(centr_tuning,color=clust_color[clu_num])
            x = np.arange(centr_tuning.shape[0])
            plt.fill_between(x, centr_tuning - std_tuning, centr_tuning + std_tuning, facecolor=p.get_color(),
                             alpha=0.4)

            plt.xticks([])
            cc += 1

    plt.figure()
    snsheatmap(beta_matrix[sort_idx,:])