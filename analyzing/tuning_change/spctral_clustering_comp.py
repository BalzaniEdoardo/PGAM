import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth
from spectral_clustering import *

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
from knots_constructor import knots_cerate
from copy import deepcopy
from time import sleep
from spline_basis_toolbox import *
from bisect import bisect_left
from statsmodels.distributions import ECDF
import venn

from seaborn import heatmap
cond_pval = {'density':0.005, 'ptb':0., 'controlgain':1}

# S = np.exp(-(distance_matrix) ** 2 / np.nanmax((distance_matrix) ** 2))
# _, eigValCheck, _ = normalized_spectClust(S, clust_num, graph_method='KKN_graph', neigh_num=50)


npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'
tuning_change_fld = '/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_change/significance_tuning_function_change/'
condition = 'controlgain'
ba = 'PPC'
session = 'm53s42'

var = 't_flyOFF'

dtype_dict = {'names':('session','condition','brain_area','variable','p-val'),
                                 'formats':('U30','U30','U30','U30',float)}
result_table = np.zeros(0,dtype=dtype_dict)



lst_files = os.listdir(tuning_change_fld)
pattern = '^m\d+s\d+_[a-z]+_tuningChange.npz$'

first = True
for fh in lst_files:
    if not re.match(pattern,fh):
        continue

    splt = fh.split('_')
    session = splt[0]
    condition = splt[1]
    if condition == 'odd' or condition == 'controlgain':
        continue

    print(session,condition)
    dat = np.load(os.path.join(tuning_change_fld,fh),
        allow_pickle=True)

    npz_dat = np.load(os.path.join(npz_folder,'%s.npz'%(session)),
        allow_pickle=True)

    unit_info = npz_dat['unit_info'].all()
    brain_area = unit_info['brain_area']

    tensor_A = dat['tensor_A']
    index_dict_A = dat['index_dict_A'].all()
    unit_list = dat['unit_list']
    var_sign = dat['var_sign']

    sel = (var_sign['variable'] == var) & (var_sign['p-val cond %.4f'%cond_pval[condition]]<0.001)
    unit_sign = var_sign['unit'][sel]

    bool_unt = np.zeros(len(unit_list),dtype=bool)

    for unt in unit_sign:
        bool_unt[unit_list==unt] = True

    tun_sign = tensor_A[bool_unt, 0, :]
    tun_sign = tun_sign[:, index_dict_A[var]]
    if tun_sign.shape[0] == 0:
        continue
    tun_sign = tun_sign[:,~np.isnan(tun_sign[0])]
    if first:
        tun_sign_all = deepcopy(tun_sign)
        first = False
    else:
        tun_sign_all = np.vstack((tun_sign_all,tun_sign))
    # break

corr_matrix = np.zeros((tun_sign_all.shape[0],tun_sign_all.shape[0]))

for unt1 in range(0,tun_sign_all.shape[0]):
    for unt2 in  range(unt1+1, tun_sign_all.shape[0]):
        corr_matrix[unt1,unt2] = sts.pearsonr(tun_sign_all[unt1,:],tun_sign_all[unt2,:])[0]


corr_matrix = corr_matrix + corr_matrix.T
corr_matrix[np.diag_indices(corr_matrix.shape[0])] =1

S = np.exp(-(1-corr_matrix)**2)

clust_num = 30
_, eigValCheck, _ = normalized_spectClust(S, clust_num, graph_method='KKN_graph', neigh_num=50)



use_clust = 3
kmeans, eigVal, eigVec = normalized_spectClust(S, use_clust, graph_method='KKN_graph', neigh_num=50)
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
plt.suptitle('Sorted tuning similarity')
heatmap(Ssort,vmin=np.percentile(Ssort.flatten(),2),vmax=np.percentile(Ssort.flatten(),98))


# center heights:
plt.figure()
for lab in np.unique(labels):
    # if lab == 2:
    #     continue

    tun_lab = tun_sign_all[labels == lab]
    # tun_lab  = (tun_lab.T - np.mean(tun_lab-tun_lab[0],axis=1)).T

    p,=plt.plot(tun_lab.mean(axis=0),lw=2)
    plt.fill_between(range(tun_lab.shape[1]),
                     tun_lab.mean(axis=0) - 1. * tun_lab.std(axis=0),
                     tun_lab.mean(axis=0)+1.*tun_lab.std(axis=0),alpha=0.4,color=p.get_color())
    # for k in range(5):
    #     plt.plot(tun_lab[k],color=p.get_color(),lw=0.8)

