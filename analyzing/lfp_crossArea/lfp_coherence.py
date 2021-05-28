import numpy as np
from scipy.io import loadmat
# import matplotlib.pylab as plt
from cross_coherence_funcs import *
from time import perf_counter
import os,sys, re


bruno_ppc_map = np.hstack(([np.nan],np.arange(1,9),[np.nan], np.arange(9,89), [np.nan], np.arange(89,97),[np.nan])).reshape((10,10))

consec_elect_dist_linear = 100
consec_elect_dist_utah = 400

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno','m72':'Marco','m73':'Jimmy'}


electrode_map_dict = {
    'Schro': {'PPC': np.arange(1,49).reshape((8,6)), 'PFC': np.arange(49,97).reshape((8,6)),'MST':np.arange(1,25),'VIP':np.arange(1,25)},
    'Bruno': {'PPC': bruno_ppc_map},
    'Quigley':{'PPC':bruno_ppc_map,'MST':np.arange(1,25),'VIP':np.arange(1,25)}
    }


use_ele = {}
use_ele['Quigley'] = {
    'PPC':[10,17,80,87,54],
    'MST':list(range(1,25))[::5],
    'VIP':list(range(1,25))[::5]
    }

use_ele['Schro'] = {
    'PPC':[1,6,43,48,28],
    'PFC':[49,54,91,96,76],
    'MST':list(range(1,25))[::5],
    'VIP':list(range(1,25))[::5]
    }

use_ele['Bruno'] = {
    'PPC':[10,17,80,87,54],
    }

use_ele['Jimmy'] = {
    'MST':list(range(1,25))[::5],
    'PFC':list(range(1,25))[::5],
    'VIP':list(range(1,25))[::5],
    }

use_ele['Viktor'] = {
    'MST':list(range(1,25))[::5],
    'PFC':list(range(1,25))[::5],
    'VIP':list(range(1,25))[::5],
    }
#
#
#
# if os.path.exists('/scratch/jpn5'):
#     fh_folder = '/scratch/jpn5/mat_files/'
#     JOB = int(sys.argv[1]) - 1
# else:
#     fh_folder = 'D:\Savin-Angelaki\saved\mat_files\\'
#     JOB = 0
#
# all_fh = os.listdir(fh_folder)
# lst_fh = []
# pattern = '^m\d+s\d+.mat$'
#
# for name in all_fh:
#     if re.match(pattern,name):
#         lst_fh += [name]
#
# fh = lst_fh[JOB]
#
# # unpack LFP
# dat = loadmat(os.path.join(fh_folder,fh))['lfps']
#
# # select the electrode
# monkey = monkey_dict[fh.split('s')[0]]
# ch_list = []
#
# for ch in range(dat.shape[1]):
#     area = dat[0,ch]['brain_area'][0]
#     ele_id = dat[0,ch]['electrode_id'][0][0]
#     if ele_id in use_ele[monkey][area]:
#        ch_list += [ch]
#
# # compute coherence
# trNum = dat[:,0]['trials'][0].shape[1]
# dt = 0.006
# edges = np.linspace(0,80,40)
# N = len(ch_list)*(len(ch_list)-1)//2
# cohHist = np.zeros((N, edges.shape[0]-1))
#
# dtype_dict = {
#     'names':('monkey','session','area_ele1','area_ele2','electrode_id1','electrode_id2'),
#     'formats':('U30',)*4+(int,)*2
#     }
# info = np.zeros(N,dtype=dtype_dict)
# cc = 0
# for k in range(len(ch_list)):
#     for j in range(k+1,len(ch_list)):
#         print(k,j)
#         ch1 = ch_list[k]
#         ch2 = ch_list[j]
#         for tr in range(trNum):
#
#             lfp1 = np.squeeze(dat[:,ch1]['trials'][0][0,tr][0])
#             lfp2 = np.squeeze(dat[:,ch2]['trials'][0][0,tr][0])
#
#             f, fki, fkj, cij, ph, coh = mtem(lfp1, lfp2, dt)
#
#             if tr==0:
#                 coh_all = np.zeros((trNum,),dtype=object)
#                 freq_all = np.zeros((trNum,),dtype=object)
#             coh_all[tr] = np.real(coh)
#             freq_all[tr] = f
#
#         # stack all coherence estimates
#         CH = np.hstack(coh_all)
#         F = np.hstack(freq_all)
#         for kk in range(cohHist.shape[1]):
#             cohHist[cc, kk] = CH[(F >= edges[kk]) & (F < edges[kk+1])].mean()
#         info[cc]['monkey'] = monkey
#         info[cc]['session'] = fh.split('.')[0]
#         info[cc]['area_ele1'] = dat[0,ch1]['brain_area'][0]
#         info[cc]['area_ele2'] = dat[0,ch2]['brain_area'][0]
#         info[cc]['electrode_id1'] = dat[0,ch1]['electrode_id'][0][0]
#         info[cc]['electrode_id2'] = dat[0,ch2]['electrode_id'][0][0]
#
#         cc+=1
# np.savez('%s_LFP_coherence.npz'%fh.split('.')[0],info=info,choerence=cohHist,
#          freq=edges[:-1])

