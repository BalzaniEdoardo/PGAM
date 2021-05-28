#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:18:36 2021

@author: edoardo
"""
import os, inspect, sys, re

print(inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

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

monkey_dict = {'m44': 'Quigley', 'm53': 'Schro', 'm91': 'Ody', 'm51': 'Bruno'}

path_gen = get_paths_class()
info_selectivity = np.load('response_strength_info.npy')
dat = np.load('eval_matrix_and_info.npz', allow_pickle=True)

coupl_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy',allow_pickle=True)
# eval_matrix=eval_matrix,info=info_matrix,index_list=index_list)
info_evals = dat['info']
eval_matrix = dat['eval_matrix']
var_list = dat['index_list']

variable_combine = ['rad_vel']
monkey = 'm53'

cond_dict = {'density': [0.005, 0.0001]}
# man_type = 'odd'
# odd_val = 1
# even_val  = 0

subselect = 'density'
# slice the variable of interest
tuning_matrix_dict = {}
for var in variable_combine:
    tuning_matrix_dict[var] = eval_matrix[:, var_list == var]

session_list = []  # ['m53s95','m53s98','m53s114','m53s115','m53s105',]
for session in np.unique(info_selectivity['session']):

    if not monkey in session:
        continue

    bool_type = np.zeros(len(cond_dict.keys()))
    cnt = 0
    for man_type in cond_dict.keys():

        man_type_sess = np.unique(info_selectivity[info_selectivity['session'] == session]['manipulation type'])

        sess_filt = []

        filt = info_selectivity['session'] == session

        if (man_type in man_type_sess):
            bool_type[cnt] = True

        cnt += 1
    if all(bool_type):
        session_list += [session]

    # session_list.remove('m53s39')

brain_area_similarity = {}

# cnt_session = 0
for man_type in cond_dict.keys():
    brain_area_similarity[man_type] = {}

for man_type in cond_dict.keys():
    odd_val = cond_dict[man_type][0]
    even_val = cond_dict[man_type][1]

    for ba in ['PPC', 'PFC', 'MST']:
        similarity_all = np.zeros((2, 0))

        if ((info_selectivity['brain_area'] == ba) *
            (info_selectivity['manipulation type'] == man_type) *
            (info_selectivity['monkey'] == monkey_dict[monkey])).sum() == 0:
            continue
        for session in session_list:
            for variable in variable_combine:

                sele = ((info_evals['session'] == session) &
                        (info_evals['manipulation type'] == man_type) *
                        (info_evals['brain area'] == ba))

                sele_selctivity = ((info_selectivity['session'] == session) *
                                   (info_selectivity['manipulation type'] == man_type) *
                                   (info_selectivity['brain_area'] == ba)
                                   )

                odd_info = info_selectivity[sele_selctivity *
                                            (info_selectivity['manipulation value'] == odd_val)]
                even_info = info_selectivity[sele_selctivity *
                                             (info_selectivity['manipulation value'] == even_val)]

                # extract variables that are tuned to both
                tuned_odd = []
                tuned_even = []
                for row in odd_info:
                    if row[variable]:
                        tuned_odd += [row['unit']]

                for row in even_info:
                    if row[variable]:
                        tuned_even += [row['unit']]

                # tuned_both = list(set(tuned_even).intersection(tuned_odd))
                tuned_both = list(set(tuned_even).union(tuned_odd))

                # compute the similarity
                sim_session = np.zeros((2, len(tuned_both))) * np.nan

                bool_odd = sele * (info_evals['manipulation value'] == odd_val)
                bool_even = sele * (info_evals['manipulation value'] == even_val)
                if bool_odd.sum() == 0 or bool_even.sum() == 0:
                    continue
                # # shuffle significant
                # shuffle_id = np.random.permutation(tuned_both)

                # shuffle all
                shuffle_id_odd = np.random.permutation(
                    odd_info['unit'])
                shuffle_id_even = np.random.permutation(
                    even_info['unit'])

                cc = 0
                for unit in tuned_both:
                    tun_odd = tuning_matrix_dict[variable][bool_odd * (info_evals['unit id'] == unit), :]
                    tun_even = tuning_matrix_dict[variable][bool_even * (info_evals['unit id'] == unit), :]

                    # shuffle significant
                    # tun_shuffle = tuning_matrix[bool_even * (info_evals['unit id']==shuffle_id[cc]),:]

                    # shuffle all
                    tun_shuffle = tuning_matrix_dict[variable][
                                  bool_even * (info_evals['unit id'] == shuffle_id_odd[cc % (shuffle_id_odd.shape[0])]),
                                  :]
                    tun_shuffle2 = tuning_matrix_dict[variable][bool_even * (
                                info_evals['unit id'] == shuffle_id_even[cc % (shuffle_id_even.shape[0])]), :]

                    # in the case the unit has not been processed
                    if tun_even.shape[0] == 0 or tun_odd.shape[0] == 0:
                        # cc+=1
                        # continue
                        pass
                    else:
                        sim_session[0, cc] = sts.pearsonr(tun_odd[0], tun_even[0])[0]

                    # shuffle significant
                    # if  tun_odd.shape[0] == 0 or tun_shuffle.shape[0] == 0:
                    #     pass
                    # else:
                    #     sim_session[1,cc] = sts.pearsonr(tun_odd[0],tun_shuffle[0])[0]

                    # shuffle all
                    if tun_shuffle.shape[0] == 0 or tun_shuffle2.shape[0] == 0:
                        pass
                    else:
                        sim_session[1, cc] = sts.pearsonr(tun_shuffle[0], tun_shuffle2[0])[0]
                    cc += 1

                evals = eval_matrix[sele]
                resp = evals[:, var_list == variable]
                similarity_all = np.hstack((similarity_all, sim_session))
        brain_area_similarity[man_type][ba] = deepcopy(similarity_all)
        # plt.plot(resp[0,:])
        # plt.plot(resp[1,:])
        # break

plt.figure(figsize=[4.73, 4.8])
plt.title('similarity ' + variable + ' ' + man_type)

x = np.linspace(-1, 1, 100)

ls = ['-', '--']
cc = 0
for man_type in cond_dict.keys():
    if 'PPC' in brain_area_similarity[man_type].keys():
        similarity_all = brain_area_similarity[man_type]['PPC']
        ecdf_true = ECDF(similarity_all[0, ~np.isnan(similarity_all[0, :])])
        ecdf_sh = ECDF(similarity_all[1, ~np.isnan(similarity_all[1, :])])
        plt.plot(x, ecdf_true(x), color='b', label='PPC - %s' % man_type, ls=ls[cc])
    # plt.plot(x,ecdf_sh(x),label='sh PPC',color=(0.5,)*3)

    if 'PFC' in brain_area_similarity[man_type].keys():
        similarity_all = brain_area_similarity[man_type]['PFC']
        if similarity_all[0, ~np.isnan(similarity_all[0, :])].shape[0] == 0:
            pass
        else:
            ecdf_true = ECDF(similarity_all[0, ~np.isnan(similarity_all[0, :])])
            ecdf_sh = ECDF(similarity_all[1, ~np.isnan(similarity_all[1, :])])
            plt.plot(x, ecdf_true(x), color='r', label='PFC - %s' % man_type, ls=ls[cc])
        # plt.plot(x,ecdf_sh(x),label='sh PFC',color='k')

    if 'MST' in brain_area_similarity[man_type].keys():
        similarity_all = brain_area_similarity[man_type]['MST']
        if similarity_all[0, ~np.isnan(similarity_all[0, :])].shape[0] == 0:
            pass
        else:
            ecdf_true = ECDF(similarity_all[0, ~np.isnan(similarity_all[0, :])])
            ecdf_sh = ECDF(similarity_all[1, ~np.isnan(similarity_all[1, :])])
            plt.plot(x, ecdf_true(x), color='g', label='MST - %s' % man_type, ls=ls[cc])
    cc += 1
    # plt.plot(x,ecdf_sh(x),label='sh MST',color='k')

plt.xlabel('similarity index')
plt.ylabel('CDF')
plt.tight_layout()
plt.legend()
# plt.savefig('Figs/%s_stability/%s_sim_index_%s_%s.png'%(subselect,monkey_dict[monkey],variable,man_type))


# if man_type == 'controlgain':
#     sign_gain_ppc = {'gain=1.0':[], 'gain=1.5':[],'gain=2.0':[]}

#     # selectivity plot
#     for session in session_list:
#         sele_ppc = ((info_selectivity['session']==session)*
#                            (info_selectivity['manipulation type']==man_type)*
#                            (info_selectivity['brain_area']=='PPC')
#                            )

#         sele_pfc = ((info_selectivity['session']==session)*
#                            (info_selectivity['manipulation type']==man_type)*
#                            (info_selectivity['brain_area']=='PFC')
#                            )


#         info_ppc = info_selectivity[sele_ppc]

#         for row in info_ppc:
#             if row[variable]:
#                 ID = '%s_c%d'%(session,row['unit'])
#                 sign_gain_ppc['gain=%.1f'%(row['manipulation value'])] += [ID]


#         print(session,'PPC',info_selectivity[sele_ppc][variable].mean())
#         print(session,'PFC',info_selectivity[sele_pfc][variable].mean())

#     # transform to set
#     for key in sign_gain_ppc.keys():
#         sign_gain_ppc[key] = set(sign_gain_ppc[key])

#     plt.figure(figsize=(8,8))
#     ax = plt.subplot(111)
#     venn.venn(sign_gain_ppc,ax=ax)
#     plt.suptitle('Control Gain: PPC units responding to %s'%variable)

# if man_type == 'ptb':
#     sign_gain_ppc = {'ptb=1.0':[], 'ptb=0.0':[]}

#     # selectivity plot
#     for session in session_list:
#         sele_ppc = ((info_selectivity['session']==session)*
#                            (info_selectivity['manipulation type']==man_type)*
#                            (info_selectivity['brain_area']=='PPC')
#                            )

#         sele_pfc = ((info_selectivity['session']==session)*
#                            (info_selectivity['manipulation type']==man_type)*
#                            (info_selectivity['brain_area']=='PFC')
#                            )


#         info_ppc = info_selectivity[sele_ppc]

#         for row in info_ppc:
#             if row[variable]:
#                 ID = '%s_c%d'%(session,row['unit'])
#                 sign_gain_ppc['ptb=%.1f'%(row['manipulation value'])] += [ID]


#         print(session,'PPC',info_selectivity[sele_ppc][variable].mean())
#         print(session,'PFC',info_selectivity[sele_pfc][variable].mean())

#     # transform to set
#     for key in sign_gain_ppc.keys():
#         sign_gain_ppc[key] = set(sign_gain_ppc[key])

#     plt.figure(figsize=(8,8))
#     ax = plt.subplot(111)
#     plt.suptitle('Perturbation: PPC units responding to %s'%variable)
#     venn.venn(sign_gain_ppc,ax=ax)

# if man_type == 'odd':
#     sign_gain_ppc = {'odd=1.0':[], 'odd=0.0':[]}

#     # selectivity plot
#     for session in session_list:
#         sele_ppc = ((info_selectivity['session']==session)*
#                            (info_selectivity['manipulation type']==man_type)*
#                            (info_selectivity['brain_area']=='PPC')
#                            )

#         sele_pfc = ((info_selectivity['session']==session)*
#                            (info_selectivity['manipulation type']==man_type)*
#                            (info_selectivity['brain_area']=='PFC')
#                            )


#         info_ppc = info_selectivity[sele_ppc]
#         units_odd = info_selectivity[sele_ppc *
#                                      (info_selectivity['manipulation value']==1)]['unit']
#         units_even = info_selectivity[sele_ppc*
#                                       (info_selectivity['manipulation value']==0)]['unit']
#         for row in info_ppc:

#             if row[variable]:
#                 if (row['unit'] in units_even) and (row['unit'] in units_odd):

#                     ID = '%s_c%d'%(session,row['unit'])
#                     sign_gain_ppc['odd=%.1f'%(row['manipulation value'])] += [ID]

#                 else:
#                     print('skip',session,row['unit'])
#         # print(session,'PPC',units_odd.shape[0]-units_even.shape[0])
#         # print(session,'PFC',units_odd.shape)

#     # transform to set
#     for key in sign_gain_ppc.keys():
#         sign_gain_ppc[key] = set(sign_gain_ppc[key])

#     plt.figure(figsize=(8,8))
#     ax = plt.subplot(111)
#     plt.suptitle('Odd/Even: PPC units responding to %s'%variable)
#     venn.venn(sign_gain_ppc,ax=ax)
