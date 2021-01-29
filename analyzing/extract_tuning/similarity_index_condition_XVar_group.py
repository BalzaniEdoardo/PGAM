#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:18:36 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
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

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}

path_gen = get_paths_class()
info_selectivity = np.load('response_strength_info.npy')
dat = np.load('eval_matrix_and_info.npz',allow_pickle=True)
        #eval_matrix=eval_matrix,info=info_matrix,index_list=index_list)
info_evals = dat['info']
eval_matrix = dat['eval_matrix']
var_list = dat['index_list']

# variable_combine = ['rad_acc','ang_acc']
# label_combine = 'acceleration'

#

variable_group_dict = {'sensory':['rad_vel','ang_vel'],
                       'internal':['rad_path','rad_target','ang_path','ang_path'],
                       'acceleration':['rad_acc','ang_acc'],
                       'movment_onOFF':['t_move','t_stop'],
                       'reward':['t_reward'],
                       'taget':['t_flyOFF']}

# variable_combine = ['eye_vert',]
label_combine = 'eye vertical position'

monkey = 'm44'

man_type = 'odd'
odd_val = 1
even_val  = 0

subselect = 'odd'
# slice the variable of interest
similarity_dict_group = {}
for variable_group in variable_group_dict.keys():
    variable_combine = variable_group_dict[variable_group]
    boolsele = np.zeros(eval_matrix.shape[1],dtype=bool)
    tuning_matrix_dict = {}
    
    
    for var in variable_combine:
        tuning_matrix_dict[var] = eval_matrix[:, var_list == var]
    
    
    session_list = []#['m53s95','m53s98','m53s114','m53s115','m53s105',]
    for session in np.unique(info_selectivity['session']):
        
        if not monkey in session:
            continue
        
        man_type_sess = np.unique(info_selectivity[info_selectivity['session']==session]['manipulation type'])
        
        sess_filt = []
        
        filt = info_selectivity['session']==session
        if man_type == 'odd' and (('controlgain' in man_type_sess) or ('ptb' in man_type_sess)):
            continue
        if (subselect == 'all') and ( man_type in man_type_sess):
            session_list += [session]
        elif any(info_selectivity[filt]['manipulation type'] == subselect) and ( man_type in man_type_sess):
            session_list += [session]
        # if ('controlgain' in man_type_sess) or ('ptb' in man_type_sess in man_type_sess):
        #     continue
        
        # if not (man_type in man_type_sess):
        #     continue
    
        
            
        
    # session_list.remove('m53s39')
    
    # cnt_session = 0
    if monkey == 'm53':
        brain_area_list = ['PPC','PFC','MST']
    if monkey == 'm44':
        brain_area_list = ['PPC','MST']
    similarity_dict = {}
    for ba in brain_area_list:
        similarity_all = np.zeros((2, 0))
    
        for session in session_list:
            
            for variable in variable_combine:
            
                sele = ((info_evals['session'] == session) & 
                        (info_evals['manipulation type']==man_type)*
                        (info_evals['brain area']==ba))
                
                sele_selctivity = ((info_selectivity['session']==session)*
                                   (info_selectivity['manipulation type']==man_type)*
                                   (info_selectivity['brain_area']==ba)
                                   )
                
                
                odd_info = info_selectivity[sele_selctivity * 
                                        (info_selectivity['manipulation value']==odd_val)]
                even_info = info_selectivity[sele_selctivity * 
                                        (info_selectivity['manipulation value']==even_val)]
                
                # extract variables that are tuned to both
                tuned_odd = []
                tuned_even = []
                for row in odd_info:
                    if row[variable]:
                        tuned_odd += [row['unit']]
                        
                for row in even_info:
                    if row[variable]:
                        tuned_even += [row['unit']]
                
                # intesection
                # tuned_both = list(set(tuned_even).intersection(tuned_odd))
                
                # UNION
                tuned_both = list(set(tuned_even).union(tuned_odd))
        
                
                # compute the similarity
                sim_session = np.zeros((2, len(tuned_both)))* np.nan
                
                
                bool_odd = sele * (info_evals['manipulation value']==odd_val)
                bool_even = sele * (info_evals['manipulation value']==even_val)
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
                    tun_odd = tuning_matrix_dict[var][bool_odd * (info_evals['unit id']==unit),:]
                    tun_even = tuning_matrix_dict[var][bool_even * (info_evals['unit id']==unit),:]
                    
                    # shuffle significant
                    # tun_shuffle = tuning_matrix[bool_even * (info_evals['unit id']==shuffle_id[cc]),:]
                    
                    # shuffle all
                    tun_shuffle = tuning_matrix_dict[var][bool_even * (info_evals['unit id']==shuffle_id_odd[cc%(shuffle_id_odd.shape[0])]),:]
                    tun_shuffle2 = tuning_matrix_dict[var][bool_even * (info_evals['unit id']==shuffle_id_even[cc%(shuffle_id_even.shape[0])]),:]
            
                    
                    # in the case the unit has not been processed
                    if tun_even.shape[0] == 0 or tun_odd.shape[0] == 0:
                        # cc+=1
                        # continue
                        pass
                    else:
                        sim_session[0,cc] = sts.pearsonr(tun_odd[0],tun_even[0])[0]
                        
                    
                    # shuffle significant
                    # if  tun_odd.shape[0] == 0 or tun_shuffle.shape[0] == 0:
                    #     pass
                    # else:
                    #     sim_session[1,cc] = sts.pearsonr(tun_odd[0],tun_shuffle[0])[0]
                    
                    # shuffle all
                    if  tun_shuffle.shape[0] == 0 or tun_shuffle2.shape[0] == 0:
                        pass
                    else:
                        sim_session[1,cc] = sts.pearsonr(tun_shuffle[0],tun_shuffle2[0])[0]
                    cc+=1
                    
                
                evals = eval_matrix[sele]
                resp = evals[:, var_list==variable]
                similarity_all = np.hstack((similarity_all,sim_session))
        similarity_dict[ba] = deepcopy(similarity_all)
    similarity_dict_group[variable_group] = deepcopy(similarity_dict)
    # plt.plot(resp[0,:])
    # plt.plot(resp[1,:])
    # break



for group in similarity_dict_group.keys():
    
# plt.figure(figsize=[4.73, 4.8 ])
# plt.title('similarity '+label_combine)

# color_ba = {'MST':np.array([0,1,0]),
#             'PPC':np.array([0,0,1]),
#             'PFC':np.array([1,0,0])}
# for ba in brain_area_list:
#     x = np.linspace(-1,1,100)
#     similarity_all = similarity_dict[ba]
#     ecdf_true = ECDF(similarity_all[0,~np.isnan(similarity_all[1,:])])
#     ecdf_sh = ECDF(similarity_all[1,~np.isnan(similarity_all[1,:])])
#     plt.plot(x,ecdf_true(x), color=color_ba[ba],lw=1.5,label=ba)
#     plt.plot(x,ecdf_sh(x),color=color_ba[ba]*0.5,lw=0.5)
# plt.legend()

# plt.savefig('Figs/%s_areaComp_sim_index_%s_%s.png'%(monkey_dict[monkey],label_combine,man_type))


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
    