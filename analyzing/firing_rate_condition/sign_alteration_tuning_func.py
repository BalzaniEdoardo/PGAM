#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:34:01 2021

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


def compute_pval_and_parameters(fh_path1,fh_path2,mean_sub=False):
    """
    Compute significance for variable differences
    """
    monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}

    with open(fh_path1,'rb') as fh:
        res = dill.load(fh)
        fit_condA = res['full']
        del res
    
    with open(fh_path2,'rb') as fh:
        res = dill.load(fh)
        fit_condB = res['full']
        del res
        
    first = True
    cc = 0
    cond_valA = float( '.'.join(os.path.basename(fh_path1).split('_')[5].split('.')[0:2]))
    cond_valB = float( '.'.join(os.path.basename(fh_path2).split('_')[5].split('.')[0:2]))
    
    cond_min = min(cond_valA,cond_valB)
    cond_max = max(cond_valA,cond_valB)

    var_sign_table = np.zeros(len(fit_condA.var_list),dtype={'names':('monkey','session','condition','unit','variable','p-val','p-val cond %.4f'%cond_min,'p-val cond %.4f'%cond_max),
                                                             'formats':('U30','U30','U30',int,'U30',float,float,float)})
    
    var_sign_table['session'] = os.path.basename(fh_path1).split('results_')[1].split('_')[0]
    var_sign_table['monkey'] = monkey_dict[var_sign_table['session'][0].split('s')[0]]
    var_sign_table['condition'] = os.path.basename(fh_path1).split('_')[4]
    var_sign_table['variable'] = fit_condA.var_list
    var_sign_table['unit'] = int(os.path.basename(fh_path1).split('_c')[1].split('_')[0])
    pval_list = np.zeros(var_sign_table.shape)*np.nan
    
    for var_name in fit_condA.var_list:

        beta_A = fit_condA.beta[fit_condA.index_dict[var_name]]
        beta_B = fit_condB.beta[fit_condB.index_dict[var_name]]

        # if beta_A.shape[0] <= beta_B.shape[0]:

        # the height is not well defined, (conservative approach, make the funtion close)
        if mean_sub and (not var_name.startswith('neu_')) and (not var_name =='spike_hist'):
            beta_A_transl = beta_A - np.nanmean(beta_A-beta_B[:beta_A.shape[0]])
        else:
            beta_A_transl = beta_A
        cov_A = fit_condA.cov_beta
        cov_A = cov_A[fit_condA.index_dict[var_name],:]
        cov_A = cov_A[:,fit_condA.index_dict[var_name]]
        
        cov_B = fit_condB.cov_beta
        cov_B = cov_B[fit_condB.index_dict[var_name],:]
        cov_B = cov_B[:,fit_condB.index_dict[var_name]]


        
        eigA = np.linalg.eigh(cov_A)[0]
        eigB = np.linalg.eigh(cov_B)[0]
        
        

        # max_eigA[cc] = max(eigA)
        # max_eigB[cc] = max(eigB)
        
        
            
        beta_mci_A = beta_A_transl - 2*np.sqrt(np.diag(cov_A))
        beta_pci_A = beta_A_transl + 2*np.sqrt(np.diag(cov_A))
        
        beta_mci_B = beta_B - 2*np.sqrt(np.diag(cov_B))
        beta_pci_B = beta_B + 2*np.sqrt(np.diag(cov_B))
        # if var_sign_table['unit'][0] == 2 and var_name == 'rad_target':
        #     xxx=1
        # if len(beta_A) != len(beta_B):
        #     print(var_name, 'different basis set')
        #     cc+=1
        #     continue
        if first:
            first = False
            # set all beta filters equal to the within area filters (the longest)
            max_beta_len = 0
            cnt_neu = 1
            len_beta_var = 0
            unit_list = [var_sign_table['unit'][0]]
            index_dict = {}
            ii = 0
            for var in fit_condA.var_list:
                if var.startswith('neu_'):
                    max_beta_len = max(max_beta_len,len(fit_condA.index_dict[var]))
                    unit_list += [int(var.split('_')[1])]
                    cnt_neu+=1
                else:
                    len_beta_var += len(fit_condA.index_dict[var])
                    index_dict[var] = np.arange(ii, ii+len(fit_condA.index_dict[var]))
                    ii+=len(index_dict[var])

            beta_vec_len = len_beta_var + max_beta_len*cnt_neu

            unit_list = np.sort(unit_list)
            for unt in unit_list:
                var = 'neu_%d' % unt
                if unt != var_sign_table['unit'][0]:
                    index_dict[var] = np.arange(ii, ii + max_beta_len)
                    ii += max_beta_len
                else:
                    index_dict[var] = np.arange(ii, ii + max_beta_len)

                    ii += max_beta_len


            # same for beta_B (controlgain has a different length)
            max_beta_len_B = 0
            cnt_neu = 1
            len_beta_var_B = 0
            # unit_list = [var_sign_table['unit'][0]]
            index_dict_B = {}
            ii = 0
            for var in fit_condB.var_list:
                if var.startswith('neu_'):
                    max_beta_len_B = max(max_beta_len_B, len(fit_condB.index_dict[var]))
                    # unit_list += [int(var.split('_')[1])]
                    cnt_neu += 1
                else:
                    len_beta_var_B += len(fit_condB.index_dict[var])
                    index_dict_B[var] = np.arange(ii, ii + len(fit_condB.index_dict[var]))
                    ii += len(index_dict_B[var])

            beta_vec_len_B = len_beta_var_B + max_beta_len_B * cnt_neu

            
            
            unit_list = np.sort(unit_list)
            for unt in unit_list:
                var = 'neu_%d'%unt
                if unt != var_sign_table['unit'][0]:
                    index_dict_B[var] = np.arange(ii, ii+max_beta_len_B)
                    ii += max_beta_len_B
                else:
                    index_dict_B[var] = np.arange(ii, ii+max_beta_len_B)

                    ii += max_beta_len_B
            
            beta_A_tens = np.zeros((1, 3, beta_vec_len))*np.nan
            beta_B_tens = np.zeros((1, 3, beta_vec_len_B))*np.nan

        beta_A_tens[0, 0, index_dict[var_name][:len(beta_A)]] = beta_A_transl
        beta_A_tens[0, 1, index_dict[var_name][:len(beta_A)]] = beta_mci_A
        beta_A_tens[0, 2, index_dict[var_name][:len(beta_A)]] = beta_pci_A
        
        # print(var_name)
        beta_B_tens[0, 0, index_dict_B[var_name][:len(beta_B)]] = beta_B
        beta_B_tens[0, 1, index_dict_B[var_name][:len(beta_B)]] = beta_mci_B
        beta_B_tens[0, 2, index_dict_B[var_name][:len(beta_B)]] = beta_pci_B

        idxA = fit_condA.covariate_significance['covariate'] == var_name
        idxB = fit_condB.covariate_significance['covariate'] == var_name
        
        var_sign_table['p-val cond %.4f'%cond_valA][cc] = fit_condA.covariate_significance['p-val'][idxA]
        var_sign_table['p-val cond %.4f'%cond_valB][cc] = fit_condB.covariate_significance['p-val'][idxB]

        # cov_B size is larger for control gain
        cov_B = cov_B[:beta_A.shape[0],:]
        cov_B = cov_B[:,:beta_A.shape[0]]

        eigV, U = np.linalg.eigh(cov_A + cov_B)

        nonZero = np.where(eigV > 10 ** -5)[0]
        diag = np.zeros(len(eigV))
        diag[nonZero] = 1/eigV[nonZero]
        pINV = np.dot(np.dot(U, np.diag(diag)), U.T)

        dof = len(nonZero)
        idxA = fit_condA.covariate_significance['covariate'] == var_name
        idxB = fit_condB.covariate_significance['covariate'] == var_name
        if not ((fit_condA.covariate_significance['p-val'][idxA] > 0.001) and \
                (fit_condB.covariate_significance['p-val'][idxB] > 0.001)):

            if dof < cov_B.shape[0] and not ('neu' in var_name):
                xxxx = 1

        chi2 = sts.chi2(dof)
        if (fit_condA.covariate_significance['p-val'][idxA] > 0.001) and (fit_condB.covariate_significance['p-val'][idxB] > 0.001):
            pval_list[cc] = 1.
        elif (max(eigA) < 10**-5) and (max(eigB) < 10**-5):
            pval_list[cc] = 1.
        # elif (max(eigA) < 10**-5):
        #     pval_list[cc] = 1-chi2.cdf(np.dot(np.dot(beta_B[:beta_A.shape[0]]-beta_A_transl,np.linalg.pinv(2*cov_B)),beta_B[:beta_A.shape[0]]-beta_A_transl))
        # elif (max(eigB) < 10**-5):
        #     pval_list[cc] = 1-chi2.cdf(np.dot(np.dot(beta_B[:beta_A.shape[0]]-beta_A_transl,np.linalg.pinv(2*cov_A)),beta_B[:beta_A.shape[0]]-beta_A_transl))
        else:
            pval_list[cc] = 1-chi2.cdf(np.dot(np.dot(beta_B[:beta_A.shape[0]]-beta_A_transl,pINV),beta_B[:beta_A.shape[0]]-beta_A_transl))
        # pval_list[cc] = 1-chi2.cdf(np.dot(np.dot(beta_B-beta_A,np.linalg.pinv(cov_A+cov_B)),beta_B-beta_A))
        # if not  var_name.startswith('neu'):
        #     prd = np.dot(np.dot(beta_B[:beta_A.shape[0]]-beta_A_transl,pINV,beta_B[:beta_A.shape[0]]-beta_A_transl)

            # print('mean sub',mean_sub,var_name,prd,pval_list[cc])
        cc+=1
    var_sign_table['p-val'] = pval_list
     
    
    return var_sign_table, index_dict, index_dict_B, beta_A_tens, beta_B_tens

npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/eval_matrix_and_info.npz',allow_pickle=True)
info = dat['info']

monkey = 'Schro'


condition_dict = {
    'controlgain':[[1],[1.5,2]],
    'ptb':[[0],[1]],
    'density':[[0.005],[0.0001]]
    }

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}


dtype_dict = {'names':['monkey','session','brain area','channel id','electrode id','cluster id','condition','value','unit','rate [Hz]'],
                               'formats':['U30','U30','U30',int,int,int,'U30',float, int,float]}


# check_cond = ['density','ptb']
sess_list = {'ptb':[]}#'controlgain':[],'density':[],'ptb':[]}
cond_extract = {'controlgain':{'controlgain':[1,2],'odd':[0,1]}, 'density':{'odd':[0,1],'density':[0.005,0.0001]},'ptb':{'odd':[0,1],'ptb':[0,1]}}
for root, dirs, files in os.walk('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'):
    for fh_name in files:
        for cond in sess_list.keys():
            if cond in fh_name:
                sess_list[cond] += [fh_name.split('results_')[1].split('_')[0]]
for cond in sess_list.keys():
    sess_list[cond] = np.unique(sess_list[cond])




# firing_rate = np.load('firing_rate_x_cond.npy')
for cond_out in sess_list.keys():
    for session in sess_list[cond_out]:
        # if session != 'm53s36':
        #     continue
        for cond in cond_extract[cond_out].keys():
            val1,val2 = cond_extract[cond_out][cond]
            pattern_cond1 = '^fit_results_%s_c\d+_%s_%.4f.dill$'%(session,cond,val1)
            pattern_cond2 = '^fit_results_%s_c\d+_%s_%.4f.dill$'%(session,cond,val2)

            cond_1_units = []
            cond_2_units = []
            for name in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/'%session):
                if re.match(pattern_cond1, name):
                    cond_1_units += [int(name.split('_c')[1].split('_')[0])]
                if re.match(pattern_cond2, name):
                    cond_2_units += [int(name.split('_c')[1].split('_')[0])]


            similarity_list = list(set(cond_2_units).intersection(cond_1_units))


            first = True
            # pval_list = np.zeros(len(similarity_list),dtype=float)
            # max_eigA = np.zeros(len(similarity_list),dtype=float)
            # max_eigB = np.zeros(len(similarity_list),dtype=float)

            cc = 0

            prev_dict = None
            # if not session in ['m53s36','m53s39']:
            #     continue
            for unit in similarity_list:
                print(session,unit,cond)
                # if unit!=121:
                #     continue

                fh_name_A = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_%s_%.4f.dill'%(session,session,unit,cond,val1)
                fh_name_B = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_%s_%.4f.dill'%(session,session,unit,cond,val2)

                var_sign,index_dict_A,index_dict_B,tensor_A,tensor_B = compute_pval_and_parameters(fh_name_A,fh_name_B,mean_sub=True)
                # var_sign_sub,index_dict_sub,tensor_A_sub,tensor_B_sub = compute_pval_and_parameters(fh_name_A,fh_name_B,True)
                if first:
                    var_sign_sess = deepcopy(var_sign)
                    index_dict_A = deepcopy(index_dict_A)
                    index_dict_B = deepcopy(index_dict_B)
                    tensor_A_sess = deepcopy(tensor_A)
                    tensor_B_sess = deepcopy(tensor_B)
                    first = False

                    # var_sign_sess_sub  = deepcopy(var_sign_sub)
                    # index_dict_sub  = deepcopy(index_dict_sub)
                    # tensor_A_sess_sub = deepcopy(tensor_A_sub)
                    # tensor_B_sess_sub  = deepcopy(tensor_B_sub)

                else:
                    var_sign_sess = np.hstack((var_sign_sess,var_sign))
                    tensor_A_sess = np.vstack((tensor_A_sess,tensor_A))
                    tensor_B_sess = np.vstack((tensor_B_sess,tensor_B))

                    # var_sign_sess_sub = np.hstack((var_sign_sess_sub,var_sign_sub))
                    # tensor_A_sess_sub = np.vstack((tensor_A_sess_sub,tensor_A_sub))
                    # tensor_B_sess_sub = np.vstack((tensor_B_sess_sub,tensor_B_sub))

                if prev_dict is None:
                    prev_dict = deepcopy(index_dict_A)

                else:
                    for var in index_dict_A.keys():
                        assert(all(index_dict_A[var] == prev_dict[var]))
                # break
            # counts_sub = {}
            # counts = {}
            # for var in index_dict.keys():
            #     counts_sub[var] = 0
            #     counts[var] = 0
            #
            # for row in var_sign_sess_sub[var_sign_sess_sub['p-val']<0.001]:
            #     counts_sub[row['variable']] = counts_sub[row['variable']] + 1
            #
            # for row in var_sign_sess[var_sign_sess['p-val']<0.001]:
            #     counts[row['variable']] = counts[row['variable']] + 1
            np.savez('/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_change/significance_tuning_function_change/newTest_%s_%s_tuningChange.npz'%(var_sign['session'][0],var_sign['condition'][0]),
                     unit_list=similarity_list,
                     index_dict_A=index_dict_A,index_dict_B=index_dict_B,tensor_A=tensor_A_sess,
                     tensor_B=tensor_B_sess,var_sign=var_sign_sess)
    

     
        
        
        