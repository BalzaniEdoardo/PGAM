#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:56:20 2021

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
from matplotlib import cm
import seaborn as sns

plt.close('all')


npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel'
tuning_change_fld = '/Users/edoardo/Work/Code/GAM_code/analyzing/tuning_change/significance_tuning_function_change/'
condition = 'controlgain'
ba = 'PFC'
session = 'm53s42'

dtype_dict = {'names':('monkey','session','unit','condition','brain_area','variable','p-val','is sign 0.0001','is sign 0.0050'),
                                 'formats':('U30','U30',int,'U30','U30','U30',float,bool,bool)}


dtype_dict2 = {'names':('monkey','session','condition','brain_area','variable','unit','density','significance'),
                                 'formats':('U30','U30','U30','U30','U30',int,float, bool)}
result_table = np.zeros(0,dtype=dtype_dict)
result_table2 = np.zeros(0,dtype=dtype_dict2)

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}

lst_files = os.listdir(tuning_change_fld)
pattern = '^newTest_m\d+s\d+_[a-z]+_tuningChange.npz$'
for fh in lst_files:
    if not re.match(pattern,fh):
        continue

    splt = fh.split('_')
    session = splt[1]
    condition = splt[2]
    monk = monkey_dict[session.split('s')[0]]

    if condition!='density':
        continue

    print(session,condition)
    dat = np.load(os.path.join(tuning_change_fld,fh),
        allow_pickle=True)

    npz_dat = np.load(os.path.join(npz_folder,'%s.npz'%(session)),
        allow_pickle=True)

    unit_info = npz_dat['unit_info'].all()
    brain_area = unit_info['brain_area']

    # tensor_A = dat['tensor_A']
    # tensor_B = dat['tensor_B']
    # index_dict_A = dat['index_dict_A'].all()
    # index_dict_B = dat['index_dict_B'].all()


    var_sign = dat['var_sign']
    sele_non_coupling = np.zeros(var_sign.shape,dtype=bool)

    cc = 0
    var_vector2 = var_sign['variable']

    var_vector = []
    for var in var_sign['variable']:
        if var.startswith('neu') or var.startswith('spike_hist'):
            cc+=1
            continue
        sele_non_coupling[cc] = True

        if var in ['rad_vel','ang_vel']:
            var_vector += ['sensory']

        elif var in ['rad_acc','ang_acc']:
            var_vector += ['acceleration']

        elif var in ['t_stop','t_move']:
            var_vector += ['move ON/OFF']

        elif var in ['eye_hori','eye_vert']:
            var_vector += ['eye']

        elif var in ['ang_path','rad_path']:
            var_vector += ['dist orig']

        elif var in ['rad_target','ang_target']:
            var_vector += ['dist targ']
        else:
            var_vector += [var]
        cc+=1

    var_sign = var_sign[sele_non_coupling]

    var_vector2 = var_vector2[sele_non_coupling]

    var_vector = var_sign['variable']

    # unit_list = dat['unit_list']

    tmp = np.zeros(var_sign.shape[0], dtype=dtype_dict)
    tmp['variable'] = var_vector#var_sign['variable']
    tmp['session'] = session
    tmp['condition'] = condition
    tmp['brain_area'] = brain_area[var_sign['unit']-1]
    tmp['p-val'] = var_sign['p-val']
    tmp['monkey'] = monk
    tmp['is sign 0.0001'] = var_sign['p-val cond 0.0001'] < 0.001
    tmp['is sign 0.0050'] = var_sign['p-val cond 0.0050'] < 0.001
    tmp['unit'] = var_sign['unit']

    result_table = np.hstack((result_table,tmp))

    tmp2 = np.zeros(var_vector2.shape[0]*2, dtype=dtype_dict2)
    tmp2['variable'] = np.hstack((var_vector2,var_vector2)) # var_sign['variable']
    tmp2['session'] = session
    tmp2['condition'] = condition
    tmp2['brain_area'][:var_vector2.shape[0]] = brain_area[var_sign['unit'] - 1]
    tmp2['brain_area'][var_vector2.shape[0]:] = brain_area[var_sign['unit'] - 1]
    tmp2['unit'][:var_vector2.shape[0]] = var_sign['unit']
    tmp2['unit'][var_vector2.shape[0]:] = var_sign['unit']

    tmp2['monkey'] = monk
    tmp2['density'][:var_vector2.shape[0]] = 0.005
    tmp2['density'][var_vector2.shape[0]:] = 0.0001
    tmp2['significance'][:var_vector2.shape[0]] = var_sign['p-val cond 0.0050'] < 0.001
    tmp2['significance'][var_vector2.shape[0]:] = var_sign['p-val cond 0.0001'] < 0.001

    result_table2 = np.hstack((result_table2,tmp2))


df = pd.DataFrame(result_table2)
df_ppc = df[df['brain_area']=='PPC']
df_pfc = df[df['brain_area']=='PFC']
df_mst = df[df['brain_area']=='MST']


# order = ['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF','rad_target',
#          'ang_target','rad_path','ang_path','lfp_beta','lfp_alpha','lfp_theta','t_reward','eye_vert','eye_hori']


order = ['rad_vel','ang_vel','t_stop','t_reward','eye_vert']

cm_pfc = cm.get_cmap('Reds')
cm_ppc = cm.get_cmap('Blues')
cm_mst = cm.get_cmap('Greens')



# figure pfc
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_pfc,
            palette=sns.color_palette([cm_pfc(0.5)[:3],cm_pfc(0.9)[:3]]),height=5,aspect=1.,legend_out=False)
plt.xticks(rotation=90)
plt.title('PFC fraction tuned')

plt.tight_layout()
plt.ylim(0,0.8)

plt.savefig('sub_condtion_frac_tuned_pfc.png')


# figure ppc
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_ppc,
            palette=sns.color_palette([cm_ppc(0.5)[:3],cm_ppc(0.9)[:3]]),height=5,aspect=1.,legend_out=False)
plt.xticks(rotation=90)
plt.title('PPC fraction tuned')

plt.tight_layout()
plt.ylim(0,0.8)

plt.savefig('sub_condtion_frac_tuned_ppc.png')


# figure mst
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_mst,
            palette=sns.color_palette([cm_mst(0.5)[:3],cm_mst(0.9)[:3]]),height=5,aspect=1.,legend_out=False)
plt.xticks(rotation=90)
plt.title('MST fraction tuned')

plt.tight_layout()
plt.ylim(0,0.8)
plt.savefig('sub_condtion_frac_tuned_mst.png')




### QUIGLEY

df_ppc = df[(df['brain_area']=='PPC') & (df['monkey']=='Quigley')]
# df_pfc = df[df['brain_area']=='PFC']
df_mst = df[(df['brain_area']=='MST') & (df['monkey']=='Quigley')]
# figure pfc

# figure ppc
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_ppc,
            palette=sns.color_palette([cm_ppc(0.5)[:3],cm_ppc(0.9)[:3]]),height=5,aspect=2.5,legend_out=False)
plt.xticks(rotation=90)
plt.title('Quigley PPC fraction tuned')

plt.tight_layout()


# figure mst
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_mst,
            palette=sns.color_palette([cm_mst(0.5)[:3],cm_mst(0.9)[:3]]),height=5,aspect=2.5,legend_out=False)
plt.xticks(rotation=90)
plt.title('Quigley MST fraction tuned')

plt.tight_layout()



### SCHRO

df_ppc = df[(df['brain_area']=='PPC') & (df['monkey']=='Schro')]
df_pfc = df[(df['brain_area']=='PFC') & (df['monkey']=='Schro')]
df_mst = df[(df['brain_area']=='MST') & (df['monkey']=='Schro')]
# figure pfc

# figure ppc
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_pfc,
            palette=sns.color_palette([cm_pfc(0.5)[:3],cm_pfc(0.9)[:3]]),height=5,aspect=2.5,legend_out=False)
plt.xticks(rotation=90)
plt.title('Schro PFC fraction tuned')

plt.tight_layout()

# figure ppc
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_ppc,
            palette=sns.color_palette([cm_ppc(0.5)[:3],cm_ppc(0.9)[:3]]),height=5,aspect=2.5,legend_out=False)
plt.xticks(rotation=90)

plt.title('Schro PPC fraction tuned')

plt.tight_layout()


# figure mst
sns.catplot(x="variable", y="significance", hue="density", kind="bar", order=order, data=df_mst,
            palette=sns.color_palette([cm_mst(0.5)[:3],cm_mst(0.9)[:3]]),height=5,aspect=2.5,legend_out=False)
plt.xticks(rotation=90)
plt.title('Schro MST fraction tuned')

plt.tight_layout()

# plt.close('all')
#
#
#
# # plot some tuning changes
# result_table = result_table[(result_table['is sign 0.0001']) | (result_table['is sign 0.0050'])]
# idx_sort = np.argsort(result_table['p-val'])
# sbcnt = 1
# plt.figure(figsize=(10, 8))
# for k in range(25):
#     tmp = result_table[idx_sort[15000:]]
#     bl = (tmp['variable'] != 'lfp_beta') & (tmp['variable'] != 'lfp_alpha') & (tmp['variable'] != 'lfp_theta')
#     res = tmp[bl][k]
#
#     neuron_id = res['unit']
#     session = res['session']
#     variable = res['variable']
#
#     dat = np.load(os.path.join(tuning_change_fld,'newTest_%s_density_tuningChange.npz'%session), allow_pickle=True)
#
#     tensor_A = dat['tensor_A']
#     tensor_B = dat['tensor_B']
#     index_dict_A = dat['index_dict_A'].all()
#     index_dict_B = dat['index_dict_B'].all()
#
#
#     var_sign = dat['var_sign']
#     unit_list = dat['unit_list']
#
#     unit = np.where(unit_list==neuron_id)[0][0]
#
#     #variable = 't_stop'
#     pv_th = 0.01
#
#
#
#
#
#     plt.subplot(5, 5, sbcnt)
#
#     neuron_id = unit_list[unit]
#     filt = (var_sign['variable'] == variable) * (var_sign['unit'] == neuron_id)
#     pval = var_sign[filt]['p-val']
#
#     plt.title('%s %.3f' % (variable, pval[0]))
#     plt.plot(tensor_A[unit, 0, index_dict_A[variable]])
#     plt.fill_between(range(index_dict_A[variable].shape[0]), tensor_A[unit, 1, index_dict_A[variable]],
#                      tensor_A[unit, 2, index_dict_A[variable]], alpha=0.4)
#
#     plt.plot(tensor_B[unit, 0, index_dict_B[variable]])
#     plt.fill_between(range(index_dict_B[variable].shape[0]), tensor_B[unit, 1, index_dict_B[variable]],
#                      tensor_B[unit, 2, index_dict_B[variable]], alpha=0.4)
#     sbcnt += 1
# plt.tight_layout()


# condition_list = ['controlgain','ptb','density','odd']
# var_list = ['sensory','acceleration', 'move ON/OFF','dist orig','dist targ','t_flyOFF', 't_reward','eye']
# for ba in np.unique(result_table['brain_area']):
#     res_ba = result_table[(result_table['brain_area'] == ba)]
#
#     plt.figure(figsize=(10,6))
#     plt.suptitle(ba)
#     sbplt_cnt = 1
#     for var in var_list:
#         ax = plt.subplot(4,2,sbplt_cnt)
#         sbplt_cnt+=1
#
#         vec_mean_changed = []
#         vec_cond = []
#         for cond in condition_list:
#
#             res_cond = res_ba[(res_ba['condition']==cond) *(res_ba['variable']==var)]
#             if res_cond.shape[0] == 0:
#                 continue
#
#             vec_mean_changed += [(res_cond['p-val']<0.001).mean()]
#             vec_cond += [cond]
#
#         plt.barh(range(len(vec_cond)),vec_mean_changed[::-1])
#         plt.yticks(range(len(vec_cond)),vec_cond[::-1])
#         plt.xlim(0,1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         plt.title(var)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # plt.savefig('%s_tuningChange.pdf'%(ba))
#
#         # df = pd.DataFrame(res_cond)
#         # df['p-val'] = df['p-val'] < 0.001
#         # data = df.groupby(["variable"])["p-val"].mean()
#         # plt.figure()
#         # plt.title(ba + ' ' + cond)
#         # data.plot.pie()
# # get the npz