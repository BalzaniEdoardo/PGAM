#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:36:06 2020

@author: edoardo
"""
import numpy as np
import sys, os, dill
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
# sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline/util_preproc'))
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils/'))
from utils_loading import unpack_preproc_data, add_smooth
from GAM_library import *
from time import perf_counter
import statsmodels.api as sm
from basis_set_param_per_session import *
from knots_util import *
from path_class import get_paths_class
import statsmodels.api as sm
import matplotlib.pylab as plt
from copy import deepcopy
from time import perf_counter
from scipy.io import savemat


session = 'm53s91'
dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/eval_matrix_and_info.npz')
info = dat['info']
sele = info['session'] == session
info = info[sele]

info_selectivity = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/response_strength_info.npy')
keep = ((info_selectivity['session'] == session) * 
    (info_selectivity['manipulation type']=='odd')*
    ((info_selectivity['manipulation value'] == 0)|
    (info_selectivity['manipulation value'] == 1))
)
info_selectivity = info_selectivity[keep]
info_selectivity = info_selectivity[info_selectivity['rad_vel']]


unit_list = np.sort(np.unique(info_selectivity['unit']))

plot_boolean = 1
skip_first = 0
if plot_boolean:
    cck = 0

    for unit in unit_list:#dill_name in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/'%session):
        # load file
        # if unit !
        
        # if not '_controlgain_' in dill_name:
        #     continue
        if cck < skip_first:
            cck+=1
            continue
        try:
            # unit = int(dill_name.split('_controlgain_')[0].split('c')[-1])
            with open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_odd_1.0000.dill'%(session,session,unit),'rb') as fh:
                result_dict = dill.load(fh)
            
            with open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s/fit_results_%s_c%d_odd_0.0000.dill'%(session,session,unit),'rb') as fh:
                result_dict2 = dill.load(fh)
        except:
            continue
        # fit_slow = result_dict['fit_slow']
        fit_fast = result_dict2['full']
        lab_red = 'gain 2'
        fit_slow = result_dict['full']
        lab_ful = 'gain 1'
        
        
        # sm_handler_gam = smooths_handler()
        if fit_fast is None:
            continue
        xxdict = {}
        for var in fit_fast.var_list:
            knots = fit_fast.smooth_info[var]['knots'][0]
            xmin = fit_fast.smooth_info[var]['xmin'][0]
            xmax = fit_fast.smooth_info[var]['xmax'][0]
            if var == 'ang_vel':
                xmin = -50
                xmax = 50
            if var == 'rad_vel':
                xmin = 0
                xmax = 390
            if not fit_fast.smooth_info[var]['is_temporal_kernel']:
                x = np.linspace(xmin,xmax,100)
                xxdict[var] = x
            else:
                dim_kern = fit_fast.smooth_info[var]['basis_kernel'].shape[0]
                knots_num = fit_fast.smooth_info[var]['knots'][0].shape[0]
                x = np.zeros(dim_kern)
                x[(dim_kern-1)//2] = 1
                xxdict[var] = x
                
            # sm_handler_gam = add_smooth(sm_handler_gam, x, var, knots, session, np.ones(len(x)), time_bin=0.006, lam=50)
            
            
        
        # sm_handler_gam_fit_slow = smooths_handler()
        xxdict_slow = {}
        for var in fit_slow.var_list:
            knots = fit_slow.smooth_info[var]['knots'][0]
            xmin = fit_slow.smooth_info[var]['xmin'][0]
            xmax = fit_slow.smooth_info[var]['xmax'][0]
            if var == 'ang_vel':
                xmin = -50
                xmax = 50
            if var == 'rad_vel':
                xmin = 0
                xmax = 190
            if not fit_slow.smooth_info[var]['is_temporal_kernel']:
                x = np.linspace(xmin,xmax,100)
                xxdict_slow[var] = x
            else:
                dim_kern = fit_slow.smooth_info[var]['basis_kernel'].shape[0]
                knots_num = fit_slow.smooth_info[var]['knots'][0].shape[0]
                x = np.zeros(dim_kern)
                x[(dim_kern-1)//2] = 1
                xxdict_slow[var] = x
                
            # sm_handler_gam_fit_slow = add_smooth(sm_handler_gam_fit_slow, x, var, knots, session, np.ones(len(x)), time_bin=0.006, lam=50)
            
            
            
            
        plt.figure(figsize=(12,10))
        plt.suptitle(session + ' unit %d'%unit )

        ax_dict = {}
        
        k = 1
        for var in fit_slow.var_list:#['rad_vel']:
            if var.startswith('neu'):
                continue
            ax_dict[var] = plt.subplot(5,4,k)
            ax = ax_dict[var]
            ax.set_title(var)
            
            fX,fX_p,fX_m = fit_slow.smooth_compute([xxdict[var]],var,0.99)
            if (not var.startswith('t_')) or var == 'spike_hist':
                fX[xxdict[var] <= xxdict_slow[var][0]] = np.nan
                fX[xxdict[var] >= xxdict_slow[var][-1]] = np.nan
                fX_p[xxdict[var] <= xxdict_slow[var][0]] = np.nan
                fX_p[xxdict[var] >= xxdict_slow[var][-1]] = np.nan
                fX_m[xxdict[var] <= xxdict_slow[var][0]] = np.nan
                fX_m[xxdict[var] >= xxdict_slow[var][-1]] = np.nan
            # X = sm_handler_gam[var].X.toarray()
            # X = X[:,:-1] - np.mean(X[:,:-1],axis=0)
            # X = np.hstack((np.ones((X.shape[0],1)),X))
        
            # pred = np.dot(X[:,1:],fit_fast.beta[fit_fast.index_dict[var]])
            xx = np.arange(xxdict_slow[var].shape[0])
            # if not var in fit_fast.var_list:
            idx = np.where(fit_slow.covariate_significance['covariate'] == var)[0]

            if fit_slow.covariate_significance['p-val'][idx]<0.001:
                color='g'
            else:
                color='y'
            ax.plot(xx,fX,color=color,label=lab_ful)

            ax.fill_between(xx,fX_m,fX_p,color=color,alpha=0.3)
            if var == 'spike_hist':
                ax.legend()
            ax.set_xticks([])
            
            
            k += 1
            
        k = 1
        for var in fit_fast.var_list:#['rad_vel']:
            if var.startswith('neu'):
                continue
            
            ax = ax_dict[var]
            fX,fX_p,fX_m = fit_fast.smooth_compute([xxdict[var]],var,0.99)
            
            xx = np.arange(fX.shape[0])
            idx = np.where(fit_fast.covariate_significance['covariate'] == var)[0]
            if fit_fast.covariate_significance['p-val'][idx]<0.001:
                color='r'
            else:
                color='k'
            ax.plot(xx,fX,color=color,label=lab_red)
            
            ax.fill_between(xx,fX_m,fX_p,color=color,alpha=0.3)
            
            k += 1
            
        
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        
        # plt.legend()
        # plt.savefig('Figs/example_tuning_%s_%d.png'%(session,unit))
        # plt.close('all')
        # break
        # break

        # var = 'rad_vel'
        # xx_400 = np.linspace(0,390,100)
        # xx_200 = np.linspace(0,180,100)
        # fX_400, fX_p_ci_400, fX_m_ci_400 = fit_fast.smooth_compute([xx_400], 'rad_vel', perc=0.99)
        # fX_200, fX_p_ci_200, fX_m_ci_200 = fit_slow.smooth_compute([xx_200], 'rad_vel', perc=0.99)
        
        
        # plt.figure()
        # p, = plt.plot(xx_200,fX_200,label='controlgain=1')
        # plt.fill_between(xx_200, fX_m_ci_200, fX_p_ci_200, color=p.get_color(), alpha=0.4)
        
        
        # p,=plt.plot(xx_400,fX_400,label='controlgain=2')
        # plt.fill_between(xx_400, fX_m_ci_400, fX_p_ci_400, color=p.get_color(), alpha=0.4)
        
        # plt.legend()
        
        # plt.tight_layout()
        # plt.savefig('Figs/example_tuning_%s_%d.png'%(session,unit))
        # plt.close('all')
# plot hist
# fld_npz = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'
# dat_npz = np.load(fld_npz+'%s.npz'%(session),allow_pickle=True)
# info_trial = dat_npz['info_trial'].all()
# concat = dat_npz['data_concat'].all()
# X = concat['Xt']
# var_ordered = dat_npz['var_names']
# x_rad_vel = X[:,var_ordered=='rad_vel']

# fast_tr = np.where(info_trial.trial_type['controlgain'] == 2)[0]
# slow_tr = np.where(info_trial.trial_type['controlgain'] == 1)[0]

# trial_idx = concat['trial_idx']
# sele_fast = np.zeros(trial_idx.shape,dtype=bool)
# sele_slow = np.zeros(trial_idx.shape,dtype=bool)
# for tr in fast_tr:
#     sele_fast[trial_idx==tr] = True
    
# for tr in slow_tr:
#     sele_slow[trial_idx==tr] = True

# plt.figure()
# hst_fast = plt.hist(x_rad_vel[sele_fast],alpha=0.4,label='gain=2')
# hst_slow = plt.hist(x_rad_vel[sele_slow],alpha=0.4,label='gain=1')
# plt.legend()
# knots_fast = fit_fast.smooth_info['rad_vel']['knots']






# # plt.savefig('tuning_example.pdf')




# # GET NEURON INFO
# dat = np.load('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/%s.npz'%session,allow_pickle=True)
# unit_info = dat['unit_info'].all()
# cluster_id = unit_info['cluster_id']
# electrode_id = unit_info['electrode_id']
# brain_area = unit_info['brain_area']
# channel_id = unit_info['channel_id']


# variables = dat['var_names']
# keep = np.ones(variables.shape[0],dtype=bool)
# k = 0
# for var in variables:
#     if var.startswith('neu_'):
#         keep[k] = False
#     k+=1

# variables = variables[keep]
# dict_type = {
#     'names':('session','unit','cluster_id','electrode_id','channel_id','brain_area',)+tuple(variables),
#     'formats':('U30',) + (int,)*4 + ('U3',) + (bool,)*variables.shape[0]
#         }



# table_report = np.zeros(0,dtype=dict_type)


# for gam_sess in os.listdir('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/GAM_comparison_m91s25/'):
#     if not gam_sess.startswith('gam'):
#         continue
#     session = gam_sess.split('_')[1]
#     try:
#         dat = np.load('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/%s.npz'%session,allow_pickle=True)
#     except:
#         continue
#     unit_info = dat['unit_info'].all()
#     cluster_id = unit_info['cluster_id']
#     electrode_id = unit_info['electrode_id']
#     brain_area = unit_info['brain_area']
#     channel_id = unit_info['channel_id']

#     for dill_name in os.listdir(os.path.join('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/GAM_comparison_m91s25/',gam_sess)):
#         print(dill_name + '....works')
#         if not 'all' in dill_name:
#             continue
#         try:
#             with open(os.path.join('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/GAM_comparison_m91s25/',gam_sess,dill_name),'rb') as fh:
#                 result_dict = dill.load(fh)
            
            
            
#             fit_slow = result_dict['fit_slow']
#             fit_fast = result_dict['fit_fast']
#             unit = int(dill_name.split('_all_')[0].split('c')[-1])
           

#             variables = np.copy(fit_slow.var_list)
#             keep = np.ones(variables.shape[0],dtype=bool)
#             k = 0
#             for var in fit_slow.var_list:
#                 if var.startswith('neu_'):
#                     keep[k] = False
#                 k+=1
            
#             variables = variables[keep]
            
#             table_report_tmp = np.zeros(1,dtype=dict_type)

            
#             table_report_tmp['session'][0] = session
#             table_report_tmp['unit'][0] = unit
            
#             # matlab indexing was used for the name
#             table_report_tmp['cluster_id'][0] = cluster_id[unit-1]
#             table_report_tmp['channel_id'][0] = channel_id[unit-1]
#             table_report_tmp['electrode_id'][0] = electrode_id[unit-1]
#             table_report_tmp['brain_area'][0] = brain_area[unit-1]
            
#             for var in fit_fast.var_list:
#                 if var == 'spike_hist':
#                     continue
#                 if var.startswith('neu') :
#                     continue
#                 idx = np.where(fit_fast.covariate_significance['covariate'] == var)[0]
#                 if fit_fast.covariate_significance['p-val'][idx]<0.001:
#                     table_report_tmp[var][0] = True
                    
#             # if you want to stack another neuron you can do this
#             # insert the info and then
#             table_report = np.hstack((table_report,table_report_tmp))
            
            
#             # this saves it as a struct
           
                
                
                
#         except Exception as e:
#             print(e)
#             #print("can't open ",os.path.join('/Users/jean-paulnoel/Documents/Savin-Angelaki/saved/GAM_results_Schro/',gam_sess,dill_name))
        
            
    
# savemat('table_report.mat',{'table_report':table_report})            

    
    


                    