#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:59:52 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from matplotlib import collections  as mc

plt.figure(figsize=(10,8))
plt.suptitle('Delta R^2: High Density - Low Density')
use_test = 'indep test'
cnt = 1
dict_ylim = {}
# for var in ['rad_target','rad_path','ang_target','ang_path','rad_vel','ang_vel',
#            'eye_vert','t_flyOFF']:
    
#     plt.subplot(4,2,cnt)
#     cnt+=1
#     dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/decoding/decoding_density_%s.npy'%var,allow_pickle=True).all()
    
    
    
    
#     ba_result = {}
#     for ba in ['PFC','PPC','MST']:
        
#         num_sess = len(dat[ba].keys())
#         ba_result[ba] = np.zeros((2,num_sess))
#         cc = 0
#         for session in dat[ba].keys():
#             ba_result[ba][0,cc] = dat[ba][session]['LD %s'%use_test]
#             ba_result[ba][1,cc] = dat[ba][session]['HD %s'%use_test]
            
#             cc+=1
#         print(ba, var, ba_result[ba])
#     for ba in ['PFC','PPC','MST']:
#         ba_result[ba] = ba_result[ba][:, (ba_result[ba][1,:] > 0.) |
#                                         (ba_result[ba][0,:] > 0.)]
#     color = {'MST':'g','PPC':'b','PFC':'r'}
#     plt.title('%s'%var)
#     kk=0
#     for ba in ['MST','PPC','PFC']:
#         plt.boxplot(
#                  ba_result[ba][1]-ba_result[ba][0],positions=[kk])
#         kk+=1
#     plt.xticks([0,1,2],['MST','PPC','PFC'])
#     plt.plot([0,2],[0,0],'--r')
#     dict_ylim[cnt-1] = plt.ylim()
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# # plt.savefig('r2_HD_vs_LD_%s.png'%use_test)


# plt.figure(figsize=(10,8))

# cnt = 1
# for var in ['rad_target','rad_path','ang_target','ang_path','rad_vel','ang_vel',
#            'eye_vert','t_flyOFF']:
    
#     ax = plt.subplot(4,2,cnt)
#     cnt+=1
#     dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/decoding/decoding_density_%s.npy'%var,allow_pickle=True).all()
    
    
    
    
#     ba_result = {}
#     for ba in ['PFC','PPC','MST']:
        
#         num_sess = len(dat[ba].keys())
#         ba_result[ba] = np.zeros((2,num_sess))
#         cc = 0
#         for session in dat[ba].keys():
#             ba_result[ba][0,cc] = dat[ba][session]['LD %s'%use_test]
#             ba_result[ba][1,cc] = dat[ba][session]['HD %s'%use_test]
            
#             cc+=1
#         print(ba, var, ba_result[ba])
#     for ba in ['PFC','PPC','MST']:
#         ba_result[ba] = ba_result[ba][:, (ba_result[ba][1,:] > 0.) |
#                                         (ba_result[ba][0,:] > 0.)]
#     color = {'MST':'g','PPC':'b','PFC':'r'}
#     plt.title('%s'%var)
    
#     xx_axis = 0
#     min_x = 10000
#     max_x = -10000
#     for ba in ['MST','PPC','PFC']:
#         lines = []
#         for k  in range(ba_result[ba].shape[1]):
#             lines += [[(xx_axis-0.5,ba_result[ba][0][k]), (xx_axis+1.5,ba_result[ba][1][k])]]
#         min_x = min(lines[-1][0][1],min_x ,lines[-1][1][1])
#         max_x = max(lines[-1][0][1],max_x ,lines[-1][1][1])
#         xx_axis += 3
#         lc = mc.LineCollection(lines, colors=color[ba], linewidths=1.5,label=ba)
#         ax.add_collection(lc)
        
#     # kk=0
#     # for ba in ['MST','PPC','PFC']:
#     #     plt.boxplot(
#     #              ba_result[ba][1]-ba_result[ba][0],positions=[kk])
#     #     kk+=1
#     # plt.xticks([0.5,3.5,6.5],['MST','PPC','PFC'])
    
#     plt.xticks([-0.5,1.5,2.5,4.5,5.5,7.5],['LD','HD','LD','HD','LD','HD'])
    
#     plt.ylim(min_x-0.5,max_x+0.3)
#     plt.xlim(-2,9)
#     plt.legend(loc=2)
#     # plt.plot(,[0,0],'--r')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('accuracy_decoding_density.png')



plt.figure(figsize=(10,8))

cnt = 1
for var in ['rad_target','rad_path','ang_target','ang_path','rad_vel','ang_vel',
           'eye_hori','t_flyOFF']:
    
    ax = plt.subplot(4,2,cnt)
    cnt+=1
    # if var !='eye_vert':
    #     continue
    dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/decoding/decoding_density_%s.npy'%var,allow_pickle=True).all()
    
   
    
    
    ba_result = {}
    for ba in ['PFC','PPC','MST']:
        
        num_sess = len(dat[ba].keys())
        ba_result[ba] = np.zeros((2,num_sess))
        cc = 0
        for session in dat[ba].keys():
            dat_npz = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,allow_pickle=True)
    
            unit_info = dat_npz['unit_info'].all()
            if (unit_info['brain_area'] == ba).sum() < 35:
                ba_result[ba][0,cc] = np.nan
                ba_result[ba][1,cc] = np.nan
            else:
                ba_result[ba][0,cc] = dat[ba][session]['LD %s'%use_test]
                ba_result[ba][1,cc] = dat[ba][session]['HD %s'%use_test]
            
            cc+=1
        print(ba, var, ba_result[ba])
    for ba in ['PFC','PPC','MST']:
        ba_result[ba] = ba_result[ba][:, np.abs(ba_result[ba][1,:] )<1]
    color = {'MST':'g','PPC':'b','PFC':'r'}
    plt.title('%s'%var)
    
    xx_axis = 0
    
    mst_mean = np.nanmean(ba_result['MST'],axis=1)
    mst_std = ba_result['MST'].std(axis=1)/np.sqrt(ba_result['MST'][:,~np.isnan(ba_result['MST'][0])].shape[1])
    
    ppc_mean = np.nanmean(ba_result['PPC'], axis=1)
    ppc_std = np.nanstd(ba_result['PPC'], axis=1)/np.sqrt(ba_result['PPC'][:,~np.isnan(ba_result['PPC'][0])].shape[1])
    
    pfc_mean = np.nanmean(ba_result['PFC'], axis=1)
    pfc_std = np.nanstd(ba_result['PFC'], axis=1)/np.sqrt(ba_result['PFC'][:,~np.isnan(ba_result['PFC'][0])].shape[1])
    
    plt.errorbar([0,1],mst_mean,yerr=mst_std,color='g')
    
    plt.errorbar([3,4],ppc_mean,yerr=ppc_std,color='b')
    
    plt.errorbar([6,7],pfc_mean,yerr=pfc_std,color='r')
    # min_x = 10000
    # max_x = -10000
    
    # mean_mst = ba_result['MST'][]
    # for ba in ['MST','PPC','PFC']:
    #     lines = []
    #     for k  in range(ba_result[ba].shape[1]):
    #         lines += [[(xx_axis-0.5,ba_result[ba][0][k]), (xx_axis+1.5,ba_result[ba][1][k])]]
    #     min_x = min(lines[-1][0][1],min_x ,lines[-1][1][1])
    #     max_x = max(lines[-1][0][1],max_x ,lines[-1][1][1])
    #     xx_axis += 3
    #     lc = mc.LineCollection(lines, colors=color[ba], linewidths=1.5,label=ba)
    #     ax.add_collection(lc)
        
    # kk=0
    # for ba in ['MST','PPC','PFC']:
    #     plt.boxplot(
    #              ba_result[ba][1]-ba_result[ba][0],positions=[kk])
    #     kk+=1
    # plt.xticks([0.5,3.5,6.5],['MST','PPC','PFC'])
    
    plt.xticks([0,1,3,4,6,7],['LD','HD','LD','HD','LD','HD'])
    plt.plot([0,7],[0,0],'--k')
    if var != 't_flyOFF':
        plt.ylim(-0.1,0.35)
    
    # plt.ylim(min_x-0.5,max_x+0.3)
    # plt.xlim(-2,9)
    # plt.legend(loc=2)
    # plt.plot(,[0,0],'--r')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('accuracy_decoding_density_mean_%s.png'%use_test)



plt.figure(figsize=(10,8))
test_A = 'indep test'
test_B = 'opposite test'

cnt = 1
for var in ['rad_target','rad_path','ang_target','ang_path','rad_vel','ang_vel',
           'eye_hori','t_flyOFF']:
    
    ax = plt.subplot(4,2,cnt)
    cnt+=1
    # if var !='eye_vert':
    #     continue
    dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/decoding/decoding_density_%s.npy'%var,allow_pickle=True).all()
    
   
    
    
    ba_result = {}
    for ba in ['PFC','PPC','MST']:
        
        num_sess = len(dat[ba].keys())
        ba_result[ba] = np.zeros((2,num_sess))
        cc = 0
        for session in dat[ba].keys():
            dat_npz = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz'%session,allow_pickle=True)
    
            unit_info = dat_npz['unit_info'].all()
            if (unit_info['brain_area'] == ba).sum() < 35:
                ba_result[ba][0,cc] = np.nan
                ba_result[ba][1,cc] = np.nan
            else:
                ba_result[ba][0,cc] = dat[ba][session]['LD %s'%test_A]
                ba_result[ba][1,cc] = dat[ba][session]['LD %s'%test_B]
            
            cc+=1
        print(ba, var, ba_result[ba])
    for ba in ['PFC', 'PPC','MST']:
        ba_result[ba] = ba_result[ba][:, np.abs(ba_result[ba][1,:] )<1]
    color = {'MST':'g','PPC':'b','PFC':'r'}
    plt.title('%s'%var)
    
    xx_axis = 0
    
    mst_mean = np.nanmean(ba_result['MST'],axis=1)
    mst_std = ba_result['MST'].std(axis=1)/np.sqrt(ba_result['MST'][:,~np.isnan(ba_result['MST'][0])].shape[1])
    
    ppc_mean = np.nanmean(ba_result['PPC'], axis=1)
    ppc_std = np.nanstd(ba_result['PPC'], axis=1)/np.sqrt(ba_result['PPC'][:,~np.isnan(ba_result['PPC'][0])].shape[1])
    
    pfc_mean = np.nanmean(ba_result['PFC'], axis=1)
    pfc_std = np.nanstd(ba_result['PFC'], axis=1)/np.sqrt(ba_result['PFC'][:,~np.isnan(ba_result['PFC'][0])].shape[1])
    
    plt.errorbar([0,1],mst_mean,yerr=mst_std,color='g')
    
    plt.errorbar([3,4],ppc_mean,yerr=ppc_std,color='b')
    
    plt.errorbar([6,7],pfc_mean,yerr=pfc_std,color='r')
    # min_x = 10000
    # max_x = -10000
    
    # mean_mst = ba_result['MST'][]
    # for ba in ['MST','PPC','PFC']:
    #     lines = []
    #     for k  in range(ba_result[ba].shape[1]):
    #         lines += [[(xx_axis-0.5,ba_result[ba][0][k]), (xx_axis+1.5,ba_result[ba][1][k])]]
    #     min_x = min(lines[-1][0][1],min_x ,lines[-1][1][1])
    #     max_x = max(lines[-1][0][1],max_x ,lines[-1][1][1])
    #     xx_axis += 3
    #     lc = mc.LineCollection(lines, colors=color[ba], linewidths=1.5,label=ba)
    #     ax.add_collection(lc)
        
    # kk=0
    # for ba in ['MST','PPC','PFC']:
    #     plt.boxplot(
    #              ba_result[ba][1]-ba_result[ba][0],positions=[kk])
    #     kk+=1
    # plt.xticks([0.5,3.5,6.5],['MST','PPC','PFC'])
    
    plt.xticks([0,1,3,4,6,7],['ind','opp','ind','opp','ind','opp'])
    plt.plot([0,7],[0,0],'--k')
    if var != 't_flyOFF':
        plt.ylim(-0.1,0.35)
    
    # plt.ylim(min_x-0.5,max_x+0.3)
    # plt.xlim(-2,9)
    # plt.legend(loc=2)
    # plt.plot(,[0,0],'--r')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('accuracy_decoding_density_mean_indep_vs_opposite.png')

# plt.figure(figsize=(10,8))
# plt.suptitle('Delta R^2: High Density - Low Density')
# use_test = 'same test'
# cnt = 1
# for var in['rad_target','rad_path','ang_target','ang_path','rad_vel','ang_vel',
#            'eye_vert','eye_hori']:
    
#     plt.subplot(4,2,cnt)
#     cnt+=1
#     dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/decoding/decoding_density_%s.npy'%var,allow_pickle=True).all()
    
    
    
    
#     ba_result = {}
#     for ba in ['PFC','PPC','MST']:
        
#         num_sess = len(dat[ba].keys())
#         ba_result[ba] = np.zeros((2,num_sess))
#         cc = 0
#         for session in dat[ba].keys():
#             ba_result[ba][0,cc] = dat[ba][session]['LD %s'%use_test]
#             ba_result[ba][1,cc] = dat[ba][session]['HD %s'%use_test]
            
#             cc+=1
    
#     for ba in ['PFC','PPC','MST']:
#         ba_result[ba] = ba_result[ba][:, (ba_result[ba][1,:] > 0) |
#                                         (ba_result[ba][0,:] > 0)]
#     color = {'MST':'g','PPC':'b','PFC':'r'}
#     plt.title('%s'%var)
#     kk=0
#     for ba in ['MST','PPC','PFC']:
#         plt.boxplot(
#                  ba_result[ba][1]-ba_result[ba][0],positions=[kk])
#         kk+=1
#     plt.xticks([0,1,2],['MST','PPC','PFC'])
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('r2_HD_vs_LD_%s.png'%use_test)