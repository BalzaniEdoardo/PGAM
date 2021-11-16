#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:11:59 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import os,sys
import dill
# candidate units
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import pandas as pd
from matplotlib.patches import Rectangle


import scipy.stats as sts

def cramers_corrected_stat(x,y):

    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:   
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = sts.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return chi2, round(result,6)


mutual_info = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/data/mutual_info.npy')
tuning = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/data/tuning_func.npy',allow_pickle=True)

keep_sess = np.unique(mutual_info['session'][mutual_info['manipulation_type']=='density'])
filt_sess = np.zeros(mutual_info.shape,dtype=bool)
for sess in keep_sess:
    filt_sess[mutual_info['session']==sess] = True



filt = (mutual_info['manipulation_type'] == 'all') & (mutual_info['pseudo-r2'] > 0.01) &\
         (~np.isnan(mutual_info['mutual_info']))

tuning = tuning[filt]
mutual_info = mutual_info[filt]


dprime_vec = np.zeros(tuning.shape)
cc = 0
for tun in tuning:
    #
    # dprime_vec[cc] = np.abs(np.mean(tun['y_raw'] - tun['y_model']))/(np.mean(tun['y_raw']))#/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))
    
    dprime_vec[cc] = np.max(np.abs(tun['y_raw'] - tun['y_model']))/(np.mean(tun['y_raw']))#/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))

    cc += 1

# remove crazy outliers and distribution tails
filt = dprime_vec < np.nanpercentile(dprime_vec,98)
tuning = tuning[filt]
mutual_info = mutual_info[filt]
dprime_vec = dprime_vec[filt]
    
# list variables

var_order = ['rad_vel','ang_vel','rad_acc','ang_acc','rad_target','ang_target',
 'rad_path','ang_path','t_move','t_flyOFF','t_stop','t_reward','lfp_beta','lfp_alpha','lfp_theta','eye_vert','eye_hori']





# split monkey
sess_list = np.unique(mutual_info['session'])
sign_mat = np.zeros(len(sess_list)*4,dtype={'names':['monk','session','brain_area']+var_order,
                             'formats':['U30','U30','U30']+[float]*len(var_order)})


mnk_num = 0
for session in sess_list:
    sel1 = mutual_info['session'] == session
    if ('m51' in session):#or ('m72' in session)
        continue
    
    for ba in ['MST','PPC','PFC','VIP']:
        sel2 = mutual_info['brain_area'] == ba
        sign_mat['session'][mnk_num] = session
        sign_mat['brain_area'][mnk_num] = ba
        
        for var in var_order:
            
            sel3 = (mutual_info['variable'] == var) & sel1 #& sel
            # if sel3.shape[0]<10:
            
            sign_mat[mnk_num][var] = mutual_info['significance'][sel3].sum()/sel3.sum()
        mnk_num += 1

sign_mat = sign_mat[~np.isnan(sign_mat['t_move'])]
# pool monkey
sign_mat_pooled = np.zeros(4*len(np.unique(sign_mat['session'])),dtype={'names':['brain_area','session']+var_order,
                             'formats':['U30','U30']+[float]*len(var_order)})

cc = 0
for sess in np.unique(sign_mat['session']):
    for ba in ['MST','PPC','PFC','VIP']:
        sel2 = mutual_info['brain_area'] == ba
        sign_mat_pooled['brain_area'][cc] = ba
        sign_mat_pooled['session'][cc] = sess
        for var in var_order:
            
            sel3 = (mutual_info['variable'] == var) & sel2 & (mutual_info['session'] == sess) 
            
            sign_mat_pooled[cc][var] = mutual_info['significance'][sel3].sum()/sel3.sum()
        cc += 1

sign_mat_pooled = sign_mat_pooled[~np.isnan(sign_mat_pooled['t_move'])]

color_dict = {'PFC':'r','MST':'g','PPC':'b','VIP':'k'}

group_var = {
    'sensorimotor':['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF'],
    'internal':['rad_target','ang_target','rad_path','ang_path'],
    'LFP':['lfp_beta','lfp_alpha','lfp_theta'],
    'other':['t_reward','eye_vert','eye_hori'],
    }

dx_group = 7
dx_var = 5
dx_ba = 0.4


x0 = 0
x_ticks = []
x_ticks_lab = []
for gr in group_var.keys():
    x_ticks = np.hstack((x_ticks, (dx_var*np.arange(len(group_var[gr]) )+ x0)))
    x0 = x_ticks[-1] + dx_group
    x_ticks_lab += group_var[gr]

 
plt.figure(figsize=(12,4))
ax = plt.subplot(111)
vec_val = []
# marker_dict = {'Schro':'^','Bruno':'o','Quigley':'s'}
ccba = 0

triplet_x = {}
triplet_y = {}
for ba in ['MST','PPC','PFC']:
    selmat = sign_mat_pooled[(sign_mat_pooled['brain_area'] == ba)]
    if np.isnan(selmat['t_move'][0]):
        continue
    
    y_axis = []
    y_axis_err = []
    ccx = 0
    for gr in group_var.keys():
        for var in group_var[gr]:
            y_axis += [np.nanmean(selmat[var])]
            y_axis_err += [1.96*np.nanstd(selmat[var])/np.sqrt((~np.isnan(selmat[var])).sum())]
            if var in triplet_y.keys():
                triplet_x[var] += [x_ticks[ccx]+ccba*dx_ba]
                triplet_y[var] += [y_axis[-1]]
            else:
                triplet_x[var] = [x_ticks[ccx]+ccba*dx_ba]
                triplet_y[var] = [y_axis[-1]]
            ccx+=1
                
    y_axis = np.array(y_axis)
    y_axis_err = np.array(y_axis_err)
    for ii in range(len(y_axis)):
        print(ba, x_ticks_lab[ii], y_axis[ii])
    
    for k in range(len(x_ticks)):
        plt.errorbar([x_ticks[k]+ccba*dx_ba], [y_axis[k]], yerr=[y_axis_err[k]],color=color_dict[ba],lw=2)
        
        
    plt.plot(x_ticks+ccba*dx_ba, y_axis,ls='none', marker='o',ms=8,color=color_dict[ba])
    
    ccba+=1
    
    plt.yticks([0,1],[0,1], fontsize=10)
plt.xticks(x_ticks, x_ticks_lab,rotation=90)

# for var in triplet_x.keys():
#     plt.plot(triplet_x[var],triplet_y[var],'--k',lw=0.5)
xlim = plt.xlim()
# rect = Rectangle([xlim[0],0], np.diff(xlim), 0.5, color=(0.5,)*3,alpha=0.3)
# ax.add_patch(rect)
# rect1 = Rectangle([xlim[0],0.5], np.diff(xlim), 0.5, color=(0.5,)*3,alpha=0.4)
# ax.add_patch(rect1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylim(-0.,1.0)
plt.xlim(xlim)
plt.tight_layout()
plt.savefig('/Volumes/WD_Edo/firefly_analysis/LFP_band/FINALFIG/Figure2/raw_pdf/frac_tuned_SEM_AllMarco.pdf')

# plt.plot([x_ticks[0],x_ticks[-1]],[0.5,0.5])

# contingengy
var = 'rad_target'
sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'MST') &\
    (mutual_info['variable'] == var)
mst_vec = ['MST']*sel.sum()
mst_bl = mutual_info[sel]['significance']


sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'PPC') &\
    (mutual_info['variable'] == var)
ppc_vec = ['PPC']*sel.sum()
ppc_bl = mutual_info[sel]['significance']



sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'PFC') &\
    (mutual_info['variable'] == var)
pfc_vec = ['PFC']*sel.sum()
pfc_bl = mutual_info[sel]['significance']

label_area = pd.Series(np.hstack((mst_vec,ppc_vec,pfc_vec)))
bl_label = pd.Series(np.hstack((mst_bl,ppc_bl,pfc_bl)))
cross_tab = pd.crosstab(label_area,bl_label)


# table_stat = np.zeros(len(np.unique(mutual_info['variable'])),
#                       dtype={'names':('variable','MST num','PPC num','PFC num',
#                                       'MST freq sign','PPC freq sign','PFC freq sign','Chi2-stat',
#                                       'p-val','Cramer-V','effect-size'),
#                       'formats':('U30',int,int,int,
#                                       float,float,float,float,
#                                       float,float,'U30')})
# cc = 0
# for gr in group_var.keys():
#     for var in group_var[gr]:
#         sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'PFC') &\
#             (mutual_info['variable'] == var)
#         pfc_vec = ['PFC']*sel.sum()
#         pfc_bl = mutual_info[sel]['significance']
        
        
#         sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'MST') &\
#     (mutual_info['variable'] == var)
#         mst_vec = ['MST']*sel.sum()
#         mst_bl = mutual_info[sel]['significance']


#         sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'PPC') &\
#     (mutual_info['variable'] == var)
#         ppc_vec = ['PPC']*sel.sum()
#         ppc_bl = mutual_info[sel]['significance']
        
#         label_area = pd.Series(np.hstack((mst_vec,ppc_vec,pfc_vec)))
#         bl_label = pd.Series(np.hstack((mst_bl,ppc_bl,pfc_bl)))
        

#         counts_ba = label_area.value_counts()
#         cross_tab = pd.crosstab(label_area,bl_label)
        
#         table_stat[cc]['variable'] = var
#         table_stat[cc]['MST num'] = counts_ba.MST
#         table_stat[cc]['PFC num'] = counts_ba.PFC
#         table_stat[cc]['PPC num'] = counts_ba.PPC
        
#         table_stat[cc]['MST freq sign'] = cross_tab.loc['MST'].loc[True] / cross_tab.loc['MST'].sum()
#         table_stat[cc]['PFC freq sign'] = cross_tab.loc['PFC'].loc[True] / cross_tab.loc['PFC'].sum()
#         table_stat[cc]['PPC freq sign'] = cross_tab.loc['PPC'].loc[True] / cross_tab.loc['PPC'].sum()
        
#         table_stat[cc]['p-val'] = sts.chi2_contingency(cross_tab)[1]
#         chi2,cramV = cramers_corrected_stat(label_area,bl_label)
#         table_stat[cc]['Chi2-stat'] = chi2
#         table_stat[cc]['Cramer-V'] = sts.chi2_contingency(cross_tab)[1]
        
#         if cramV < 0.1:
#             lab = 'No effect'
#         elif cramV >= 0.1 and cramV < 0.3:
#             lab = 'Small effect'
#         if cramV >= 0.3 and cramV < 0.5:
#             lab = 'Medium effect'
#         if cramV >= 0.5:
#             lab = 'Large effect'
#         table_stat[cc]['effect-size'] = lab
        
#         cc+=1
# ftun_stat = pd.DataFrame(table_stat)
# writer = pd.ExcelWriter('frac_tuned_statistics.xlsx')
# ftun_stat.to_excel(writer,index=False)
# writer.save()
# writer.close()
        