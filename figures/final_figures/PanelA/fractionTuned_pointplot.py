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
import seaborn as sbn

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

dat = np.load('mutual_info.npz')
mutual_info = dat['mutual_info']
tuning = dat['tuning_Hz']

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
    
    dprime_vec[cc] = np.max(np.abs(tun['y_raw'] - tun['y_model']))/(np.mean(tun['y_raw']))#/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))

    cc += 1

# remove crazy outliers and distribution tails
filt = dprime_vec < np.nanpercentile(dprime_vec,98)
tuning = tuning[filt]
mutual_info = mutual_info[filt]
dprime_vec = dprime_vec[filt]
    
# list variables

var_order = ['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF','rad_target','ang_target',
 'rad_path','ang_path','lfp_beta','lfp_alpha','lfp_theta','t_reward','eye_vert','eye_hori']

group_var = {
    'sensorimotor':['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF'],
    'internal':['rad_target','ang_target','rad_path','ang_path'],
    'LFP':['lfp_beta','lfp_alpha','lfp_theta'],
    'other':['t_reward','eye_vert','eye_hori'],
    }


df = pd.DataFrame(mutual_info)

plt.figure(figsize=(12,4))
ax = plt.subplot(111)
sbn.pointplot(x='variable',y='significance',hue='brain_area',
              hue_order=['MST','PPC','PFC'],palette={'MST':'g','PFC':'r','PPC':'b'},order=var_order,linestyles='none',dodge=0.2,data=df,ax=ax,
              legend=False)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend([],[], frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('frac. tuned')


# summary statistics
table_stat = np.zeros(len(np.unique(mutual_info['variable'])),
                      dtype={'names':('variable','MST num','PPC num','PFC num',
                                      'MST freq sign','PPC freq sign','PFC freq sign','Chi2-stat',
                                      'p-val','Cramer-V','effect-size'),
                      'formats':('U30',int,int,int,
                                      float,float,float,float,
                                      float,float,'U30')})
cc = 0
for gr in group_var.keys():
    for var in group_var[gr]:
        sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'PFC') &\
            (mutual_info['variable'] == var)
        pfc_vec = ['PFC']*sel.sum()
        pfc_bl = mutual_info[sel]['significance']


        sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'MST') &\
    (mutual_info['variable'] == var)
        mst_vec = ['MST']*sel.sum()
        mst_bl = mutual_info[sel]['significance']


        sel = (mutual_info['manipulation_type'] == 'all') & (mutual_info['brain_area'] == 'PPC') &\
    (mutual_info['variable'] == var)
        ppc_vec = ['PPC']*sel.sum()
        ppc_bl = mutual_info[sel]['significance']

        label_area = pd.Series(np.hstack((mst_vec,ppc_vec,pfc_vec)))
        bl_label = pd.Series(np.hstack((mst_bl,ppc_bl,pfc_bl)))


        counts_ba = label_area.value_counts()
        cross_tab = pd.crosstab(label_area,bl_label)

        table_stat[cc]['variable'] = var
        table_stat[cc]['MST num'] = counts_ba.MST
        table_stat[cc]['PFC num'] = counts_ba.PFC
        table_stat[cc]['PPC num'] = counts_ba.PPC

        table_stat[cc]['MST freq sign'] = cross_tab.loc['MST'].loc[True] / cross_tab.loc['MST'].sum()
        table_stat[cc]['PFC freq sign'] = cross_tab.loc['PFC'].loc[True] / cross_tab.loc['PFC'].sum()
        table_stat[cc]['PPC freq sign'] = cross_tab.loc['PPC'].loc[True] / cross_tab.loc['PPC'].sum()

        table_stat[cc]['p-val'] = sts.chi2_contingency(cross_tab)[1]
        chi2,cramV = cramers_corrected_stat(label_area,bl_label)
        table_stat[cc]['Chi2-stat'] = chi2
        table_stat[cc]['Cramer-V'] = sts.chi2_contingency(cross_tab)[1]

        if cramV < 0.1:
            lab = 'No effect'
        elif cramV >= 0.1 and cramV < 0.3:
            lab = 'Small effect'
        if cramV >= 0.3 and cramV < 0.5:
            lab = 'Medium effect'
        if cramV >= 0.5:
            lab = 'Large effect'
        table_stat[cc]['effect-size'] = lab

        cc+=1
ftun_stat = pd.DataFrame(table_stat)
writer = pd.ExcelWriter('frac_tuned_statistics.xlsx')
ftun_stat.to_excel(writer,index=False)
writer.save()
writer.close()
#