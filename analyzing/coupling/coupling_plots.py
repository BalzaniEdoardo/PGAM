#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:30:42 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
import pandas as pd
coupl_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy')

# sel monkey
coupl_info = coupl_info[(coupl_info['pseudo-r2']>=0.02)]
coupl_info = coupl_info[(coupl_info['monkey']=='Schro')]


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

# table_stat = np.zeros(len(np.unique(mutual_info['variable'])),
#                       dtype={'names':('variable','MST num','PPC num','PFC num',
#                                       'MST freq sign','PPC freq sign','PFC freq sign','Chi2-stat',
#                                       'p-val','Cramer-V','effect-size'),
#                       'formats':('U30',int,int,int,
#                                       float,float,float,float,
#                                       float,float,'U30')})



label_coupling = []#pd.Series(np.hstack((mst_vec,ppc_vec,pfc_vec)))
bl_label = []#pd.Series(np.hstack((mst_bl,ppc_bl,pfc_bl)))


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('MST->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))

sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('MST->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('PPC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('PFC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['PFC->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PPC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PFC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PFC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


label_coupling = pd.Series(label_coupling)
bl_label = pd.Series(bl_label)

cross_tab = pd.crosstab(label_coupling,bl_label)
cramers_corrected_stat(label_coupling,bl_label)

# sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
# betw_coupl = coupl_info[sel]
# print('MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
#
#
# sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
# betw_coupl = coupl_info[sel]
# print('PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
#
# sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
# betw_coupl = coupl_info[sel]
# print('PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])



