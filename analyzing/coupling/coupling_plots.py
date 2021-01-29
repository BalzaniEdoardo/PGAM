#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:30:42 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt

coupl_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info_old.npy')

sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('MST->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('MST->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('PPC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('PFC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PPC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PFC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])



sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation type'] == 'all') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

