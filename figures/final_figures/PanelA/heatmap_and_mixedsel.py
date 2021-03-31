#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:37:02 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.io import loadmat
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from statsmodels.distributions import ECDF
import pandas as pd

sparse = loadmat('/Users/edoardo/Downloads/sparse.mat')

withe =  np.array([[1.,1.,1.,1.]])
green = np.array([[0.        , 0.69019608, 0.31372549,1.]])
blue = np.array([[40./256        , 20./256, 205./256,1.]])
red = np.array([[255./256        , 0./256, 0./256,1.]])


clist =  np.vstack((withe,green))
cmapMST = ListedColormap(clist)


clist =  np.vstack((withe,blue))
cmapPPC = ListedColormap(clist)

clist =  np.vstack((withe,red))
cmapPFC = ListedColormap(clist)


mst = np.array(sparse['MST_cell'],dtype=float)
ppc = np.array(sparse['PPC_cell'],dtype=float)
pfc = np.array(sparse['PFC_cell'],dtype=float)



plt.figure(figsize=(5,5))
ax1 = plt.subplot(131)
sbn.heatmap(mst[:200,:],cmap=cmapMST, ax=ax1,cbar=False)
plt.xticks([])
plt.yticks([])

ax2 = plt.subplot(132)
sbn.heatmap(ppc[:200,:],cmap=cmapPPC, ax=ax2,cbar=False)
plt.xticks([])
plt.yticks([])

ax2 = plt.subplot(133)
sbn.heatmap(pfc[:200,:],cmap=cmapPFC, ax=ax2,cbar=False)
plt.xticks([])
plt.yticks([])


plt.figure()
ax = plt.subplot(111)
xx = np.linspace(0,17,100)
cdfmst = ECDF(mst.sum(axis=1))
cdfppc = ECDF(ppc.sum(axis=1))
cdfpfc = ECDF(pfc.sum(axis=1))
plt.plot(xx,cdfmst(xx),color=green[0][:3],lw=2)
plt.plot(xx,cdfpfc(xx),color=red[0][:3],lw=2)
plt.plot(xx,cdfppc(xx),color=blue[0][:3],lw=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('mixed selectivity')
plt.ylabel('cdf')
plt.xticks([0,5,10,15],fontsize=10)
plt.yticks([0,1],fontsize=10)
plt.xlabel('num tuned variables',fontsize=10)


sensorimotor = np.arange(0,7,dtype=int)
internal = np.arange(7,11,dtype=int)
lfp = np.arange(11,14,dtype=int)
other = np.arange(14,17,dtype=int)

sm_mst = mst[:,sensorimotor].sum(axis=1)
sm_ppc = ppc[:,sensorimotor].sum(axis=1)
sm_pfc = pfc[:,sensorimotor].sum(axis=1)


int_mst = mst[:,internal].sum(axis=1)
int_ppc = ppc[:,internal].sum(axis=1)
int_pfc = pfc[:,internal].sum(axis=1)


lfp_mst = mst[:,lfp].sum(axis=1)
lfp_ppc = ppc[:,lfp].sum(axis=1)
lfp_pfc = pfc[:,lfp].sum(axis=1)

other_mst = mst[:,other].sum(axis=1)
other_ppc = ppc[:,other].sum(axis=1)
other_pfc = pfc[:,other].sum(axis=1)

dtype_dict = {'names':('brain area','selectivity','group'),'formats':('U30',int,'U30')}
table = np.zeros(4*(sm_mst.shape[0]+sm_ppc.shape[0]+sm_pfc.shape[0]),dtype_dict)

cc = 0
table['selectivity'][cc:cc+sm_mst.shape[0]] = sm_mst
table['brain area'][cc:cc+sm_mst.shape[0]] = 'MSTd'
table['group'][cc:cc+sm_mst.shape[0]] = 'sensorimotor'

cc += sm_mst.shape[0]

table['selectivity'][cc:cc+int_mst.shape[0]] = int_mst
table['brain area'][cc:cc+int_mst.shape[0]] = 'MSTd'
table['group'][cc:cc+int_mst.shape[0]] = 'internal'
cc += int_mst.shape[0]


table['selectivity'][cc:cc+lfp_mst.shape[0]] = lfp_mst
table['brain area'][cc:cc+lfp_mst.shape[0]] = 'MSTd'
table['group'][cc:cc+lfp_mst.shape[0]] = 'LFP'
cc += lfp_mst.shape[0]


table['selectivity'][cc:cc+other_mst.shape[0]] = other_mst
table['brain area'][cc:cc+other_mst.shape[0]] = 'MSTd'
table['group'][cc:cc+other_mst.shape[0]] = 'other'
cc += other_mst.shape[0]




table['selectivity'][cc:cc+sm_ppc.shape[0]] = sm_ppc
table['brain area'][cc:cc+sm_ppc.shape[0]] = '7a'
table['group'][cc:cc+sm_ppc.shape[0]] = 'sensorimotor'

cc += sm_ppc.shape[0]

table['selectivity'][cc:cc+int_ppc.shape[0]] = int_ppc
table['brain area'][cc:cc+int_ppc.shape[0]] = '7a'
table['group'][cc:cc+int_ppc.shape[0]] = 'internal'
cc += int_ppc.shape[0]


table['selectivity'][cc:cc+lfp_ppc.shape[0]] = lfp_ppc
table['brain area'][cc:cc+lfp_ppc.shape[0]] = '7a'
table['group'][cc:cc+lfp_ppc.shape[0]] = 'LFP'
cc += lfp_ppc.shape[0]


table['selectivity'][cc:cc+other_ppc.shape[0]] = other_ppc
table['brain area'][cc:cc+other_ppc.shape[0]] = '7a'
table['group'][cc:cc+other_ppc.shape[0]] = 'other'
cc += other_ppc.shape[0]




table['selectivity'][cc:cc+sm_pfc.shape[0]] = sm_pfc
table['brain area'][cc:cc+sm_pfc.shape[0]] = 'dlPFC'
table['group'][cc:cc+sm_pfc.shape[0]] = 'sensorimotor'

cc += sm_pfc.shape[0]

table['selectivity'][cc:cc+int_pfc.shape[0]] = int_pfc
table['brain area'][cc:cc+int_pfc.shape[0]] = 'dlPFC'
table['group'][cc:cc+int_pfc.shape[0]] = 'internal'
cc += int_pfc.shape[0]


table['selectivity'][cc:cc+lfp_pfc.shape[0]] = lfp_pfc
table['brain area'][cc:cc+lfp_pfc.shape[0]] = 'dlPFC'
table['group'][cc:cc+lfp_pfc.shape[0]] = 'LFP'
cc += lfp_pfc.shape[0]


table['selectivity'][cc:cc+other_pfc.shape[0]] = other_pfc
table['brain area'][cc:cc+other_pfc.shape[0]] = 'dlPFC'
table['group'][cc:cc+other_pfc.shape[0]] = 'other'
cc += other_pfc.shape[0]


plt.close('all')
for gr in np.unique(table['group']):
    plt.figure()
    plt.title(gr)
    ax = plt.subplot(111)
    df = pd.DataFrame(table[table['group']==gr])
    
    g=sbn.histplot(x="selectivity", hue="brain area",hue_order=['MSTd','7a','dlPFC'],
                palette={"MSTd":green[0][:3],'dlPFC':red[0][:3],'7a':blue[0][:3]},
                data=df,stat='probability',common_norm=False,
                cumulative=False,fill=True,multiple="dodge",shrink=1,ax=ax,
                legend=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.savefig('%s_distr.pdf'%gr,transparent=True)
    
# plt.savefig('heatmap.pdf',transparent=True)