#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:12:54 2020

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(thisPath)),'GAM_Library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import matplotlib.pylab as plt

plt.close('all')
dat = np.load('input_hist.npz')
hist_matrix = dat['hist']
edge_matrix = dat['edge']
info = dat['info']


for var in np.unique(info['variable']):
    keep  = info['variable'] == var
    hist_matrix[keep,:]
    cs = np.cumsum(hist_matrix[keep,:],axis=1)
    cs = (cs.T/np.max(cs,axis=1)).T
    
    plt.figure()
    plt.title(var)
    plt.plot(edge_matrix[keep,:-1].T,cs.T)
    plt.ylabel('cdf')
    plt.xlabel('input domain')
    plt.savefig('input_cdf_%s.png'%var)
    
    
for var in np.unique(info['variable']):
    keep  = info['variable'] == var
    edge = edge_matrix[keep,:]
    
    min_max = np.zeros((edge.shape[0],2))
    min_max[:, 0] = edge[:,0]
    min_max[:, 1] = edge[:,-1]
    srt_idx = np.argsort(min_max[:,1])
    min_max = min_max[srt_idx, :]
    plt.figure()
    plt.title(var)
    for kk in range(edge.shape[0]):
        plt.hlines(y=kk, 
               xmin=min_max[kk,0], 
               xmax=min_max[kk,1])
    plt.savefig('input_domain_%s.png'%var)
    
    

session_excluded = []

sele = info['variable'] == 'ang_target'
sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]


idx_edge_0 = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_0[k] = np.where(sele_edges[k,:] < 0)[0][-1]
    except:
         session_excluded = np.hstack((session_excluded,[sele_info['session'][k]]))
         
         
session_excluded = np.hstack((session_excluded, sele_info['session'][idx_edge_0 == 0]))
session_excluded = np.hstack((session_excluded, sele_info['session'][idx_edge_0 == 400]))

session_excluded = np.hstack((session_excluded, sele_info['session'][sele_edges[:,0] > 20]))
session_excluded = np.hstack((session_excluded, sele_info['session'][sele_edges[:,-1] < 25]))

frac_zeros = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_0[kk] == 400:
        continue
    frac_zeros[kk] = sele_hist[kk,idx_edge_0[kk]]/sele_hist[kk].sum()
session_excluded = np.hstack((session_excluded, sele_info['session'][frac_zeros>0.1]))
session_excluded = np.unique(session_excluded)


sele = info['variable'] == 'rad_target'
sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

session_excluded = np.hstack((session_excluded, sele_info['session'][sele_edges[:,0] > 10]))
session_excluded = np.unique(session_excluded)


sele = info['variable'] == 'rad_path'
sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_0 = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_0[k] = np.where(sele_edges[k,:] < 0)[0][-1]
    except:
         session_excluded = np.hstack((session_excluded,[sele_info['session'][k]]))
         

frac_zeros = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_0[kk] == 400:
        continue
    frac_zeros[kk] = sele_hist[kk,idx_edge_0[kk]]/sele_hist[kk].sum()
    
session_excluded = np.hstack((session_excluded, sele_info['session'][frac_zeros > 0.6]))
session_excluded = np.unique(session_excluded)


sele = info['variable'] == 'ang_vel'
sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_0 = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_0[k] = np.where(sele_edges[k,:] < 0)[0][-1]
    except:
         session_excluded = np.hstack((session_excluded,[sele_info['session'][k]]))
         

frac_zeros = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_0[kk] == 400:
        continue
    frac_zeros[kk] = sele_hist[kk,idx_edge_0[kk]]/sele_hist[kk].sum()
    
session_excluded = np.hstack((session_excluded, sele_info['session'][frac_zeros > 0.6]))
session_excluded = np.unique(session_excluded)

sele = info['variable'] == 'rad_vel'
sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_0 = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_0[k] = np.where(sele_edges[k,:] < 0)[0][-1]
    except:
         session_excluded = np.hstack((session_excluded,[sele_info['session'][k]]))
         

frac_zeros = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_0[kk] == 400:
        continue
    frac_zeros[kk] = sele_hist[kk,idx_edge_0[kk]]/sele_hist[kk].sum()
    
session_excluded = np.hstack((session_excluded, sele_info['session'][frac_zeros > 0.6]))
session_excluded = np.unique(session_excluded)



sele = info['variable'] == 'ang_path'
sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_0 = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_0[k] = np.where(sele_edges[k,:] < 0)[0][-1]
    except:
         session_excluded = np.hstack((session_excluded,[sele_info['session'][k]]))
         

frac_zeros = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_0[kk] == 400:
        continue
    frac_zeros[kk] = sele_hist[kk,idx_edge_0[kk]]/sele_hist[kk].sum()
    
session_excluded = np.hstack((session_excluded, sele_info['session'][frac_zeros > 0.5]))
session_excluded = np.unique(session_excluded)



plt.close('all')
dat = np.load('input_hist.npz')
hist_matrix = dat['hist']
edge_matrix = dat['edge']
info = dat['info']

sele_bool = np.ones(info.shape[0],dtype=bool)
for sess in session_excluded:
    sele_bool[info['session']==sess] = False

hist_matrix = hist_matrix[sele_bool]
edge_matrix = edge_matrix[sele_bool]
info = info[sele_bool]

    

for var in np.unique(info['variable']):
    keep  = info['variable'] == var
    hist_matrix[keep,:]
    cs = np.cumsum(hist_matrix[keep,:],axis=1)
    cs = (cs.T/np.max(cs,axis=1)).T
    
    plt.figure()
    plt.title(var)
    plt.plot(edge_matrix[keep,:-1].T,cs.T)
    plt.ylabel('cdf')
    plt.xlabel('input domain')
    plt.savefig('input_cdf_%s.png'%var)
    
    
for var in np.unique(info['variable']):
    keep  = info['variable'] == var
    edge = edge_matrix[keep,:]
    
    min_max = np.zeros((edge.shape[0],2))
    min_max[:, 0] = edge[:,0]
    min_max[:, 1] = edge[:,-1]
    srt_idx = np.argsort(min_max[:,1])
    min_max = min_max[srt_idx, :]
    plt.figure()
    plt.title(var)
    for kk in range(edge.shape[0]):
        plt.hlines(y=kk, 
               xmin=min_max[kk,0], 
               xmax=min_max[kk,1])
    plt.savefig('input_domain_%s.png'%var)
    

plt.close('all')
## PLOT the eye position
kk = 0
for session in np.unique(info['session']):
    if kk % 20 == 0:
        if kk != 0:
            plt.tight_layout()
            plt.savefig('vert_eye_pos_%d.png'%(kk//4))
        plt.figure(figsize=(10,8))
        sub_num = 1
    
    plt.subplot(4,5,sub_num)
    sele = (info['variable'] == 'eye_vert') * (info['session'] == session)
    sele_edges = edge_matrix[sele]
    sele_hist = hist_matrix[sele]
    plt.bar(sele_edges[0,:-1],sele_hist[0,:],width=sele_edges[0,1]-sele_edges[0,0])
    plt.title(session)
    kk += 1
    sub_num += 1
    plt.yticks([])
    
plt.tight_layout()
plt.savefig('vert_eye_pos_%d.png'%(kk//4))

## PLOT the eye position
kk = 0
for session in np.unique(info['session']):
    if kk % 20 == 0:
        if kk != 0:
            plt.tight_layout()
            plt.savefig('hori_eye_pos_%d.png'%(kk//4))
        plt.figure(figsize=(10,8))
        sub_num = 1
    
    plt.subplot(4,5,sub_num)
    sele = (info['variable'] == 'eye_hori') * (info['session'] == session)
    sele_edges = edge_matrix[sele]
    sele_hist = hist_matrix[sele]
    plt.bar(sele_edges[0,:-1],sele_hist[0,:],width=sele_edges[0,1]-sele_edges[0,0])
    # plt.xlim()
    plt.title(session)
    kk += 1
    sub_num += 1
    plt.yticks([])
    
plt.tight_layout()
plt.savefig('hori_eye_pos_%d.png'%(kk//4))


# check domaiin rad path
var = 'rad_path'
sele = info['variable'] == var

sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_neg = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_neg[k] = np.where(sele_edges[k,:] < 0)[0][-1] 
       
    except:
        pass
frac_neg = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_neg[kk] == -1:
        continue
    frac_neg[kk] = sele_hist[kk,:idx_edge_neg[kk]].sum()/sele_hist[kk].sum()
    

idx_edge_pos = np.ones(sele.sum(),dtype=int)*400
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_pos[k] = np.where(sele_edges[k,:] > 350)[0][0] 
       
    except:
        pass
frac_pos = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    
    frac_pos[kk] = sele_hist[kk,idx_edge_pos[kk]:].sum()/sele_hist[kk].sum()
    
print('rad path outside (0,350):',max(frac_neg)+max(frac_pos))


# check domaiin rad path
var = 'rad_target'
sele = info['variable'] == var

sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_neg = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_neg[k] = np.where(sele_edges[k,:] < 0)[0][-1] 
       
    except:
        pass
frac_neg = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_neg[kk] == -1:
        continue
    frac_neg[kk] = sele_hist[kk,:idx_edge_neg[kk]].sum()/sele_hist[kk].sum()
    

idx_edge_pos = np.ones(sele.sum(),dtype=int)*400
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_pos[k] = np.where(sele_edges[k,:] > 400)[0][0] 
       
    except:
        pass
frac_pos = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    
    frac_pos[kk] = sele_hist[kk,idx_edge_pos[kk]:].sum()/sele_hist[kk].sum()
    
print('rad target outside (0,400):',max(frac_neg)+max(frac_pos))


# check domaiin rad path
var = 'rad_vel'
sele = info['variable'] == var

sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_neg = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_neg[k] = np.where(sele_edges[k,:] < 0)[0][-1] 
       
    except:
        pass
frac_neg = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_neg[kk] == -1:
        continue
    frac_neg[kk] = sele_hist[kk,:idx_edge_neg[kk]].sum()/sele_hist[kk].sum()
    

idx_edge_pos = np.ones(sele.sum(),dtype=int)*400
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_pos[k] = np.where(sele_edges[k,:] > 200)[0][0] 
       
    except:
        pass
frac_pos = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    
    frac_pos[kk] = sele_hist[kk,idx_edge_pos[kk]:].sum()/sele_hist[kk].sum()
    
print('rad vel outside (0,200):',max(frac_neg)+max(frac_pos))

# check domaiin rad path
var = 'ang_path'
sele = info['variable'] == var



sele_edges = edge_matrix[sele]
sele_hist = hist_matrix[sele]
sele_info = info[sele]

idx_edge_neg = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_neg[k] = np.where(sele_edges[k,:] < -65)[0][-1] 
       
    except:
        pass
frac_neg = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_neg[kk] == -1:
        continue
    frac_neg[kk] = sele_hist[kk,:idx_edge_neg[kk]].sum()/sele_hist[kk].sum()
    

idx_edge_pos = np.ones(sele.sum(),dtype=int)*400
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_pos[k] = np.where(sele_edges[k,:] > 65)[0][0] 
       
    except:
        pass
frac_pos = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    
    frac_pos[kk] = sele_hist[kk,idx_edge_pos[kk]:].sum()/sele_hist[kk].sum()
    
print('ang path outside (-50,50):',max(frac_neg)+max(frac_pos))


# check domaiin ang vel
var = 'ang_vel'
sele = info['variable'] == var



sele_edges = edge_matrix[sele]
sele_hist = np.array(hist_matrix[sele],dtype=float)
sele_info = info[sele]
# set nan to central
for k in range(sele_hist.shape[0]):
    sele_hist[k,np.argmax(sele_hist[k])] = np.nan

idx_edge_neg = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_neg[k] = np.where(sele_edges[k,:] < -50)[0][-1] 
       
    except:
        pass
frac_neg = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_neg[kk] == -1:
        continue
    frac_neg[kk] = np.nansum(sele_hist[kk,:idx_edge_neg[kk]])/np.nansum(sele_hist[kk])
    

idx_edge_pos = np.ones(sele.sum(),dtype=int)*400
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_pos[k] = np.where(sele_edges[k,:] > 50)[0][0] 
       
    except:
        pass
frac_pos = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    
    frac_pos[kk] = np.nansum(sele_hist[kk,idx_edge_pos[kk]:])/np.nansum(sele_hist[kk])
    
print('ang path outside (-50,50):',max(frac_neg)+max(frac_pos))

# check domaiin ang vel
var = 'ang_target'
sele = info['variable'] == var



sele_edges = edge_matrix[sele]
sele_hist = np.array(hist_matrix[sele],dtype=float)
sele_info = info[sele]
# set nan to central
for k in range(sele_hist.shape[0]):
    sele_hist[k,np.argmax(sele_hist[k])] = np.nan

idx_edge_neg = np.zeros(sele.sum(),dtype=int)
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_neg[k] = np.where(sele_edges[k,:] < -50)[0][-1] 
       
    except:
        pass
frac_neg = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    if idx_edge_neg[kk] == -1:
        continue
    frac_neg[kk] = np.nansum(sele_hist[kk,:idx_edge_neg[kk]])/np.nansum(sele_hist[kk])
    

idx_edge_pos = np.ones(sele.sum(),dtype=int)*400
for k in range(idx_edge_0.shape[0]):
    try:
        idx_edge_pos[k] = np.where(sele_edges[k,:] > 50)[0][0] 
       
    except:
        pass
frac_pos = np.zeros(sele_hist.shape[0])
for kk in range(sele_hist.shape[0]):
    
    frac_pos[kk] = np.nansum(sele_hist[kk,idx_edge_pos[kk]:])/np.nansum(sele_hist[kk])
    
print('ang path outside (-50,50):',max(frac_neg)+max(frac_pos))

exclude_eye_position = ['m44s213','m53s133','m53s134','m53s105','m53s182']



