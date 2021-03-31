#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:23:07 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy
import seaborn as sbn
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata
import matplotlib.cm as cm
print('ciao')
def compute_xy_targ_rel_monkey(trajectories,init_conditions,spikes,info,idx0=1,idx1=50,
                               cond='all',value=True):
    
    tridx = np.where(info[cond]==value)[0]
    vect_res = np.zeros(tridx.shape[0],dtype=object)
    
    spikecut = np.zeros((tridx.shape[0],spikes.shape[0],idx1-idx0))
    
    cnt_tr = 0
    skip=False
    for tr in tridx:
        print(tr)
        for j in range(spikes.shape[0]):
            try:
                spikecut[cnt_tr,j,:] = spikes[j,tr][idx0:idx1]
            except:
                print(tr)
                spikecut[cnt_tr,j,:] = np.nan
                skip=True
                continue
        if skip:
            skip=False
            continue
        
        
        x = trajectories[tr]['x_monk'][idx0:idx1]
        y = trajectories[tr]['y_monk'][idx0:idx1]
        
        # try:
        #     first = np.where((~np.isnan(x)) & (~np.isnan(y)))[0][0]
        # except:
        #     continue
        # x0 = x[first]
        # y0 = y[first]
        
        # xend = x[-1]
        # yend = y[-1]
        
        x_fly = init_conditions[tr]['x_fly']
        y_fly = init_conditions[tr]['y_fly']
        vect_res[cnt_tr] = np.zeros(x.shape[0], dtype={
        'names':('x_rel', 'y_rel'),'formats':(float,float)})
        vect_res[cnt_tr]['x_rel'] = x_fly - x
        vect_res[cnt_tr]['y_rel'] = y_fly - y
        cnt_tr += 1
        
    return vect_res,spikecut


def align_spikes_and_traj(trajectories,spikes,init_conditions,info,cond,value):

    tridx = np.where(info[cond] == value)[0]

    dtype_dict = {'names':('neuron','spike_count','x_rel','y_rel'),
                  'formats':(int,float,float,float)}

    size = 0
    for tr in tridx:
        assert(spikes[0,tr].shape[0] == trajectories[tr].shape[0])
        size += spikes[0,tr].shape[0]


    dict_trajectory = {}

    for neu in range(spikes.shape[0]):
        table = np.zeros(size, dtype=dtype_dict)
        table['neuron'] = neu
        cc = 0
        for tr in tridx:

            x = trajectories[tr]['x_monk']
            y = trajectories[tr]['y_monk']
            x_fly = init_conditions[tr]['x_fly']
            y_fly = init_conditions[tr]['y_fly']
            table['x_rel'][cc:cc+x.shape[0]] = x_fly - x
            table['y_rel'][cc:cc + x.shape[0]] = y_fly - y
            table['spike_count'][cc:cc + x.shape[0]] = spikes[neu,tr]
            cc += x.shape[0]
        dict_trajectory[neu] = deepcopy(table)

    return dict_trajectory

def stack_spikes(spikes,xy_rel):
    reshaped_spk = np.zeros((spikes.shape[1], spikes.shape[0]*spikes.shape[2]))
    xy_reshape = np.zeros((spikes.shape[0]*spikes.shape[2]),
                          dtype={'names':('x_rel','y_rel'),
                                 'formats':(float,float)})
    
    cc = 0
    for tr in range(spikes.shape[0]):
        reshaped_spk[:,cc:cc+spikes.shape[2]] = spikes[tr]
        xy_reshape[cc:cc+spikes.shape[2]] = xy_rel[tr]
        cc += spikes.shape[2]
        
    
    return reshaped_spk,xy_reshape

def firing(xe, ye, xy_rel,spikes):
    firing = np.zeros((xe.shape[0]-1, ye.shape[0]-1) + (spikes.shape[0],), dtype=float)
    
    for i in range(xe.shape[0]-1):
        for j in range(ye.shape[0]-1):
            selx = (xy_rel['x_rel'] >= xe[i]) & (xy_rel['x_rel'] < xe[i+1])
            sely = (xy_rel['y_rel'] >= ye[j]) & (xy_rel['y_rel'] < ye[j+1])
            firing[i,j,:] = (spikes[:,sely*selx]).sum(axis=1) / (0.006*(sely*selx).sum())
            
    return firing


def firing_vec(xe, ye, xy_rel, spikes):
    firing = np.zeros((xe.shape[0] - 1, ye.shape[0] - 1) , dtype=float)

    for i in range(xe.shape[0] - 1):
        for j in range(ye.shape[0] - 1):
            selx = (xy_rel['x_rel'] >= xe[i]) & (xy_rel['x_rel'] < xe[i + 1])
            sely = (xy_rel['y_rel'] >= ye[j]) & (xy_rel['y_rel'] < ye[j + 1])
            firing[i, j] = (spikes[sely * selx]).sum() / (0.006 * (sely * selx).sum())

    return firing

traj_and_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/traj_and_info.npz',allow_pickle=True)
# trajectory are from target on
trajectories = traj_and_info['trajectories']
info_all = traj_and_info['info_all']
init_cond = traj_and_info['init_cond']


session = 'm53s113'

spikes = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/spikes/spike_trajectory_%s.npy'%session,allow_pickle=True)
npz_dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'+session+'.npz',allow_pickle=True)
unit_info = npz_dat['unit_info'].all()

sele = info_all['session']==session
trajectories = trajectories[sele]
info_all = info_all[sele]
init_cond = init_cond[sele]

nan_mask = np.load('nan_mask.npy')

result = align_spikes_and_traj(trajectories,spikes,init_cond,info_all,'reward',1)
neu = 10
H,xe,ye = np.histogram2d(result[0]['x_rel'], result[0]['y_rel'],range=[[-100,100],[0,250]],bins=10)


# top neuron = 71 session m53s183

hmp = sbn.heatmap(H[::-1,::-1].T*0.006,cbar_kws={'label': 'time [sec]'})
keys = list(result.keys())
fr_dict = {}
cc  = 1

YY = np.zeros(H.shape,dtype=int)
for k in range(YY.shape[0]):
    YY[:,k]=k
XX = YY.T

for k in range(len(keys)):
    fr_dict[k] = firing_vec(xe, ye, result[keys[k]], result[keys[k]]['spike_count'])
    # fr_dict[k][H<200] = np.nan
    # works for dim = 10

def slow_convolve_nan(fr,X,Y,radius = 1.5):
    smFR = np.zeros(fr.shape)
    for ii in range(X.shape[0]):
        for jj in range(X.shape[1]):
            i = X[ii,jj]
            j = Y[ii,jj]
            keep = np.sqrt((X-i)**2 + (Y-j)**2) < radius
            smFR[ii,jj] = np.nanmean(fr[keep])
    smFR[np.isnan(fr)] = np.nan
    return smFR
fig = None
cmap_dict = {'PPC':cm.Blues_r,'PFC':cm.Reds_r,'MST':cm.Greens_r}
for k in range(len(keys)):
    ba = unit_info['brain_area'][k]
   
    fr = deepcopy(fr_dict[k])
    if k % 25 == 0:
        if not fig is None:
            plt.savefig('heatmap_%d_%s.pdf'%(k,session
                                 ))
        fig = plt.figure(figsize=(12,10))
        cc = 1
    
    # fll nan
    fr[H<400] = np.nan
    nonNan = ~np.isnan(fr)
    nonNan =nonNan.flatten()
    points = np.zeros((np.prod(XX.shape),2))
    points[:,0]=XX.flatten()
    points[:,1]=YY.flatten()
    grid_z0 = griddata(points[nonNan], fr.flatten()[nonNan], (XX, YY), method='cubic')
    
    smFR = slow_convolve_nan(grid_z0,XX,YY)
    nonNan = ~np.isnan(smFR)
    nonNan =nonNan.flatten()
    X,Y = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))
    grid_z0 = griddata(points[nonNan], smFR.flatten()[nonNan], (X, Y), method='cubic')

    
    ax = fig.add_subplot(5,5,cc)
    plt.title('%d'%k)

    grid_z0[nan_mask]=np.nan
    sbn.heatmap(grid_z0[::-1,::-1],cbar=True,ax=ax,vmin=0,cmap=cmap_dict[ba])
    plt.xticks([0.5,50.5,99.5],[-100,0,100],rotation=0,fontsize=10)
    plt.yticks([0.5,50.5,99.5],[200,125,0],rotation=0,fontsize=10)
    # xaxis
    # plt.imshow(PLT, interpolation='spline16',
    #     extent=[ xe[0], xe[-1],ye[0], ye[-1]])
    cc+=1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('heatmap_%d_%s.pdf'%(k,session
                                 ))
# for neu in result.keys():



# xy_rel, spikecut = compute_xy_targ_rel_monkey(trajectories,init_cond,spikes,info_all)
# keep = ~np.isnan(spikecut[:,0].sum(axis=1))
# xy_rel = xy_rel[keep]
# spikecut = spikecut[keep]
#
# reshaped_spk,xy_reshape = stack_spikes(spikecut,xy_rel)
#
#
# H,xe,ye = np.histogram2d(xy_reshape['x_rel'], xy_reshape['y_rel'],range=[[-225,225],[50,395]],bins=15)
# fr = firing(xe, ye, xy_reshape,reshaped_spk)
#
#
# for k in range(fr.shape[2]):
#     fr[H<50,k] = np.nan
#
# X, Y = np.meshgrid(xe, ye)
#
# # fig = plt.figure(figsize=(12,10))
# # ax = fig.add_subplot(1,1,1)
# # plt.imshow(fr[::-1,::-1,48].T, interpolation='spline16', extent=[ xe[0], xe[-1],ye[0], ye[-1]])
#
# cc  = 1
# for k in range(fr.shape[2]):
#
# # ax = fig.add_subplot(121, title='histogram counts')
# # ax.pcolormesh(X, Y, H)
#     if k % 25 == 0:
#         fig = plt.figure(figsize=(12,10))
#         cc = 1
#
#     ax = fig.add_subplot(5,5,cc)
#     plt.title('%d'%k)
#     # ax.pcolormesh(Y, X, fr[:,:,k])
#     plt.imshow(fr[::-1,::-1,k].T, interpolation='spline16',
#
#         extent=[ xe[0], xe[-1],ye[0], ye[-1]])
#     cc+=1
# plt.tight_layout()
# plt.figure()
# plt.title('first 50ms after target onset')
# for tr in range(xy_rel.shape[0]):
#     plt.scatter(xy_rel[tr]['x_rel'],xy_rel[tr]['y_rel'])
#
# plt.xlabel('x-coordinate [cm]')
# plt.ylabel('y-coordinate [cm]')






