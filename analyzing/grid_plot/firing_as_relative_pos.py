#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:23:07 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt

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
    for tr in range(tridx):
        assert(spikes[0,tr].shape[0] == trajectories[tr].shape[0])
        size += spikes[0,tr].shape[0]


    dict_trajectory = {}

    for neu in range(spikes.shape[0]):
        table = np.zeros(size, dtype=dtype_dict)
        cc = 0
        for tr in range(tridx):

            x = trajectories[tr]['x_monk']
            y = trajectories[tr]['y_monk']
            x_fly = init_conditions[tr]['x_fly']
            y_fly = init_conditions[tr]['y_fly']
            table['x_rel'][cc:cc+x.shape[0]] = x_fly - x
            table['y_rel'][cc:cc + x.shape[0]] = y_fly - y
            table['spike_count'] = spikes[tr]
            cc += x.shape[0]
        dict_trajectory[neu] = table

    return table

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

traj_and_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/traj_and_info.npz',allow_pickle=True)

# trajectory are from target on
trajectories = traj_and_info['trajectories']
info_all = traj_and_info['info_all']
init_cond = traj_and_info['init_cond']


session = 'm44s183'

spikes = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/spikes/spike_trajectory_%s.npy'%session,allow_pickle=True)

sele = info_all['session']==session
trajectories = trajectories[sele]
info_all = info_all[sele]
init_cond = init_cond[sele]

xy_rel, spikecut = compute_xy_targ_rel_monkey(trajectories,init_cond,spikes,info_all)
keep = ~np.isnan(spikecut[:,0].sum(axis=1))
xy_rel = xy_rel[keep]
spikecut = spikecut[keep]

reshaped_spk,xy_reshape = stack_spikes(spikecut,xy_rel)


H,xe,ye = np.histogram2d(xy_reshape['x_rel'], xy_reshape['y_rel'],range=[[-225,225],[50,395]],bins=15)
fr = firing(xe, ye, xy_reshape,reshaped_spk)


for k in range(fr.shape[2]):
    fr[H<50,k] = np.nan

X, Y = np.meshgrid(xe, ye)

# fig = plt.figure(figsize=(12,10))
# ax = fig.add_subplot(1,1,1)
# plt.imshow(fr[::-1,::-1,48].T, interpolation='spline16', extent=[ xe[0], xe[-1],ye[0], ye[-1]])

cc  = 1
for k in range(fr.shape[2]):

# ax = fig.add_subplot(121, title='histogram counts')
# ax.pcolormesh(X, Y, H)
    if k % 25 == 0:
        fig = plt.figure(figsize=(12,10))
        cc = 1

    ax = fig.add_subplot(5,5,cc)
    plt.title('%d'%k)
    # ax.pcolormesh(Y, X, fr[:,:,k])
    plt.imshow(fr[::-1,::-1,k].T, interpolation='spline16',

        extent=[ xe[0], xe[-1],ye[0], ye[-1]])
    cc+=1
plt.tight_layout()
plt.figure()
plt.title('first 50ms after target onset')
for tr in range(xy_rel.shape[0]):
    plt.scatter(xy_rel[tr]['x_rel'],xy_rel[tr]['y_rel'])

plt.xlabel('x-coordinate [cm]')
plt.ylabel('y-coordinate [cm]')






