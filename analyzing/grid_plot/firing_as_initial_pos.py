#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:23:07 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt

def compute_xy(trajectories,init_conditions,spikes,info,cond='all',value=True):
    ddict = {
        'names':('x_monk', 'y_monk'),'formats':(float,float)}
    vect_res = np.zeros(0, dtype=ddict)
    spikecut = np.zeros((0,spikes.shape[0]))
    
    
    tridx = np.where(info[cond]==value)[0]
    for tr in tridx:
        
        
        
        
        
        x = trajectories[tr]['x_monk']
        y = trajectories[tr]['y_monk']
        
        if spikes[0,tr].shape[0] != x.shape[0]:
            continue
        
        vect_res_tmp = np.zeros(x.shape[0]-1,dtype=ddict)
        vect_res_tmp['x_monk'] = x[1:]
        vect_res_tmp['y_monk'] = y[1:]
        spikecut_tmp = np.zeros((x.shape[0]-1, spikes.shape[0]))
        for j in range(spikes.shape[0]):
            spikecut_tmp[:,j] = spikes[j,tr][1:]
            
        spikecut = np.vstack((spikecut,spikecut_tmp))
        vect_res = np.hstack((vect_res,vect_res_tmp))
        
        
    return vect_res,spikecut

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
            selx = (xy_rel['x_monk'] >= xe[i]) & (xy_rel['x_monk'] < xe[i+1])
            sely = (xy_rel['y_monk'] >= ye[j]) & (xy_rel['y_monk'] < ye[j+1])
            firing[i,j,:] = (spikes[:,sely*selx]).sum(axis=1) / (0.006*(sely*selx).sum())
            
    return firing

traj_and_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/traj_and_info.npz',allow_pickle=True)

# trajectory are from target on
trajectories = traj_and_info['trajectories']
info_all = traj_and_info['info_all']
init_cond = traj_and_info['init_cond']


session  = 'm53s113'

spikes = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/trajectory/spikes/spike_trajectory_%s.npy'%session,allow_pickle=True)

sele = info_all['session']==session
trajectories = trajectories[sele]
info_all = info_all[sele]
init_cond = init_cond[sele]

xy_rel, spikecut = compute_xy(trajectories,init_cond,spikes,info_all)
# reshaped_spk,xy_reshape = stack_spikes(spikecut,xy_rel)


H,xe,ye = np.histogram2d(xy_rel['x_monk'], xy_rel['y_monk'],range=[[-150,150],[0,320]],bins=20)
fr = firing(xe, ye, xy_rel,spikecut.T)




X, Y = np.meshgrid(xe, ye)
fig = plt.figure()
ax = fig.add_subplot(111, title='histogram counts')
cf = ax.pcolormesh(X, Y, H)
fig.colorbar(cf, ax=ax)


cc  = 1
for k in range(fr.shape[2]):
    

    if k % 25 == 0:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if k !=0:
            plt.savefig('firing_rate_rel_position_%s_%d.png'%(session,k))

        fig = plt.figure(figsize=(12,10))
        plt.suptitle('rate map relative to initial position')
        cc = 1
        
    ax = fig.add_subplot(5,5,cc)
    plt.title('unit %d'%(k+1))
    # ax.pcolormesh(Y, X, fr[:,:,k])
    plt.imshow(fr[::-1,::-1,k].T, interpolation='spline16',

        extent=[ xe[0], xe[-1],ye[0], ye[-1]])
    plt.colorbar()
    
    cc+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.savefig('firing_rate_rel_position_%s_%d.png'%(session,k))


# plt.figure()
# plt.title('first 50ms after target onset')
# for tr in range(xy_rel.shape[0]):
#     plt.scatter(xy_rel[tr]['x_rel'],xy_rel[tr]['y_rel'])

# plt.xlabel('x-coordinate [cm]')
# plt.ylabel('y-coordinate [cm]')






