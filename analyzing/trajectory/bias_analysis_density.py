#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:35:32 2021

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
from statsmodels.distributions import ECDF

def compute_dist(trajectories,init_conditions):
    vect_res = np.zeros(trajectories.shape[0],dtype={
        'names':('dist_traveled', 'dist_target'),'formats':(float,float)})
    
    for tr in range(trajectories.shape[0]):
        x = trajectories[tr]['x_monk']
        y = trajectories[tr]['y_monk']
        try:
            first = np.where((~np.isnan(x)) & (~np.isnan(y)))[0][0]
        except:
            continue
        x0 = x[first]
        y0 = y[first]
        
        xend = x[-1]
        yend = y[-1]
        
        x_fly = init_conditions[tr]['x_fly']
        y_fly = init_conditions[tr]['y_fly']
        
        vect_res[tr]['dist_traveled'] = np.sqrt((xend - x0)**2 + (yend - y0)**2 )
        vect_res[tr]['dist_target'] = np.sqrt((x_fly - x0)**2 + (y_fly - y0)**2 )
        

        
    return vect_res

# this file contains the trajectories from target onset to stop
# and the fly position, with an associated table with the trial info
traj_and_info = np.load('traj_and_info.npz',allow_pickle=True)

trajectories = traj_and_info['trajectories']
info_trials = traj_and_info['info_all']
init_conditions = traj_and_info['init_cond']

distances = compute_dist(trajectories,init_conditions)
delta_dist = distances['dist_traveled'] - distances['dist_target']




mean_LD = []
mean_HD = []
session_list = np.unique(info_trials['session'])
# bias analysis
plt.figure(figsize=(10,4))
plt.suptitle('all trials')
kk = 1
for session in session_list:
    print(session)
    plt.subplot(2,5,kk)
    plt.title(session)
    sele_HD = (info_trials['session'] == session) & (info_trials['density'] == 0.005)
    sele_LD = (info_trials['session'] == session) & (info_trials['density'] == 0.0001)
    
    cdf_LD = ECDF(delta_dist[sele_LD])
    cdf_HD = ECDF(delta_dist[sele_HD])
    
    mean_LD += [np.nanmean(delta_dist[sele_LD])]
    mean_HD += [np.nanmean(delta_dist[sele_HD])]

    xx = np.linspace(np.nanpercentile(delta_dist[sele_LD | sele_HD], 1),np.nanpercentile(delta_dist[sele_LD | sele_HD],99),1000)
    plt.plot(xx, cdf_HD(xx), label='hd')
    plt.plot(xx, cdf_LD(xx), label='ld')
    
    plt.xlabel('cm')
    plt.ylabel('hist')
    
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot(xlim,[0.5]*2,'--k')
    plt.legend(frameon=False,fontsize=8)

    kk+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
np.savez('mean_bias_response.npz',mean_HD=mean_HD,mean_LD=mean_LD,session_list=session_list)
plt.savefig('cdf_response_bias.png')
# bias analysis
plt.figure(figsize=(10,4))
plt.suptitle('unrewarded trials')
kk = 1
for session in np.unique(info_trials['session']):
    print(session)
    plt.subplot(2,5,kk)
    plt.title(session)
    sele_HD = (info_trials['session'] == session) & (info_trials['density'] == 0.005) & (info_trials['reward']==0)
    sele_LD = (info_trials['session'] == session) & (info_trials['density'] == 0.0001) & (info_trials['reward']==0)
    
    cdf_LD = ECDF(delta_dist[sele_LD])
    cdf_HD = ECDF(delta_dist[sele_HD])

    xx = np.linspace(np.nanpercentile(delta_dist[sele_LD | sele_HD], 1),np.nanpercentile(delta_dist[sele_LD | sele_HD],99),1000)
    plt.plot(xx, cdf_HD(xx), label='hd')
    plt.plot(xx, cdf_LD(xx), label='ld')
    
    plt.xlabel('cm')
    plt.ylabel('hist')
    
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot(xlim,[0.5]*2,'--k')
    plt.legend(frameon=False,fontsize=8)

    kk+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# bias analysis
plt.figure(figsize=(10,4))
plt.suptitle('rewarded trials')
kk = 1
for session in np.unique(info_trials['session']):
    print(session)
    plt.subplot(2,5,kk)
    plt.title(session)
    sele_HD = (info_trials['session'] == session) & (info_trials['density'] == 0.005) & (info_trials['reward']==1)
    sele_LD = (info_trials['session'] == session) & (info_trials['density'] == 0.0001) & (info_trials['reward']==1)
    
    cdf_LD = ECDF(delta_dist[sele_LD])
    cdf_HD = ECDF(delta_dist[sele_HD])

    xx = np.linspace(np.nanpercentile(delta_dist[sele_LD | sele_HD], 1),np.nanpercentile(delta_dist[sele_LD | sele_HD],99),1000)
    plt.plot(xx, cdf_HD(xx), label='hd')
    plt.plot(xx, cdf_LD(xx), label='ld')
    
    plt.xlabel('cm')
    plt.ylabel('hist')
    
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.plot(xlim,[0.5]*2,'--k')
    plt.legend(frameon=False,fontsize=8)
# 
    kk+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# bias analysis
plt.figure(figsize=(12,6))
plt.suptitle('trajectory rewarded trials')
kk = 1
session = 'm53s86'
sele_HD = (info_trials['session'] == session) & (info_trials['density'] == 0.005) & (info_trials['reward']==1)
sele_LD = (info_trials['session'] == session) & (info_trials['density'] == 0.0001) & (info_trials['reward']==1)
 
for k in range(1,11):
    ax = plt.subplot(2,5,kk)
    
      
    

    xx = trajectories[sele_HD][k-1]['x_monk']
    yy = trajectories[sele_HD][k-1]['y_monk']
    dd = delta_dist[sele_HD][k-1]
    # ts = trajectory_all_session[tr]['ts']
    # sele = (ts >= init_cond_session['t_stop'][tr]) * (exp_data.behav.time_stamps[idx[k]] <= exp_data.behav.events.t_stop[idx[k]])
    
    x_fly = init_conditions[sele_HD][k-1]['x_fly']
    y_fly = init_conditions[sele_HD][k-1]['y_fly']
    print(x_fly,y_fly)
    plt.plot(xx,yy)
    plt.plot([x_fly],[y_fly],'ok')
    tp = np.linspace(0,np.pi*2,1000)
    cc = np.cos(tp)*60
    ss = np.sin(tp)*60
    plt.plot(cc+x_fly,ss+y_fly,'r')
    plt.ylim(-0,450)
    plt.xlim(-350,450)
    ax.set_aspect('equal')
    plt.title('dd %.1f'%dd)

    
    plt.xlabel('cm')
    plt.ylabel('hist')
    
    ylim = plt.ylim()
    xlim = plt.xlim()
    plt.legend(frameon=False,fontsize=8)

    kk+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    

plt.figure()
ax = plt.subplot(111)
plt.title('Mean undershooting')
plt.boxplot([mean_HD,mean_LD])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks([1,2],['high density','low density'], fontsize=15)
plt.ylabel('cm',fontsize=15)

plt.savefig('behavior_bias.png')

sele_example = (info_trials['session'] == session) & (info_trials['density'] == 0.005) & (info_trials['reward']==0)

srt_dist = np.sort(delta_dist)
idx_sort = np.argsort(delta_dist)



under_shoot = np.where(srt_dist>-80)[0][3]
over_shoot = np.where(srt_dist<100)[0][-1]

sele_under = idx_sort[under_shoot]
sele_over = idx_sort[over_shoot]


plt.figure()
ax = plt.subplot(111)
xx = trajectories[sele_under]['x_monk']
yy = trajectories[sele_under]['y_monk']
dd = delta_dist[sele_under]
# ts = trajectory_all_session[tr]['ts']
# sele = (ts >= init_cond_session['t_stop'][tr]) * (exp_data.behav.time_stamps[idx[k]] <= exp_data.behav.events.t_stop[idx[k]])

x_fly = init_conditions[sele_under]['x_fly']
y_fly = init_conditions[sele_under]['y_fly']
print(x_fly,y_fly)
plt.plot(xx,yy)
plt.plot([x_fly],[y_fly],'ok')
tp = np.linspace(0,np.pi*2,1000)
cc = np.cos(tp)*60
ss = np.sin(tp)*60
plt.plot(cc+x_fly,ss+y_fly,'r')
plt.ylim(-100,350)
plt.xlim(-125,325)
ax.set_aspect('equal')
plt.title('undershooting')
plt.plot([xx[1],xx[-1]],[yy[1],yy[-1]],'--b')
plt.plot([xx[1],x_fly],[yy[1],y_fly],'--r')

plt.xlabel('y [cm]')
plt.ylabel('x [cm]')

ylim = plt.ylim()
xlim = plt.xlim()
plt.legend(frameon=False,fontsize=8)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('undershooting_example.png')

plt.figure()
ax = plt.subplot(111)
xx = trajectories[sele_over]['x_monk']
yy = trajectories[sele_over]['y_monk']
dd = delta_dist[sele_over]
# ts = trajectory_all_session[tr]['ts']
# sele = (ts >= init_cond_session['t_stop'][tr]) * (exp_data.behav.time_stamps[idx[k]] <= exp_data.behav.events.t_stop[idx[k]])

x_fly = init_conditions[sele_over]['x_fly']
y_fly = init_conditions[sele_over]['y_fly']

print(x_fly,y_fly)
plt.plot(xx,yy)
plt.plot([x_fly],[y_fly],'ok')
tp = np.linspace(0,np.pi*2,1000)
cc = np.cos(tp)*60
ss = np.sin(tp)*60
plt.plot(cc+x_fly,ss+y_fly,'r')
plt.ylim(-100,250)
plt.xlim(-175,175)
ax.set_aspect('equal')
plt.title('overshoothing')

plt.plot([xx[1],xx[-1]],[yy[1],yy[-1]],'--b')
plt.plot([xx[1],x_fly],[yy[1],y_fly],'--r')


plt.xlabel('y [cm]')
plt.ylabel('x [cm]')

ylim = plt.ylim()
xlim = plt.xlim()
plt.legend(frameon=False,fontsize=8)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('overshooting_example.png')

    