import numpy as np
import matplotlib.pylab as plt
import dill
import os,re
from copy import deepcopy
import pandas as pd
import seaborn as sbn


fld_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info/'
lst_done = os.listdir(fld_file)
# mutual_info_and_tunHz_m53s42.dill

first = True
for fh in lst_done:
    if not re.match('^mutual_info_and_tunHz_m\d+s\d+.dill$',fh):
        continue

    with open(os.path.join(fld_file,fh),'rb') as fh:
        res = dill.load(fh)
        mi = res['mutual_info']
        tun = res['tuning_Hz']
    if first:
        mutual_info = deepcopy(mi)
        tuning = deepcopy(tun)
        first = False
    else:
        mutual_info = np.hstack((mutual_info,mi))
        tuning = np.hstack((tuning, tun))

# filter only density manip
keep_sess = np.unique(mutual_info['session'][mutual_info['manipulation_type']=='density'])
filt_sess = np.zeros(mutual_info.shape,dtype=bool)
for sess in keep_sess:
    filt_sess[mutual_info['session']==sess] = True

dprime_vec = np.zeros(tuning.shape)
cc = 0
for tun in tuning:
    dprime_vec[cc] = np.mean(tun['y_raw'] - tun['y_model'])/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))
    cc += 1


filter = (np.abs(dprime_vec)<0.1) & (mutual_info['manipulation_type'] == 'all') & (mutual_info['pseudo-r2'] > 0.005) &\
         filt_sess & (~np.isnan(mutual_info['mutual_info']))
mutual_info = mutual_info[filter]
tuning = tuning[filter]
dprime_vec = dprime_vec[filter]

df = pd.DataFrame(mutual_info)

plt.figure(figsize=[14.36,  6.  ])
ax = plt.subplot(111)
sbn.boxplot(x='variable',y='mutual_info',hue='brain_area',palette={'PPC':'b','PFC':'r','MST':'g','VIP':'k'},
            order=['rad_vel','ang_vel','rad_acc','ang_acc','rad_path',
                   'ang_path','rad_target','ang_target','lfp_beta','lfp_alpha','lfp_theta',
                   't_move','t_flyOFF','t_stop','t_reward','eye_vert','eye_hori'],
            hue_order=['MST','PPC','VIP','PFC'],data=df,ax=ax,linewidth=.25)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
srt = np.argsort(mutual_info['mutual_info'])[::-1]
plt.tight_layout()
plt.savefig('mutual_info_bxplt.pdf')

# var = 'rad_target'
# srt = np.argsort(mutual_info['mutual_info'][sel])[::-1]

# colDict = {'PPC':'b','PFC':'r','MST':'g','VIP':'k'}
# for num_plt in range(0,30):
#     plt_num = np.arange(num_plt * 25, (num_plt + 1) * 25, dtype=int)
#     plt.figure(figsize=(12,10))
#     idx0 = srt[plt_num[0]]
#     idxend = srt[plt_num[-1]]
#     m0 = mutual_info['mutual_info'][idx0]
#     m1 = mutual_info['mutual_info'][idxend]
#     plt.suptitle('%.4f - %.4f'%(m1,m0))

#     for k in range(25):

#         tun = tuning[srt[plt_num[k]]]

#         ax = plt.subplot(5,5,k+1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         dprime = np.mean(tun['y_raw'] - tun['y_model'])/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))

#         ax.set_title('c%d %s-%s'%(mutual_info['neuron'][srt[plt_num[k]]],tun['session'],tun['variable']))
#         ax.plot(tun['x'],tun['y_raw'], label='raw',color=(0.5,)*3)
#         ax.plot(tun['x'],tun['y_model'], label='model',color=colDict[tun['brain_area']])
#         plt.xticks([])

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     plt.savefig('%s_large_MI_%d.png'%(var,num_plt))
#     plt.close('all')



# colDict = {'PPC':'b','PFC':'r','MST':'g','VIP':'k'}
# for num_plt in range(0,30):
#     plt_num = np.arange(num_plt * 25, (num_plt + 1) * 25, dtype=int)
#     plt.figure(figsize=(12,10))
#     idx0 = srt[plt_num[0]]
#     idxend = srt[plt_num[-1]]
#     m0 = mutual_info['mutual_info'][idx0]
#     m1 = mutual_info['mutual_info'][idxend]
#     plt.suptitle('%.4f - %.4f'%(m1,m0))
#
#     for k in range(25):
#
#         tun = tuning[srt[plt_num[k]]]
#
#         ax = plt.subplot(5,5,k+1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         dprime = np.mean(tun['y_raw'] - tun['y_model'])/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))
#
#         ax.set_title('c%d %s-%s'%(mutual_info['neuron'][srt[plt_num[k]]],tun['session'],tun['variable']))
#         ax.plot(tun['x'],tun['y_raw'], label='raw',color=(0.5,)*3)
#         ax.plot(tun['x'],tun['y_model'], label='model',color=colDict[tun['brain_area']])
#         plt.xticks([])
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     plt.savefig('large_MI_%d.png'%num_plt)
#     plt.close('all')

#
# srt = np.argsort(dprime_vec)
#
# colDict = {'PPC': 'b', 'PFC': 'r', 'MST': 'g', 'VIP': 'k'}
# for num_plt in range(70, 80):
#     plt_num = np.arange(num_plt * 25, (num_plt + 1) * 25, dtype=int)
#     plt.figure(figsize=(12, 10))
#     idx0 = srt[plt_num[0]]
#     idxend = srt[plt_num[-1]]
#     m0 = dprime_vec[idx0]
#     m1 = dprime_vec[idxend]
#     plt.suptitle('%.4f - %.4f' % (m1, m0))
#
#     for k in range(25):
#         tun = tuning[srt[plt_num[k]]]
#
#         ax = plt.subplot(5, 5, k + 1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         # dprime = np.mean(tun['y_raw'] - tun['y_model']) / (0.5 * (np.std(tun['y_raw']) + np.std(tun['y_model'])))
#
#         ax.set_title('%.2f %s-%s' % (mutual_info['mutual_info'][srt[plt_num[k]]], tun['session'], tun['variable']))
#         ax.plot(tun['x'], tun['y_raw'], label='raw', color=(0.5,) * 3)
#         ax.plot(tun['x'], tun['y_model'], label='model', color=colDict[tun['brain_area']])
#         plt.xticks([])
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     # plt.savefig('large_MI_%d.png'%num_plt)
#     # plt.close('all')
#
#
