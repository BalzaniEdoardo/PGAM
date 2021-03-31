import numpy as np
import matplotlib.pylab as plt
import dill
import os,re
from copy import deepcopy
import pandas as pd
import seaborn as sbn
from time import perf_counter
from matplotlib.patches import PathPatch
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts
import seaborn as sbn
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from scipy.io import savemat
def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
                        
                        

color_dict={'PPC':'b','PFC':'r','MST':'g','VIP':'k'}
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


filt = (mutual_info['manipulation_type'] == 'all') & (mutual_info['pseudo-r2'] > 0.01) &\
         (~np.isnan(mutual_info['mutual_info'])) & filt_sess

tuning = tuning[filt]
mutual_info = mutual_info[filt]


dprime_vec = np.zeros(tuning.shape)
cc = 0
for tun in tuning:
    #
    # dprime_vec[cc] = np.abs(np.mean(tun['y_raw'] - tun['y_model']))/(np.mean(tun['y_raw']))#/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))
    
    dprime_vec[cc] = np.max(np.abs(tun['y_raw'] - tun['y_model']))/(np.mean(tun['y_raw']))#/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))

    cc += 1

# remove crazy outliers and distribution tails
filt = dprime_vec < np.nanpercentile(dprime_vec,98)
tuning = tuning[filt]
mutual_info = mutual_info[filt]
dprime_vec = dprime_vec[filt]


# get the firing rate and sort them 
# mutual_info = np.sort(mutual_info,order=['session','neuron'])
# firing_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/firing_rate_condition/firing_rate_info2.npy')
# firing_info = np.sort(firing_info,order=['session','neuron'])
# firing_info = firing_info[firing_info['manipulation_type'] == 'all']


x_ticks_dict = {'t_move':[-0.4,0,0.4],
                't_stop':[-0.4,0,0.4],
                't_reward':[-0.4,0,0.4],
                't_flyOFF':[-1,0,1],
                'rad_vel':[0, 200],
                'rad_acc':[-900,900],
                'rad_target':[0,390],
                'rad_path':[0,390],
                'ang_acc':[-230,230],
                'ang_vel':[-60,60],
                'ang_target':[-45,45],
                'ang_path':[-55,55],
                'lfp_beta':['$-\pi$','0','$\pi$'],
                'lfp_alpha':['$-\pi$','0','$\pi$'],
                'lfp_theta':['$-\pi$','0','$\pi$'],
                'eye_vert':[-2,2],
                'eye_hori':[-2,2]}

title_dict = {'t_move':'move onset',
                't_stop':'move offset',
                't_reward':'reward',
                't_flyOFF':'target onset',
                'rad_vel':'linear velocity',
                'rad_acc':'acceleration',
                'rad_target':'target distance',
                'rad_path':'dist travelled',
                'ang_acc':'angular acceleration',
                'ang_vel':'angular velocity',
                'ang_target':'target angle',
                'ang_path':'angle turned',
                'lfp_beta':'beta phase',
                'lfp_alpha':'alpha phase',
                
                'lfp_theta':'lfp theta',
                'eye_vert':'eye vert',
                'eye_hori':'eye hori',
                
                }

x_label_dict = {'t_move':'time [sec]',
                't_stop':'time [sec]',
                't_reward':'time [sec]',
                't_flyOFF':'time [sec]',
                'rad_vel':'cm/sec',
                'rad_acc':'cm/sec^2',
                'rad_target':'cm',
                'rad_path':'cm',
                'rad_path':'dist travelled',
                'ang_acc':'deg/sec^2',
                'ang_vel':'deg/sec',
                'ang_target':'deg',
                'ang_path':'deg',
                'lfp_beta':'rad',
                'lfp_alpha':'rad',
                'lfp_theta':'rad',
                'eye_vert':'',
                'eye_hori':''
                
                }

# choose a var and filter
var = 'rad_target'

plt.close('all')

for var in title_dict.keys():
    keep = (mutual_info['variable']==var) & (mutual_info['significance'])
    mi_rad_target = mutual_info['mutual_info'][keep]
    tuning_var = tuning[keep]
    mutual_info_var = mutual_info[keep]

    if 'lfp' in var:
        prc = 90
    elif var =='rad_target':
        prc = 80
    else:
        prc = 90

    keep = np.zeros(mi_rad_target.shape[0],dtype=bool)
    for ba in ['MST','PPC','PFC']:
        sel = mutual_info_var['brain_area']==ba
        keep[sel] = np.log(mi_rad_target)[sel] > np.nanpercentile(np.log(mi_rad_target)[sel],prc)
    
    tuning_var =  tuning_var[keep]
    mutual_info_var =  mutual_info_var[keep]

    
    mat_tuning = np.zeros((tuning_var.shape[0],15))
    peak_vec = np.zeros(tuning_var.shape[0])
    brain_area_vec = tuning_var['brain_area']
    monkey_vec = tuning_var['monkey']
    session_vec = tuning_var['session']
    neuron_vec = tuning_var['neuron']

    cc = 0
    for row in tuning_var:
        mat_tuning[cc,:] = tuning_var[cc]['y_model']
        peak_vec[cc] = np.argmax(mat_tuning[cc,:])
        cc += 1
    
    srt = np.argsort(peak_vec)
    z_score = sts.zscore(mat_tuning,axis=1)
    
    
    
    z_score = z_score[srt]
    tuning_var = tuning_var[srt]


    model = PCA()
    model.fit(z_score)
    center_zscore = z_score - z_score.mean(axis=0)

    
    num = np.where(np.cumsum(model.explained_variance_ratio_) > 0.9)[0][0]
    if num == 1:
        num=min(2,num)
        P = model.components_.T[:, :num + 1]
        transf = np.dot(center_zscore,P)
    else:
        P = model.components_.T[:, :num + 1]
        center_zscore = np.dot(np.dot(z_score,P),P.T)
        tsne = TSNE()
        transf = tsne.fit_transform(center_zscore)
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan.fit(transf)


    dd_mat = {'monkey':monkey_vec,'brain_area':brain_area_vec,'session':session_vec,
              'zscore_tun':center_zscore,'cluster_labels':dbscan.labels_,'low_dim_proj':transf}

    matr = np.zeros(monkey_vec.shape,dtype={'names':['monkey','brain_area','session','neuron',
                                                     'zscore_tun','cluster_labels','low_dim_proj'],
                                            'formats':['U30','U30','U30',int,object,int,object]})

    matr['monkey'] = monkey_vec
    matr['brain_area'] = brain_area_vec
    matr['session'] = session_vec

    matr['cluster_labels'] = dbscan.labels_
    for k in range(matr.shape[0]):
        matr['zscore_tun'][k] = center_zscore[k]
        matr['low_dim_proj'][k] = transf[k]
    matr['neuron'] = neuron_vec


    savemat('%s_clust_result.mat'%var,mdict={'clust_res':matr})
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    plt.suptitle(var)
    for cl in np.unique(dbscan.labels_):
        cl_trans = transf[dbscan.labels_==cl]
        if cl == -1:
            p = ax1.scatter(cl_trans[:,0],cl_trans[:,1],color=(0.5,)*3,alpha=0.4,s=2)
        elif (dbscan.labels_==cl).sum()<20:
            continue
        else:
            p = ax1.scatter(cl_trans[:, 0], cl_trans[:, 1])
            col = p.get_facecolor()[0][:3]
            ax2.plot(center_zscore[dbscan.labels_==cl].mean(axis=0),color=col)
            xx = np.arange(center_zscore.shape[1])
            ax2.fill_between(xx, center_zscore[dbscan.labels_==cl].mean(axis=0)-center_zscore[dbscan.labels_==cl].std(axis=0),
                             center_zscore[dbscan.labels_==cl].mean(axis=0) + center_zscore[dbscan.labels_==cl].std(axis=0), color=col,alpha=0.4)
    # plt.savefig('tsne_%s.pdf'%var)
    # plotting stuff
    #
    # plt.figure(figsize=(10, 4))
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    #
    # plt.suptitle(var)
    # for cl in np.unique(dbscan.labels_):
    #     cl_trans = transf[dbscan.labels_ == cl]
    #     if cl == -1:
    #         p = ax1.scatter(cl_trans[:, 0], cl_trans[:, 1], color=(0.5,) * 3, alpha=0.4, s=2)
    #     # elif (dbscan.labels_ == cl).sum() < 20:
    #     #     continue
    #     else:
    #         p = ax1.scatter(cl_trans[:, 0], cl_trans[:, 1])
    #         col = p.get_facecolor()[0][:3]
    #         ax2.plot(center_zscore[dbscan.labels_ == cl].mean(axis=0), color=col)
    #         xx = np.arange(center_zscore.shape[1])
    #         ax2.fill_between(xx,
    #                          center_zscore[dbscan.labels_ == cl].mean(axis=0) - center_zscore[dbscan.labels_ == cl].std(
    #                              axis=0),
    #                          center_zscore[dbscan.labels_ == cl].mean(axis=0) + center_zscore[dbscan.labels_ == cl].std(
    #                              axis=0), color=col, alpha=0.4)
    # plt.savefig('all_tsne_%s.pdf' % var)
    # plotting stuff



    # if len(x_ticks_dict[var]) == 2:
    #     xticks_pos = [0.5,14.5]
    
    # if len(x_ticks_dict[var]) == 3:
    #     xticks_pos = [0.5, 7.5, 14.5]
    
    
    # plt.figure(figsize=[10,4.5])
    # plt.suptitle('%s - Firing Rate'%title_dict[var])
    # ax = plt.subplot(131)
    # plt.title('MST')
    # plt.ylabel('units')
    # sbn.heatmap(center_zscore[tuning_var['brain_area']=='MST'],cmap=cm.Greens,ax=ax,cbar=False)
    # plt.xticks(xticks_pos,x_ticks_dict[var],rotation=0)
    # plt.xlabel(x_label_dict[var])
    # plt.yticks([])
    
    
    
    # ax = plt.subplot(132)
    # plt.title('PPC')

    # sbn.heatmap(center_zscore[tuning_var['brain_area']=='PPC'],cmap=cm.Blues,ax=ax,cbar=False)
    # plt.xticks(xticks_pos,x_ticks_dict[var],rotation=0)
    # plt.xlabel(x_label_dict[var])
    
    # plt.yticks([])
    
    # ax = plt.subplot(133)
    # plt.title('PFC')

    # sbn.heatmap(z_score[tuning_var['brain_area']=='PFC'],cmap=cm.Reds,ax=ax,cbar=False)
    
    
    # plt.xticks(xticks_pos,x_ticks_dict[var],rotation=0)
    # plt.xlabel(x_label_dict[var])
    # plt.yticks([])
    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig('%s_firingHeatmap.png'%var)
    # plt.close('all')

