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
mutual_info = np.sort(mutual_info,order=['session','neuron'])
firing_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/firing_rate_condition/firing_rate_info2.npy')
firing_info = np.sort(firing_info,order=['session','neuron'])
firing_info = firing_info[firing_info['manipulation_type'] == 'all']

# create bit/spk 
mutual_info_bitspk = deepcopy(mutual_info)
for session in np.unique(firing_info['session']):
    fr_sess = firing_info[firing_info['session'] == session]
    bl_sess = mutual_info_bitspk['session'] == session
    mi_sess = mutual_info_bitspk[bl_sess]
    for row in fr_sess:
        neu = row['neuron']
        fr = row['firing_rate']
        bl_neu = mi_sess['neuron'] == neu
        mutual_info_bitspk[bl_sess][bl_neu]['mutual_info'] = \
            mutual_info_bitspk[bl_sess][bl_neu]['mutual_info'] / fr
            
df = pd.DataFrame(mutual_info_bitspk)

# plt.figure(figsize=[14.36,  6.  ])
# ax = plt.subplot(111)
# sbn.boxplot(x='variable',y='mutual_info',hue='brain_area',palette={'PPC':'b','PFC':'r','MST':'g','VIP':'k'},
#             order=['rad_vel','ang_vel','rad_acc','ang_acc','rad_path',
#                    'ang_path','rad_target','ang_target','lfp_beta','lfp_alpha','lfp_theta',
#                    't_move','t_flyOFF','t_stop','t_reward','eye_vert','eye_hori'],
#             hue_order=['MST','PPC','VIP','PFC'],data=df,ax=ax,linewidth=.25)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
# srt = np.argsort(mutual_info['mutual_info'])[::-1]
# plt.tight_layout()
# plt.legend(loc=1)
# plt.ylabel('mutual info bit/spk')
# plt.savefig('mutual_info_bit_spk.png')

# plt.figure(figsize=[14.36,  6.  ])
# ax = plt.subplot(111)
# sbn.boxplot(x='variable',y='mutual_info',hue='brain_area',palette={'PPC':'b','PFC':'r','MST':'g','VIP':'k'},
#             order=['rad_vel','ang_vel','rad_acc','ang_acc','rad_path',
#                    'ang_path','rad_target','ang_target','lfp_beta','lfp_alpha','lfp_theta',
#                    't_move','t_flyOFF','t_stop','t_reward','eye_vert','eye_hori'],
#             hue_order=['MST','PPC','VIP','PFC'],data=df,ax=ax,linewidth=.25,fliersize=0)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
# srt = np.argsort(mutual_info['mutual_info'])[::-1]
# plt.tight_layout()
# plt.ylabel('mutual info bit/spk')
# plt.ylim(0,2.)
# plt.legend(loc=1)
# plt.savefig('mutual_info_bit_spk_nofliers.png')


# plt.figure(figsize=(10,8))
# cc = 1
# ba = 'MST'
# plt.suptitle('log-transformed mutual info')

# filt = mutual_info_bitspk['brain_area'] == ba
# for var in np.unique(mutual_info['variable']):
#     plt.subplot(5,4,cc)
#     plt.title(var)

#     mi = mutual_info_bitspk['mutual_info'][filt&(mutual_info_bitspk['variable'] == var)]
#     plt.hist(np.log(mi),bins=10,color=color_dict[ba])
#     cc+=1
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('mt_info_dist_log.png')

# plt.figure(figsize=(10,8))
# cc = 1
# ba = 'MST'
# plt.suptitle('raw mutual info')
# filt = mutual_info_bitspk['brain_area'] == ba
# for var in np.unique(mutual_info['variable']):
#     plt.subplot(5,4,cc)
#     plt.title(var)
#     mi = mutual_info_bitspk['mutual_info'][filt&(mutual_info_bitspk['variable'] == var)]
#     plt.hist(mi,bins=10,color=color_dict[ba])
#     cc+=1
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('mt_info_dist_raw.png')





# 

# plt.close('all')
# plt.figure(figsize=[14.36,  6.  ])
# ax = plt.subplot(111)
# sbn.violinplot(x='variable',y='mutual_info',hue='brain_area',palette={'PPC':'b','PFC':'r','MST':'g','VIP':'k'},
#             order=['rad_vel','ang_vel','rad_acc','ang_acc','rad_path',
#                    'ang_path','rad_target','ang_target','lfp_beta','lfp_alpha','lfp_theta',
#                    't_move','t_flyOFF','t_stop','t_reward','eye_vert','eye_hori'],
#             hue_order=['MST','PPC','VIP','PFC'],data=df2,ax=ax,linewidth=.25,fliersize=0,trim=True)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
# srt = np.argsort(mutual_info['mutual_info'])[::-1]
# plt.tight_layout()
# plt.ylabel('log(bit/spk)')
# # plt.ylim(0,2.)
# plt.legend(loc=1)
# plt.savefig('mutual_info_bit_spk_logscale.png')


group_var = {
    'sensorimotor':['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF'],
    'internal':['rad_target','ang_target','rad_path','ang_path'],
    'LFP':['lfp_beta','lfp_alpha','lfp_theta'],
    'other':['t_reward','eye_vert','eye_hori'],
    }



width = 8.7/7
plt.close('all')
for gr in group_var.keys():
    df2 = pd.DataFrame(mutual_info)
    df2['mutual_info'] = np.log(df2['mutual_info'])
    filt = np.zeros(df2.shape[0],dtype=bool)
    for var in group_var[gr]:
        filt[df2['variable']==var] = True
    filt = filt & (df2['brain_area']!='VIP')
    df2 = df2[filt]
    fig = plt.figure(figsize=(width*len(group_var[gr]),4))
    ax = plt.subplot(111)
    bxp = sbn.boxplot(x='variable',y='mutual_info',hue='brain_area',palette={'PPC':'b','PFC':'r','MST':'g','VIP':'k'},
                order=group_var[gr],
                hue_order=['MST','PPC','PFC'],data=df2,ax=ax,linewidth=.25,fliersize=0,width=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    srt = np.argsort(mutual_info['mutual_info'])[::-1]
    plt.tight_layout()
    plt.ylabel('log(bit/sec)')
    # plt.ylim(0,2.)
    # plt.legend(loc=1)
    bxp.legend_.remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    adjust_box_widths(fig, 0.8)
    plt.savefig('bxplot_%s.png'%gr)
    


result_statistics = np.zeros(len(np.unique(mutual_info['variable'])),
                             dtype={'names':('variable','SS-brain_area','df-brain_area','SS-Residual','df-Residual','F-stat','p-val','R^2','effect-size'),
                                    'formats':('U30',float,float,float,float,float,float,float,'U30')})
# anova results
cc = 0
for gr in group_var.keys():
    
    for var in group_var[gr]:
        df2 = pd.DataFrame(mutual_info)
        df2['mutual_info'] = np.log(df2['mutual_info'])
        filt = np.zeros(df2.shape[0],dtype=bool)
        filt[df2['variable']==var] = True
        filt = filt & (df2['brain_area']!='VIP')
        df2 = df2[filt]
        mod = ols('mutual_info ~ brain_area',data=df2).fit()
        aov_table = sm.stats.anova_lm(mod,typ=2)
        if mod.rsquared < 0.01:
            lab = 'No effect'
        elif mod.rsquared >= 0.01 and mod.rsquared < 0.06:
            lab = 'Small effect'
        elif mod.rsquared >= 0.06 and mod.rsquared < 0.14:
            lab = 'Medium effect'
        elif mod.rsquared >= 0.14:
            lab = 'Large effect'
        print(var,'R^2', mod.rsquared, lab)
        print(aov_table)
        print('\n')
        result_statistics[cc]['variable'] = var
        result_statistics[cc]['SS-brain_area'] = aov_table.sum_sq.brain_area
        result_statistics[cc]['SS-Residual'] = aov_table.sum_sq.Residual
        result_statistics[cc]['df-Residual'] = aov_table.df.Residual
        result_statistics[cc]['df-brain_area'] = aov_table.df.brain_area
        result_statistics[cc]['F-stat'] = aov_table.F.brain_area
        result_statistics[cc]['p-val'] = aov_table.T.brain_area[3]
        result_statistics[cc]['R^2'] = mod.rsquared
        result_statistics[cc]['effect-size'] = lab
        
        cc+=1
        
mi_stat = pd.DataFrame(result_statistics)
writer = pd.ExcelWriter('mutual_info_statistics.xlsx')
mi_stat.to_excel(writer,index=False)
writer.save()
writer.close()
        


width = 8.7/7
plt.close('all')
for gr in group_var.keys():
    df2 = pd.DataFrame(mutual_info)
    df2['mutual_info'] = np.log(df2['mutual_info'])
    filt = np.zeros(df2.shape[0],dtype=bool)
    for var in group_var[gr]:
        filt[df2['variable']==var] = True
    filt = filt & (df2['brain_area']!='VIP')
    df2 = df2[filt]
    fig = plt.figure(figsize=(width*len(group_var[gr]),4))
    ax = plt.subplot(111)
    bxp = sbn.pointplot(x='variable',y='mutual_info',hue='brain_area',palette={'PPC':'b','PFC':'r','MST':'g','VIP':'k'},
                order=group_var[gr],
                hue_order=['MST','PPC','PFC'],data=df2,ax=ax,linestyles='none')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    srt = np.argsort(mutual_info['mutual_info'])[::-1]
    plt.tight_layout()
    plt.ylabel('log(bit/sec)')
    # plt.ylim(0,2.)
    # plt.legend(loc=1)
    bxp.legend_.remove()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    adjust_box_widths(fig, 0.8)
    plt.savefig('bxplot_%s.png'%gr)
    

        
# plt.savefig('mutual_info_bit_spk_logscale.png')
