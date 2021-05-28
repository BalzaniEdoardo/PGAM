import numpy as np
import matplotlib.pylab as plt
import dill,os,re
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import scipy.stats as sts
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import seaborn as sbs
from scipy.io import savemat

from seaborn.algorithms import bootstrap


fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info/'
##
# load MI
##
first = True
for fhName in os.listdir(fld):
    if not re.match('mutual_info_and_tunHz_m\d+s\d+.dill',fhName):
        continue

    with open(os.path.join(fld, fhName), 'rb') as fh:
        res = dill.load(fh)

    mi = res['mutual_info']
    tun = res['tuning_Hz']
    keep = (mi['manipulation_type'] == 'density') & (mi['pseudo-r2'] > 0.01) & (~np.isnan(mi['mutual_info']))
    tun = tun[keep]
    mi = mi[keep]

    if mi.shape[0] == 0:
        continue

    if first:
        first = False
        mutual_info = deepcopy(mi)
        tuning = deepcopy(tun)
    else:
        mutual_info = np.hstack((mutual_info, mi))
        tuning = np.hstack((tuning,tun))

keep = (mutual_info['variable'] != 'lfp_beta') & (mutual_info['variable'] != 'lfp_alpha') & (mutual_info['variable'] != 'lfp_theta')
mutual_info = mutual_info[keep]
tuning = tuning[keep]


dprime_vec = np.zeros(tuning.shape)
cc = 0
for tun in tuning:
    #
    # dprime_vec[cc] = np.abs(np.mean(tun['y_raw'] - tun['y_model']))/(np.mean(tun['y_raw']))#/(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))

    dprime_vec[cc] = np.max(np.abs(tun['y_raw'] - tun['y_model'])) / (
        np.mean(tun['y_raw']))  # /(0.5*(np.std(tun['y_raw']) + np.std(tun['y_model'])))

    cc += 1

# remove crazy outliers and distribution tails
filt = dprime_vec < np.nanpercentile(dprime_vec, 98)
tuning = tuning[filt]
mutual_info = mutual_info[filt]
dprime_vec = dprime_vec[filt]


keep = np.zeros(mutual_info.shape[0],dtype=bool)
for var in np.unique(mutual_info['variable']):
    idx = np.where(mutual_info['variable']==var)[0]

    sel = (mutual_info['mutual_info'][idx] < np.nanpercentile(mutual_info['mutual_info'][idx],95))
    keep[idx[sel]] = True
    # mutual_info = mutual_info[keep]


# split by type
tun_hd = tuning[mutual_info['manipulation_value'] == 0.005]
tun_ld = tuning[mutual_info['manipulation_value'] == 0.0001]

mi_hd = mutual_info[mutual_info['manipulation_value'] == 0.005]
mi_ld = mutual_info[mutual_info['manipulation_value'] == 0.0001]

mutual_info_hd = np.zeros(min(mi_ld.shape[0],mi_hd.shape[0]),dtype=mutual_info.dtype)
mutual_info_ld = np.zeros(min(mi_ld.shape[0],mi_hd.shape[0]),dtype=mutual_info.dtype)

tuning_hd = np.zeros(min(mi_ld.shape[0],mi_hd.shape[0]),dtype=tuning.dtype)
tuning_ld = np.zeros(min(mi_ld.shape[0],mi_hd.shape[0]),dtype=tuning.dtype)


cc = 0
session_list = np.unique(mi_hd['session'])
var_list = np.unique(mi_hd['variable'])
for session in session_list:
    print('pairing session',session)
    mi_hd_sess = mi_hd[mi_hd['session'] == session]
    mi_ld_sess = mi_ld[mi_ld['session'] == session]

    tuning_ld_sess = tun_ld[mi_ld['session'] == session]
    tuning_hd_sess = tun_hd[mi_hd['session'] == session]

    for var in var_list:
        mi_hd_var = mi_hd_sess[mi_hd_sess['variable']==var]
        mi_ld_var = mi_ld_sess[mi_ld_sess['variable']==var]

        tuning_ld_var = tuning_ld_sess[mi_ld_sess['variable'] == var]
        tuning_hd_var = tuning_hd_sess[mi_hd_sess['variable'] == var]

        for info_ld in mi_ld_var:
            info_hd = mi_hd_var[mi_hd_var['neuron']==info_ld['neuron']]

            if info_hd.shape[0] == 1:
                mutual_info_hd[cc] = info_hd
                mutual_info_ld[cc] = info_ld

                tuning_hd[cc] = tuning_hd_var[mi_hd_var['neuron']==info_ld['neuron']]
                tuning_ld[cc] = tuning_ld_var[mi_ld_var['neuron']==info_ld['neuron']]

                cc+=1
            elif info_hd.shape[0] == 0:
                continue
            else:
                raise ValueError('more then one variable found')

# for info in mi_ld:
del mi_ld_sess,mi_hd_sess,mi_hd,mi_ld,mi_ld_var,mi_hd_var,tuning_ld_sess,tuning_hd_sess,tuning_ld_var,tuning_hd_var


keep = mutual_info_ld['monkey'] != ''
mutual_info_ld = mutual_info_ld[mutual_info_ld['monkey'] != '']
mutual_info_hd = mutual_info_hd[mutual_info_hd['monkey'] != '']
tuning_ld = tuning_ld[keep]
tuning_hd = tuning_hd[keep]

color_dict = {'MST':'g','PFC':'r','PPC':'b'}
plt.figure(figsize=([13.44,  8.  ]))
kk = 1

order = ['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF','rad_target',
         'ang_target','rad_path','ang_path','t_reward','eye_vert','eye_hori']

for var in order:
    plt.subplot(3,5,kk)

    regr_dict = {}
    idx = mutual_info_ld['variable'] == var
    for ba in ['PPC', 'PFC', 'MST']:
        iidx = idx & (mutual_info_ld['brain_area']==ba)
        plt.scatter(mutual_info_ld['mutual_info'][iidx], mutual_info_hd['mutual_info'][iidx], c=color_dict[ba],
                    s=8, alpha=0.5)
        model = LinearRegression(fit_intercept=True)
        regr_dict[ba] = model.fit(mutual_info_ld['mutual_info'][iidx].reshape(iidx.sum(),1), mutual_info_hd['mutual_info'][iidx])

    xlim = plt.xlim()
    xlim = np.array(xlim)

    for ba in ['PPC', 'PFC', 'MST']:
        yy = xlim * regr_dict[ba].coef_ + regr_dict[ba].intercept_
        plt.plot(xlim, yy, color_dict[ba],lw=1.5,label=ba + ': %.2f'%regr_dict[ba].coef_)
    plt.legend(fontsize=8)
    # plt.plot(xlim, xlim, 'k')
    plt.title(var)
    plt.xlabel('MI low density',fontsize=8)
    plt.ylabel('MI high density',fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    kk+=1
plt.tight_layout()


plt.savefig('mutual_info_compare.png')

order = ['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF','rad_target',
         'ang_target','rad_path','ang_path','t_reward','eye_vert','eye_hori']


ba = 'MST'
for var in order:
    idx = (mutual_info_ld['variable'] == var) * (mutual_info_ld['brain_area']==ba)
    mi_ld = mutual_info_ld[idx]
    mi_hd = mutual_info_hd[idx]
    tun_ld = tuning_ld[idx]
    tun_hd = tuning_hd[idx]
    srt = np.argsort(mi_ld['mutual_info'] - mi_hd['mutual_info'])
    plt_num = 1
    plt.figure(figsize=(10,6))
    plt.suptitle(var)
    for k in range(4):
        plt.subplot(2,4,plt_num)
        plt.plot(tun_ld['x'][srt[k]],tun_ld['y_model'][srt[k]],color='g',label='low density')
        # plt.plot(tun_ld['x'][srt[k]], tun_ld['y_raw'][srt[k]], color='g',ls='--')

        plt.plot(tun_hd['x'][srt[k]], tun_hd['y_model'][srt[k]], color=(.5,)*3,label='high density')
        # plt.plot(tun_hd['x'][srt[k]], tun_hd['y_raw'][srt[k]], color=(.5,)*3,ls='--')

        plt_num += 1

    plt.legend()
    for k in range(4):
        plt.subplot(2, 4, plt_num)
        plt.plot(tun_ld['x'][srt[-(k+1)]],tun_ld['y_model'][srt[-(k+1)]],color='g')
        # plt.plot(tun_ld['x'][srt[-(k+1)]], tun_ld['y_raw'][srt[-(k+1)]], color='g',ls='--')
        plt.plot(tun_hd['x'][srt[-(k+1)]], tun_hd['y_model'][srt[-(k+1)]], color=(.5,)*3)
        # plt.plot(tun_hd['x'][srt[-(k+1)]], tun_hd['y_raw'][srt[-(k+1)]], color=(.5,)*3,ls='--')


        plt_num+=1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('%s_examples/%s.png'%(ba,var))



ba = 'PPC'
for var in order:
    idx = (mutual_info_ld['variable'] == var) * (mutual_info_ld['brain_area']==ba)
    mi_ld = mutual_info_ld[idx]
    mi_hd = mutual_info_hd[idx]
    tun_ld = tuning_ld[idx]
    tun_hd = tuning_hd[idx]
    srt = np.argsort(mi_ld['mutual_info'] - mi_hd['mutual_info'])
    plt_num = 1
    plt.figure(figsize=(10,6))
    plt.suptitle(var)
    for k in range(4):
        plt.subplot(2,4,plt_num)
        plt.plot(tun_ld['x'][srt[k]],tun_ld['y_model'][srt[k]],color='b',label='low density')
        # plt.plot(tun_ld['x'][srt[k]], tun_ld['y_raw'][srt[k]], color='b',ls='--')

        plt.plot(tun_hd['x'][srt[k]], tun_hd['y_model'][srt[k]], color=(.5,)*3,label='high density')
        # plt.plot(tun_hd['x'][srt[k]], tun_hd['y_raw'][srt[k]], color=(.5,)*3,ls='--')

        plt_num += 1

    plt.legend()

    for k in range(4):
        plt.subplot(2, 4, plt_num)
        plt.plot(tun_ld['x'][srt[-(k+1)]],tun_ld['y_model'][srt[-(k+1)]],color='b')
        # plt.plot(tun_ld['x'][srt[-(k+1)]], tun_ld['y_raw'][srt[-(k+1)]], color='b',ls='--')
        plt.plot(tun_hd['x'][srt[-(k+1)]], tun_hd['y_model'][srt[-(k+1)]], color=(.5,)*3)
        # plt.plot(tun_hd['x'][srt[-(k+1)]], tun_hd['y_raw'][srt[-(k+1)]], color=(.5,)*3,ls='--')


        plt_num+=1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('%s_examples/%s.png'%(ba,var))

ba = 'PFC'
for var in order:
    idx = (mutual_info_ld['variable'] == var) * (mutual_info_ld['brain_area'] == ba)
    mi_ld = mutual_info_ld[idx]
    mi_hd = mutual_info_hd[idx]
    tun_ld = tuning_ld[idx]
    tun_hd = tuning_hd[idx]
    srt = np.argsort(mi_ld['mutual_info'] - mi_hd['mutual_info'])
    plt_num = 1
    plt.figure(figsize=(10, 6))
    plt.suptitle(var)
    for k in range(4):
        plt.subplot(2, 4, plt_num)
        plt.plot(tun_ld['x'][srt[k]], tun_ld['y_model'][srt[k]], color='r', label='low density')
        # plt.plot(tun_ld['x'][srt[k]], tun_ld['y_raw'][srt[k]], color='r', ls='--')

        plt.plot(tun_hd['x'][srt[k]], tun_hd['y_model'][srt[k]], color=(.5,) * 3, label='high density')
        # plt.plot(tun_hd['x'][srt[k]], tun_hd['y_raw'][srt[k]], color=(.5,) * 3, ls='--')

        plt_num += 1

    plt.legend()

    for k in range(4):
        plt.subplot(2, 4, plt_num)
        plt.plot(tun_ld['x'][srt[-(k+1)]], tun_ld['y_model'][srt[-(k+1)]], color='r')
        # plt.plot(tun_ld['x'][srt[-(k+1)]], tun_ld['y_raw'][srt[-(k+1)]], color='r', ls='--')
        plt.plot(tun_hd['x'][srt[-(k+1)]], tun_hd['y_model'][srt[-(k+1)]], color=(.5,) * 3)
        # plt.plot(tun_hd['x'][srt[-(k+1)]], tun_hd['y_raw'][srt[-(k+1)]], color=(.5,) * 3, ls='--')

        plt_num += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('%s_examples/%s.png' % (ba, var))

plt.close('all')

print('LINEAR REGRESSION START')
dtype_dict = {'names':('monkey','session','unit','brain_area','variable','slope','intercept','pval','fdr_pval','rsquare','xmin','xmax','mutual_info_hd','mutual_info_ld'),
              'formats':('U30','U30',int,'U30','U30',float,float,float,float,float,float,float,float,float)}
regr_res = np.zeros(mutual_info_ld.shape,dtype=dtype_dict)
for cc in range(mutual_info_hd.shape[0]):
    if cc % 500 == 0:
        print(cc,regr_res.shape[0])
    regr_res['brain_area'][cc] = mutual_info_hd['brain_area'][cc]
    regr_res['monkey'][cc] = mutual_info_hd['monkey'][cc]
    regr_res['session'][cc] = mutual_info_hd['session'][cc]
    regr_res['unit'][cc] = mutual_info_hd['neuron'][cc]
    regr_res['mutual_info_hd'][cc] = mutual_info_hd['mutual_info'][cc]
    regr_res['mutual_info_ld'][cc] = mutual_info_ld['mutual_info'][cc]
    regr_res['variable'][cc] = mutual_info_ld['variable'][cc]


    res_ = sts.linregress(tuning_ld['y_model'][cc],tuning_hd['y_model'][cc])
    regr_res['slope'][cc] = res_.slope
    regr_res['intercept'][cc] = res_.intercept
    regr_res['pval'][cc] = res_.pvalue
    regr_res['rsquare'][cc] = res_.rvalue
    regr_res['xmin'][cc] = tuning_ld['x'][cc][0]
    regr_res['xmax'][cc] = tuning_ld['x'][cc][-1]

regr_res['fdr_pval'] = fdrcorrection(regr_res['pval'],alpha=0.05)[1]
np.save('tuning_regression_res.npy',regr_res)
savemat('tuning_regression_res.mat',mdict={'regression':regr_res})

## plot results regression

df = pd.DataFrame(regr_res)
df = df[df['fdr_pval'] < 0.005]
df = df.rename(columns = {'slope':'gain'}, inplace = False)

plt.figure(figsize=(14,4))
ax = plt.subplot(111)
pnp = sbs.pointplot(x='variable',y='gain',hue='brain_area',order=order,hue_order=['MST','PPC','PFC'],data=df,
              dodge=0.2,palette={'MST':'g','PPC':'b','PFC':'r'},linestyles='none',ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
xlim = ax.get_xlim()
ax.plot(xlim,[1,1],'--k')
plt.xticks(rotation=90)
plt.tight_layout()

# ax.set_xlim(xlim)
plt.savefig('gain_tuning_density.pdf')

df = pd.DataFrame(regr_res)
df = df[df['fdr_pval'] < 0.005]
df = df.rename(columns = {'intercept':'mean rate delta [Hz]'}, inplace = False)
plt.figure(figsize=(14,4))
ax = plt.subplot(111)
sbs.pointplot(x='variable',y='mean rate delta [Hz]',hue='brain_area',order=order,hue_order=['MST','PPC','PFC'],data=df,
              dodge=0.2,palette={'MST':'g','PPC':'b','PFC':'r'},linestyles='none',ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
xlim = ax.get_xlim()
ax.plot(xlim,[0,0],'--k')
plt.xticks(rotation=90)
plt.tight_layout()

# ax.set_xlim(xlim)
plt.savefig('intercept_tuning_density.png')