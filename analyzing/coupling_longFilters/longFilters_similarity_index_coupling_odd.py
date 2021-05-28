import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
from statsmodels.distributions import ECDF
import os, sys, inspect
thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
# extract:
#   1) correlation between filters x density condition
#   2) gain x condition


dat = np.load('paired_coupling_flt_long.npz')
info = dat['info']
tunings = dat['tunings']
cond_list = dat['cond_list']


cond_HD_idx = 2
cond_LD_idx = 1
print('Odd cond', cond_list[cond_HD_idx])
print('Even cond', cond_list[cond_LD_idx])


dict_corr = {}
dict_regr_slope = {}
dict_regr_intercept = {}
plt_first = 10
pltnum = 0
for ba in ['MST','PFC','PPC']:
    sel = (info['sender brain area'] == ba) & (info['is significant']) & (info['pseudo-r2']>0.01) & (info['monkey'] == 'Schro')
    tun_ba = tunings[sel]
    inf_ba = info[sel]
    dict_corr[ba] = np.zeros(inf_ba.shape[0],dtype=float) * np.nan
    dict_regr_slope[ba] = np.zeros(inf_ba.shape[0], dtype=float) * np.nan
    dict_regr_intercept[ba] = np.zeros(inf_ba.shape[0], dtype=float)*np.nan

    for cc in range(inf_ba.shape[0]):
        # if inf_ba['coupling strength'][cc] < np.nanpercentile(inf_ba['coupling strength'],80):
        #     continue

        # if inf_ba['coupling strength'][cc] < np.nanpercentile(inf_ba['coupling strength'],80):
        #     continue
        if (not inf_ba['is sign density 0.0050'][cc]) and ( not inf_ba['is sign density 0.0001'][cc]):
            continue
        # if (not inf_ba['is sign odd 1'][cc]) and ( not inf_ba['is sign odd 0'][cc]):
        #     continue
        # if (not inf_ba['is significant'][cc]):
        #     continue
        dict_corr[ba][cc] = sts.pearsonr(tun_ba[cc, cond_LD_idx,:-2],tun_ba[cc, cond_HD_idx,:-2])[0]


        lreg = sts.linregress(tun_ba[cc, cond_LD_idx,:-2],tun_ba[cc, cond_HD_idx,:-2])
        dict_regr_slope[ba][cc] = lreg.slope
        dict_regr_intercept[ba][cc] = lreg.intercept

        # if ba =='PPC' and dict_corr[ba][cc]<-0.9 and pltnum<plt_first:
        #     plt.figure()
        #     plt.title('%d->%d, %s'% (inf_ba['sender unit id'][cc],inf_ba['receiver unit id'][cc],inf_ba['session'][cc]))
        #     plt.plot(tun_ba[cc, cond_LD_idx,:])
        #     plt.plot(tun_ba[cc, cond_HD_idx,:])
        #     pltnum += 1

for ba in ['MST','PFC','PPC']:
    dict_regr_slope[ba] = dict_regr_slope[ba][~np.isnan(dict_regr_slope[ba])]
    dict_regr_intercept[ba] = dict_regr_intercept[ba][~np.isnan(dict_regr_intercept[ba])]
    dict_corr[ba] = dict_corr[ba][~np.isnan(dict_corr[ba])]



cond_HD_idx = 4
cond_LD_idx = 3
print('High dens', cond_list[cond_HD_idx])
print('Low dens', cond_list[cond_LD_idx])

dict_corr_d = {}
dict_regr_slope_d = {}
dict_regr_intercept_d = {}
dict_regr_slope_rev = {}
plt_first = 10
pltnum = 0
for ba in ['MST','PFC','PPC']:
    sel = (info['sender brain area'] == ba) & (info['is significant']) & (info['pseudo-r2']>0.01) & (info['monkey'] == 'Schro')
    tun_ba = tunings[sel]
    inf_ba = info[sel]
    dict_corr_d[ba] = np.zeros(inf_ba.shape[0],dtype=float) * np.nan
    dict_regr_slope_d[ba] = np.zeros(inf_ba.shape[0], dtype=float) * np.nan
    dict_regr_intercept_d[ba] = np.zeros(inf_ba.shape[0], dtype=float)*np.nan
    dict_regr_slope_rev[ba]= np.zeros(inf_ba.shape[0], dtype=float)*np.nan
    for cc in range(inf_ba.shape[0]):
        # if inf_ba['coupling strength'][cc] < np.nanpercentile(inf_ba['coupling strength'],80):
        #     continue
        if (not inf_ba['is sign density 0.0050'][cc]) and ( not inf_ba['is sign density 0.0001'][cc]):
            continue
        # if (not inf_ba['is significant'][cc]):
        #     continue
        dict_corr_d[ba][cc] = sts.pearsonr(tun_ba[cc, cond_LD_idx,:-2],tun_ba[cc, cond_HD_idx,:-2])[0]


        lreg = sts.linregress(tun_ba[cc, cond_LD_idx,:-2],tun_ba[cc, cond_HD_idx,:-2])
        dict_regr_slope_d[ba][cc] = lreg.slope
        dict_regr_intercept_d[ba][cc] = lreg.intercept

        lreg = sts.linregress(tun_ba[cc, cond_HD_idx, :-2], tun_ba[cc, cond_LD_idx, :-2])
        dict_regr_slope_rev[ba][cc] = lreg.slope

        # if ba =='PPC' and dict_corr[ba][cc]<-0.9 and pltnum<plt_first:
        #     plt.figure()
        #     plt.title('%d->%d, %s'% (inf_ba['sender unit id'][cc],inf_ba['receiver unit id'][cc],inf_ba['session'][cc]))
        #     plt.plot(tun_ba[cc, cond_LD_idx,:])
        #     plt.plot(tun_ba[cc, cond_HD_idx,:])
        #     pltnum += 1

for ba in ['MST','PFC','PPC']:
    dict_regr_slope_d[ba] = dict_regr_slope_d[ba][~np.isnan(dict_regr_slope_d[ba])]
    dict_regr_intercept_d[ba] = dict_regr_intercept_d[ba][~np.isnan(dict_regr_intercept_d[ba])]
    dict_corr_d[ba] = dict_corr_d[ba][~np.isnan(dict_corr_d[ba])]
    dict_regr_slope_rev[ba] = dict_regr_slope_rev[ba][~np.isnan(dict_regr_slope_rev[ba])]


plt.figure(figsize=(10,4))
plt.suptitle('coupling with odd')
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

ax1.set_title('filter corr')
ax2.set_title('filter gain')
ax3.set_title('filter intercept')

ax1.set_xlabel('correlation')
ax2.set_xlabel('gain')
ax3.set_xlabel('intercept')

ax1.set_ylabel('CDF')

ba_color = {'PFC':'r','PPC':'b','MST':'g'}
for ba in ['MST','PFC','PPC']:
    cdf_corr = ECDF(dict_corr[ba])
    cdf_slope = ECDF(dict_regr_slope[ba])
    cdf_intercept = ECDF(dict_regr_intercept[ba])

    cdf_corr_d = ECDF(dict_corr_d[ba])
    cdf_slope_d = ECDF(dict_regr_slope_d[ba])
    cdf_intercept_d = ECDF(dict_regr_intercept_d[ba])

    xmin_corr = np.nanpercentile(dict_corr[ba], 1)
    xmax_corr = np.nanpercentile(dict_corr[ba], 99)
    x_corr = np.linspace(xmin_corr, xmax_corr, 1000)

    xmin_slope = np.nanpercentile(dict_regr_slope[ba], 1)
    xmax_slope = np.nanpercentile(dict_regr_slope[ba], 99)
    x_slope = np.linspace(xmin_slope, xmax_slope, 1000)


    xmin_intercept = np.nanpercentile(dict_regr_intercept[ba], 1)
    xmax_intercept = np.nanpercentile(dict_regr_intercept[ba], 99)
    x_intercept = np.linspace(xmin_intercept, xmax_intercept, 1000)

    ax1.plot(x_corr, cdf_corr(x_corr),color=ba_color[ba],ls='--')
    ax2.plot(x_slope, cdf_slope(x_slope),color=ba_color[ba],ls='--')
    ax3.plot(x_intercept, cdf_intercept(x_intercept),color=ba_color[ba],ls='--')

    ax1.plot(x_corr, cdf_corr_d(x_corr), color=ba_color[ba],ls='-')
    ax2.plot(x_slope, cdf_slope_d(x_slope), color=ba_color[ba],ls='-')
    ax3.plot(x_intercept, cdf_intercept_d(x_intercept), color=ba_color[ba],ls='-')
plt.tight_layout()



plt.figure()
for ba in ['MST','PFC','PPC']:
    sel = (info['sender brain area'] == ba) & (info['is significant']) & (info['pseudo-r2']>0.01)
    # sel2 = (info['sender brain area'] == ba) & (~info['is significant']) & (info['pseudo-r2']>0.01)

    tun_ba = tunings[sel]
    inf_ba = info[sel]
    plt.hist(info[sel]['log |cov|'], label=ba, density=True,alpha=0.4)
    # plt.hist(info[sel2], label=ba +'non', density=True,alpha=0.4)

    # break
plt.legend()
sel = (info['sender brain area'] == 'MST') & (info['is significant']) & (info['pseudo-r2']>0.01) & (info['monkey'] == 'Schro')
tun_ba = tunings[sel]
inf_ba = info[sel]
iidx = np.where(inf_ba['is sign density 0.0050'] & inf_ba['is sign density 0.0001'])[0]

for cc in range(10):
    plt.figure(figsize=(12, 10))

    for k in range(25):

        plt.subplot(5,5,k+1)
        plt.plot(tun_ba[iidx[k+25*cc],cond_LD_idx,:-2 ],color='b')
        plt.plot(tun_ba[iidx[k+25*cc],cond_HD_idx,:-2 ],color='b')
        plt.plot(tun_ba[iidx[k + 25*cc], 1, :-2], color='g')
        plt.plot(tun_ba[iidx[k + 25*cc], 2, :-2], color='g')