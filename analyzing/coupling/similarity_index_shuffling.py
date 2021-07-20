import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
from statsmodels.distributions import ECDF
import os, sys, inspect
from scipy.io import savemat
thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
# extract:
#   1) correlation between filters x density condition
#   2) gain x condition
from statsmodels.stats.multitest import multipletests

def vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

dat = np.load('paired_coupling_flt.npz')
info = dat['info']
tunings = dat['tunings']
cond_list = dat['cond_list']


# select the desired subset of the data
cond_A_idx = 3
cond_B_idx = 4
seed = 4
filt = (info['is significant']) & (info['pseudo-r2']>0.01) & (info['monkey'] == 'Schro')
info_flt = info[filt]
tun_flt = tunings[filt]
tun_flt = tun_flt[:,:,:-2]
tun_condA = tun_flt[:,cond_A_idx,:]
tun_condB = tun_flt[:,cond_B_idx,:]
del tun_flt
np.random.seed(seed)
# loop over session
pairs_all = np.zeros((0,2),dtype=int)
corrs_all = np.zeros(0,dtype=float)
low_corr = np.zeros(0,dtype=float)
for session in np.unique(info_flt['session']):
    sel = np.where(info_flt['session'] == session)[0]
    pairs = np.zeros((sel.shape[0],2),dtype=int)
    tun_condA_sess = tun_condA[sel]
    tun_condB_sess = tun_condB[sel]
    choice_idx_A = np.arange(tun_condB_sess.shape[0],dtype=int)
    choice_idx_B = np.arange(tun_condB_sess.shape[0],dtype=int)

    rnd_choice_idx_A = np.arange(tun_condB_sess.shape[0], dtype=int)
    rnd_choice_idx_B = np.arange(tun_condB_sess.shape[0], dtype=int)
    
    corrs_sess = np.zeros(sel.shape[0])
    low_corr_sess = np.zeros(sel.shape[0])
    cc = 0
    while choice_idx_A.shape[0]:

        ii = np.random.choice(choice_idx_A)
        corrs = vcorrcoef(tun_condB_sess[choice_idx_B], tun_condA_sess[ii])
        amax = np.argmax(corrs)
        ii2 = choice_idx_B[amax]
        pairs[cc,:] = [sel[ii],sel[ii2]]
        choice_idx_A = choice_idx_A[choice_idx_A != ii]
        choice_idx_B = choice_idx_B[choice_idx_B != ii2]
        ii3 = np.random.choice(rnd_choice_idx_A)
        ii4 = np.random.choice(rnd_choice_idx_B)
        
        low_corr_sess[cc] = sts.pearsonr(tun_condB_sess[ii4], tun_condA_sess[ii3])[0]
        rnd_choice_idx_A = rnd_choice_idx_A[rnd_choice_idx_A != ii3]
        rnd_choice_idx_B = rnd_choice_idx_B[rnd_choice_idx_B != ii4]
        
        # print(choice_idx_A.shape[0],corrs[amax])
        corrs_sess[cc] = corrs[amax]
        cc += 1
    pairs_all = np.vstack((pairs_all,pairs))
    corrs_all = np.hstack((corrs_all,corrs_sess))
    low_corr = np.hstack((low_corr,low_corr_sess))

cdf_shuf = ECDF(corrs_all)
cdf_rnd = ECDF(low_corr)

plt.figure()
plt.plot(np.linspace(-1,1,300),cdf_shuf(np.linspace(-1,1,300)))

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

        if (not inf_ba['is sign odd 1'][cc]) or ( not inf_ba['is sign odd 0'][cc]):
            continue
        # if int(inf_ba[cc]['session'].split('s')[1].split('.')[0]) >= 50:
        #     continue
        # if (not inf_ba['is sign density 0.0050'][cc]) or ( not inf_ba['is sign density 0.0001'][cc]):
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

        # if int(inf_ba[cc]['session'].split('s')[1].split('.')[0]) >= 50:
        #     continue

        # if (not inf_ba['is sign density 0.0050'][cc]) or ( not inf_ba['is sign density 0.0001'][cc]):
        #     continue
        if (not inf_ba['is sign odd 1'][cc]) or ( not inf_ba['is sign odd 0'][cc]):
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

res_dict_cdf = {}
for ba in ['MST','PFC','PPC']:
    try:
        if ba =='MST':
            perc = 15
        else:
            perc = 15
        aa = dict_corr['MST'][dict_corr['MST'] >= np.nanpercentile(dict_corr['MST'],perc)]
        bb = dict_corr['PPC'][dict_corr['PPC'] >= np.nanpercentile(dict_corr['PPC'],perc)]
        cc = dict_corr['PFC'][dict_corr['PFC'] >= np.nanpercentile(dict_corr['PFC'],perc)]

        # cdf_corr = ECDF(np.hstack((aa,bb,cc)))

        cdf_slope = ECDF(dict_regr_slope[ba])
        cdf_intercept = ECDF(dict_regr_intercept[ba])

        cdf_corr_d = ECDF(dict_corr_d[ba])
        cdf_slope_d = ECDF(dict_regr_slope_d[ba])
        cdf_intercept_d = ECDF(dict_regr_intercept_d[ba])

        xmin_corr = np.nanpercentile(dict_corr[ba], 1)
        xmax_corr = np.nanpercentile(dict_corr[ba], 99)
        x_corr = np.linspace(-1, 1, 1000)

        xmin_slope = np.nanpercentile(dict_regr_slope[ba], 1)
        xmax_slope = np.nanpercentile(dict_regr_slope[ba], 99)
        x_slope = np.linspace(xmin_slope, xmax_slope, 1000)


        xmin_intercept = np.nanpercentile(dict_regr_intercept[ba], 1)
        xmax_intercept = np.nanpercentile(dict_regr_intercept[ba], 99)
        x_intercept = np.linspace(xmin_intercept, xmax_intercept, 1000)

        ax1.plot(x_corr, cdf_shuf(x_corr), color='k')
        ax1.plot(x_corr, cdf_rnd(x_corr), color=(0.25,)*3)
        ax2.plot(x_slope, cdf_slope(x_slope),color=ba_color[ba],ls='--')
        ax3.plot(x_intercept, cdf_intercept(x_intercept),color=ba_color[ba],ls='--')

        res_dict_cdf[ba] = {'cdf_x':x_corr,'cdf_y':cdf_corr_d(x_corr)}

        ax1.plot(x_corr, cdf_corr_d(x_corr), color=ba_color[ba],ls='-')
        ax2.plot(x_slope, cdf_slope_d(x_slope), color=ba_color[ba],ls='-')
        ax3.plot(x_intercept, cdf_intercept_d(x_intercept), color=ba_color[ba],ls='-')
    except:
        print('no example %s'%ba)

res_dict_cdf['bounds'] = {'cdf_x':x_corr,'upper':cdf_shuf(x_corr),'lower':cdf_rnd(x_corr)}

plt.tight_layout()
mdict = {'correlation_dict':dict_corr_d,'up_bound':cdf_shuf,'low_bound':cdf_rnd}
savemat('tuning_stability.mat',mdict=res_dict_cdf)

print(sts.kruskal(dict_corr_d['PPC'],dict_corr_d['PFC'],dict_corr_d['MST']))

pval_mst_pfc = sts.ttest_ind(dict_corr_d['PFC'],dict_corr_d['MST'])
pval_mst_ppc = sts.ttest_ind(dict_corr_d['PPC'],dict_corr_d['MST'])
pval_pfc_ppc = sts.ttest_ind(dict_corr_d['PFC'],dict_corr_d['PPC'])

hs_corr = multipletests(np.array([pval_mst_pfc[1],pval_mst_ppc[1],pval_pfc_ppc[1]]), alpha=0.05, method='hs', is_sorted=False, returnsorted=False)


print('Kruskall-Wallis test:',sts.kruskal(dict_corr_d['PPC'],dict_corr_d['PFC'],dict_corr_d['MST']))
print('Post-hoc comparrison holm-sidak corrected p-val: ')
print('PPC vs PFC', hs_corr[1][2])
print('PPC vs MST', hs_corr[1][1])
print('PFC vs MST', hs_corr[1][0])




# #
# # plt.figure()
# # for ba in ['MST','PFC','PPC']:
# #     sel = (info['sender brain area'] == ba) & (info['is significant']) & (info['pseudo-r2']>0.01)
# #     # sel2 = (info['sender brain area'] == ba) & (~info['is significant']) & (info['pseudo-r2']>0.01)
# #
# #     tun_ba = tunings[sel]
# #     inf_ba = info[sel]
# #     plt.hist(info[sel]['log |cov|'], label=ba, density=True,alpha=0.4)
# #     # plt.hist(info[sel2], label=ba +'non', density=True,alpha=0.4)
# #
# #     # break
# # plt.legend()
#
# #
# #
# # iidx = np.where(inf_ba['is sign density 0.0050'] & inf_ba['is sign density 0.0001'])[0]
# #
# # for cc in range(10):
# #     plt.figure(figsize=(12, 10))
# #
# #     for k in range(25):
# #
# #         plt.subplot(5,5,k+1)
# #         plt.plot(tun_ba[iidx[k+25*cc],cond_LD_idx,:-2 ],color='b')
# #         plt.plot(tun_ba[iidx[k+25*cc],cond_HD_idx,:-2 ],color='b')
# #         plt.plot(tun_ba[iidx[k + 25*cc], 1, :-2], color='g')
# #         plt.plot(tun_ba[iidx[k + 25*cc], 2, :-2], color='g')