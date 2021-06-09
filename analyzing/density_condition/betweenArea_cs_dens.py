import pymc3 as pm
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as sts
import pandas as pd
from copy import deepcopy
from statsmodels.distributions import ECDF
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import arviz as az
coupl_info = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_info.npy')
from scipy.io import savemat

# sel monkey
coupl_info = coupl_info[(coupl_info['pseudo-r2']>=0.02)]
coupl_info = coupl_info[(coupl_info['monkey']=='Schro')]


def cramers_corrected_stat(x,y):

    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = sts.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return chi2, round(result,6)

# table_stat = np.zeros(len(np.unique(mutual_info['variable'])),
#                       dtype={'names':('variable','MST num','PPC num','PFC num',
#                                       'MST freq sign','PPC freq sign','PFC freq sign','Chi2-stat',
#                                       'p-val','Cramer-V','effect-size'),
#                       'formats':('U30',int,int,int,
#                                       float,float,float,float,
#                                       float,float,'U30')})



label_coupling = []#pd.Series(np.hstack((mst_vec,ppc_vec,pfc_vec)))
bl_label = []#pd.Series(np.hstack((mst_bl,ppc_bl,pfc_bl)))


sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('MST->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('MST->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('PPC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('PFC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['PFC->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PPC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PFC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PFC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


label_coupling = pd.Series(label_coupling)
bl_label = pd.Series(bl_label)

cross_tab = pd.crosstab(label_coupling,bl_label)
print(cramers_corrected_stat(label_coupling,bl_label))


print('\nhigh density\n')
sel = (coupl_info['manipulation value']== 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('MST->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))

sel = (coupl_info['manipulation value']==  0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('MST->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['MST->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation value']==  0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('PPC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])



label_coupling = np.hstack((label_coupling,['PPC->PFC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))





sel = (coupl_info['manipulation value']==  0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('PFC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
label_coupling = np.hstack((label_coupling,['PFC->PPC']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))



sel = (coupl_info['manipulation value']==  0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PPC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

label_coupling = np.hstack((label_coupling,['PPC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


sel = (coupl_info['manipulation value']== 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('PFC->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])




label_coupling = np.hstack((label_coupling,['PFC->MST']*betw_coupl.shape[0]))
bl_label = np.hstack((bl_label, betw_coupl['p-value']<0.001))


label_coupling = pd.Series(label_coupling)
bl_label = pd.Series(bl_label)

cross_tab = pd.crosstab(label_coupling,bl_label)
print(cramers_corrected_stat(label_coupling,bl_label))




print('\nWhithin area')
sel = (coupl_info['manipulation value']== 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('HD PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PPC') * (coupl_info['receiver brain area'] == 'PPC')
betw_coupl = coupl_info[sel]
print('LD PPC->PPC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation value']== 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('HD MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])
sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'MST') * (coupl_info['receiver brain area'] == 'MST')
betw_coupl = coupl_info[sel]
print('LD MST->MST', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])


sel = (coupl_info['manipulation value']== 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('HD PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])

sel = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
betw_coupl = coupl_info[sel]
print('LD PFC->PFC', (betw_coupl['p-value']<0.001).sum()/betw_coupl.shape[0])




sel_ld = (coupl_info['manipulation value']== 0.0001) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')
sel_hd = (coupl_info['manipulation value']== 0.005) * (coupl_info['manipulation type'] == 'density') * (coupl_info['sender brain area'] == 'PFC') * (coupl_info['receiver brain area'] == 'PFC')

ld_pfc_cs = coupl_info[(coupl_info['manipulation type'] == 'density')&(coupl_info['manipulation value']== 0.0001)]
hd_pfc_cs = coupl_info[(coupl_info['manipulation type'] == 'density')&(coupl_info['manipulation value']== 0.005)]


cs_dtype = {'names': ('monkey','session','sender unit id','receiver unit id',
                      'sender brain area', 'receiver brain area',
                       'coupling strength HD','coupling strength LD','significance HD','significance LD'),
            'formats':('U30','U30',int,int,'U30','U30',float,float,bool,bool)}

cs_table_pair = np.zeros(ld_pfc_cs.shape[0],dtype=cs_dtype)
cc = 0
for session in np.unique(ld_pfc_cs['session']):
    hd_cs_sess = hd_pfc_cs[hd_pfc_cs['session'] == session]
    ld_cs_sess = ld_pfc_cs[ld_pfc_cs['session'] == session]
    for row in ld_cs_sess:
        if (row['sender electrode id'] == row['receiver electrode id']) & \
            (row['sender brain area'] == row['receiver brain area']):
            continue
        sender = row['sender unit id']
        receiver = row['receiver unit id']
        bl = (hd_cs_sess['sender unit id'] == sender) & (hd_cs_sess['receiver unit id'] == receiver)

        bl = bl & (hd_cs_sess['sender brain area'] == row['sender brain area'])
        bl = bl & (hd_cs_sess['receiver brain area'] == row['receiver brain area'])
        SM = np.sum(bl)
        assert(SM<=1)
        if SM==0:
            continue
        row_hd = np.squeeze(hd_cs_sess[bl])

        cs_table_pair['monkey'][cc] = row_hd['monkey']
        cs_table_pair['session'][cc] = row_hd['session']
        cs_table_pair['sender unit id'][cc] = sender
        cs_table_pair['receiver unit id'][cc] = receiver
        cs_table_pair['coupling strength HD'][cc] = row_hd['coupling strength']
        cs_table_pair['coupling strength LD'][cc] = row['coupling strength']
        cs_table_pair['sender brain area'][cc] = row_hd['sender brain area']
        cs_table_pair['receiver brain area'][cc] = row['receiver brain area']
        cs_table_pair['significance LD'][cc] = row['is significant']
        cs_table_pair['significance HD'][cc] = row_hd['is significant']


        cc += 1




cs_table_pair = cs_table_pair[cs_table_pair['monkey']!='']


bl = (cs_table_pair['sender brain area']=='MST') & (cs_table_pair['receiver brain area']=='PFC')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])

slope_mst_pfc = lnreg.coef_
plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='r',label='s: %.2f'%lnreg.coef_)
plt.legend()

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_
srt_idx = np.argsort(xx)
plt.plot(xx[srt_idx],yy[srt_idx],'r')

RSQ_mst_to_pfc = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)

### MST to PPC
bl = (cs_table_pair['sender brain area']=='MST') & (cs_table_pair['receiver brain area']=='PPC')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])
slope_mst_ppc = lnreg.coef_
plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='b',label='s: %.2f'%lnreg.coef_)
plt.legend()

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_

RSQ_mst_to_ppc = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)
srt_idx = np.argsort(xx)
plt.plot(xx[srt_idx],yy[srt_idx],'b')

plt.title('Sender MST')

####

# cs_table_pair = cs_table_pair[cs_table_pair['monkey']!='']

plt.figure()
bl = (cs_table_pair['sender brain area']=='PFC') & (cs_table_pair['receiver brain area']=='MST')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])
slope_pfc_mst = lnreg.coef_
plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='g',label='s: %.2f'%lnreg.coef_)
plt.legend()

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_
srt_idx = np.argsort(xx)
plt.plot(xx[srt_idx],yy[srt_idx],'g')

RSQ_pfc_to_mst = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)


### PFC to PPC
bl = (cs_table_pair['sender brain area']=='PFC') & (cs_table_pair['receiver brain area']=='PPC')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])
slope_pfc_ppc = lnreg.coef_

plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='b',label='s: %.2f'%lnreg.coef_)
plt.legend()

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_

RSQ_pfc_to_ppc = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)
srt_idx = np.argsort(xx)
plt.plot(xx[srt_idx],yy[srt_idx],'b')

plt.title('Sender PFC')


plt.figure()
bl = (cs_table_pair['sender brain area']=='PPC') & (cs_table_pair['receiver brain area']=='PFC')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])
slope_ppc_pfc = lnreg.coef_

plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='r',label='s: %.2f'%lnreg.coef_)
plt.legend()

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_
srt_idx = np.argsort(xx)
plt.plot(xx[srt_idx],yy[srt_idx],'r')

RSQ_ppc_to_pfc = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)


### PFC to PPC
bl = (cs_table_pair['sender brain area']=='PPC') & (cs_table_pair['receiver brain area']=='MST')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])
slope_ppc_mst = lnreg.coef_

plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='g',label='s: %.2f'%lnreg.coef_)
plt.legend()

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_

RSQ_ppc_to_mst = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)
srt_idx = np.argsort(xx)
plt.plot(xx[srt_idx],yy[srt_idx],'g')

plt.title('Sender PPC')
print('slopes')
print('MST->PFC',slope_mst_pfc,RSQ_mst_to_pfc)
print('MST->PPC',slope_mst_ppc,RSQ_mst_to_ppc)
print('PFC->MST',slope_pfc_mst,RSQ_pfc_to_mst)
print('PFC->PPC',slope_pfc_ppc,RSQ_pfc_to_ppc)
print('PPC->MST',slope_ppc_mst,RSQ_ppc_to_mst)
print('PPC->PFC',slope_ppc_pfc,RSQ_ppc_to_pfc)


dd = {'names':('pair','coupling strength','density','significance'),'formats':('U30',float,'U30',bool)}
tab = np.zeros(0,dtype=dd)

for area1 in ['MST','PPC','PFC']:
    for area2 in ['MST','PPC','PFC']:
        # if area2 == area1:
        #     continue
        bl = (cs_table_pair['sender brain area']==area1) & (cs_table_pair['receiver brain area']==area2)& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)
        tmp = np.zeros(bl.sum()*2,dtype=dd)
        tmp['pair'] = '%s->%s'%(area1,area2)
        tmp['coupling strength'][:bl.sum()] = cs_table_pair[bl]['coupling strength LD']
        tmp['coupling strength'][bl.sum():] = cs_table_pair[bl]['coupling strength HD']

        tmp['density'][:bl.sum()] = 'Low Density'
        tmp['density'][bl.sum():] = 'High Density'
        tmp['significance'][:bl.sum()] = cs_table_pair[bl]['significance LD']
        tmp['significance'][bl.sum():] = cs_table_pair[bl]['significance HD']

        tab = np.hstack((tab,tmp))

import pandas as pd
import seaborn as sbn

df = pd.DataFrame(tab)
plt.figure()
sbn.pointplot(x='pair',y='coupling strength',hue='density',data=df,dodge=0.2,linestyles='none')



plt.figure(figsize=(11,4))
ddf = df[(df['pair'] != 'MST->MST') & (df['pair'] != 'PPC->PPC') & (df['pair'] != 'PFC->PFC')]
ax = plt.subplot(111)
sbn.pointplot(x='pair',y='significance', hue='density', data=ddf,dodge=0.2,linestyles='none',ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('fraction coupled')
ax.set_xlabel('brain regions')
plt.savefig('pointplot_between_area_fracCoupl.pdf')

plt.figure(figsize=(8,4))
ddf = df[(df['pair'] == 'MST->MST') | (df['pair'] == 'PPC->PPC') | (df['pair'] == 'PFC->PFC')]
ax = plt.subplot(111)
sbn.pointplot(x='pair',y='significance', hue='density',order=['MST->MST','PPC->PPC','PFC->PFC'], data=ddf,dodge=0.2,linestyles='none',ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel('fraction coupled')
ax.set_xlabel('brain regions')
plt.savefig('pointplot_within_area_fracCoupl.pdf')