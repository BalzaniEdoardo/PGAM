#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:30:42 2021

@author: edoardo
"""
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
                       'coupling strength HD','coupling strength LD'),
            'formats':('U30','U30',int,int,'U30','U30',float,float)}

cs_table_pair = np.zeros(ld_pfc_cs.shape[0],dtype=cs_dtype)
cc = 0
for session in np.unique(ld_pfc_cs['session']):
    hd_cs_sess = hd_pfc_cs[hd_pfc_cs['session'] == session]
    ld_cs_sess = ld_pfc_cs[ld_pfc_cs['session'] == session]
    for row in ld_cs_sess:
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
        cc += 1




cs_table_pair = cs_table_pair[cs_table_pair['monkey']!='']

plt.figure()
ax=plt.subplot(111,aspect='equal')
bl = (cs_table_pair['sender brain area']=='PPC') & (cs_table_pair['receiver brain area']=='PPC') & (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)



model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])
plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='b',label='s: %.2f'%lnreg.coef_)

x = np.array([0,5])
y = x*lnreg.coef_
# plt.plot(x,y,color='b')
ax.set_xlim(0,5)
ax.set_ylim(0,5)



# # R-squared bisectrix
xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_
print('PPC',lnreg.score(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),cs_table_pair['coupling strength HD'][bl]))

print('START BAYESIAN REGR')
plt.plot([0,5],[0,5],'k')
basic_model = pm.Model()
with basic_model:
    # Priors for unknown model parameters
    # alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Expected value of outcome
    mu = beta[0] * xx + beta[1]

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=cs_table_pair['coupling strength HD'][bl])
    # map_estimate = pm.find_MAP(model=basic_model)
    # print(map_estimate)
    trace = pm.sample(5000, return_inferencedata=False, chains=1)
    # az.plot_trace(trace)
    # trace = pm.sample(500, return_inferencedata=False)
    XX = np.ones((xx.shape[0], 2))
    XX[:, 0] = xx
    XX = np.sort(XX, axis=0)
    # approx CI
    # intr = trace.get_values('alpha')
    bet = trace.get_values('beta')

    bet_ppc = deepcopy(bet[:, 0])
    CV = np.cov(np.c_[bet].T)
    se_y = np.sqrt(np.sum(np.dot(XX, CV) * XX, axis=1))
    norm = sts.norm()
    se_y = se_y * norm.ppf(1 - (1 - 0.95) * 0.5)
    beta_map = sts.mode(bet)[0]

    ax.plot(XX[:,0],np.squeeze(np.dot(XX, beta_map.T)),'b')
    ax.fill_between(XX[:,0], np.squeeze(np.dot(XX, beta_map.T)) - se_y,
                    np.squeeze(np.dot(XX, beta_map.T)) +se_y,color='b',alpha=0.4)

RSQ_ppc = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)
# # RSQ_bis = 1 - np.sum((xx-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)
# #
# #
# #
bl = (cs_table_pair['sender brain area']=='PFC') & (cs_table_pair['receiver brain area']=='PFC')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)
# plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='r')
model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])

plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='r',label='s: %.2f'%lnreg.coef_)

x = np.array([0,5])
y = x*lnreg.coef_
# plt.plot(x,y,color='r')
print('PFC',lnreg.score(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),cs_table_pair['coupling strength HD'][bl]))
xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_

RSQ_pfc = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    # alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=2)


    # Expected value of outcome
    mu =   beta[0] * xx + beta[1]

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=cs_table_pair['coupling strength HD'][bl])
    # map_estimate = pm.find_MAP(model=basic_model)
    # print(map_estimate)
    trace = pm.sample(5000, return_inferencedata=False,chains=1)
    # az.plot_trace(trace)
    # trace = pm.sample(500, return_inferencedata=False)
    XX = np.ones((xx.shape[0],2))
    XX[:,0] = xx
    XX = np.sort(XX, axis=0)
    # approx CI
    #intr = trace.get_values('alpha')
    bet = trace.get_values('beta')

    bet_pfc = deepcopy(bet[:,0])
    CV = np.cov(np.c_[ bet].T)
    se_y = np.sqrt(np.sum(np.dot(XX, CV) * XX, axis=1))
    norm = sts.norm()
    se_y = se_y * norm.ppf(1 - (1 - 0.95) * 0.5)
    beta_map = sts.mode(bet)[0]

    ax.plot(XX[:,0],np.squeeze(np.dot(XX, beta_map.T)),'r')
    ax.fill_between(XX[:,0], np.squeeze(np.dot(XX, beta_map.T)) - se_y,
                    np.squeeze(np.dot(XX, beta_map.T)) +se_y,color='r',alpha=0.4)

bl = (cs_table_pair['sender brain area']=='MST') & (cs_table_pair['receiver brain area']=='MST')& (cs_table_pair['coupling strength LD'] > 10**-3 ) & (cs_table_pair['coupling strength HD'] > 10**-3)
plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='g')

model = LinearRegression(fit_intercept=True)
lnreg = model.fit(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),
                   cs_table_pair['coupling strength HD'][bl])

print('MST',lnreg.score(cs_table_pair['coupling strength LD'][bl].reshape(bl.sum(),1),cs_table_pair['coupling strength HD'][bl]))
plt.scatter(cs_table_pair['coupling strength LD'][bl],cs_table_pair['coupling strength HD'][bl],s=10,alpha=0.4,color='g',label='s: %.2f'%lnreg.coef_)


x = np.array([0,5])
y = x*lnreg.coef_

# plt.plot(x,y,color='g')

xx = cs_table_pair['coupling strength LD'][bl]
yy = xx*lnreg.coef_+ lnreg.intercept_

RSQ_mst = 1 - np.sum((yy-cs_table_pair['coupling strength HD'][bl])**2) / np.sum((cs_table_pair['coupling strength HD'][bl] - np.mean(cs_table_pair['coupling strength HD'][bl]))**2)


basic_model = pm.Model()
with basic_model:

    # Priors for unknown model parameters
    # alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=2)


    # Expected value of outcome
    mu = beta[0] * xx + beta[1]

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=cs_table_pair['coupling strength HD'][bl])
    # map_estimate = pm.find_MAP(model=basic_model)
    # print(map_estimate)
    trace = pm.sample(5000, return_inferencedata=False,chains=1)
    # az.plot_trace(trace)
    # trace = pm.sample(500, return_inferencedata=False)
    XX = np.ones((xx.shape[0],2))
    XX[:,0] = xx
    XX = np.sort(XX, axis=0)
    # approx CI
    #intr = trace.get_values('alpha')
    bet = trace.get_values('beta')

    bet_mst = deepcopy(bet[:,0])
    CV = np.cov(np.c_[ bet].T)
    se_y = np.sqrt(np.sum(np.dot(XX, CV) * XX, axis=1))
    norm = sts.norm()
    se_y = se_y * norm.ppf(1 - (1 - 0.95) * 0.5)
    beta_map = sts.mode(bet)[0]

    ax.plot(XX[:,0],np.squeeze(np.dot(XX, beta_map.T)),'g')
    ax.fill_between(XX[:,0], np.squeeze(np.dot(XX, beta_map.T)) - se_y,
                    np.squeeze(np.dot(XX, beta_map.T)) +se_y,color='g',alpha=0.4)

plt.xlabel('coupling strength low density')
plt.ylabel('coupling strength high density')

#
# plt.plot(x,x,'k')
# # plt.figure()
# # plt.scatter(ld_pfc_cs,hd_pfc_cs)

gmm = GaussianMixture(n_components=3)
pfc_fit = gmm.fit(bet_pfc.reshape(bet_pfc.shape[0],1))

pdf_pfc = lambda x: (pfc_fit.weights_[0] * sts.norm(loc=pfc_fit.means_[0,0],scale=np.sqrt(pfc_fit.covariances_[0,0,0])).pdf(x)
                     + pfc_fit.weights_[1] * sts.norm(loc=pfc_fit.means_[1,0],scale=np.sqrt(pfc_fit.covariances_[1,0,0])).pdf(x)
                     + pfc_fit.weights_[2] * sts.norm(loc=pfc_fit.means_[2, 0], scale=np.sqrt(pfc_fit.covariances_[2, 0, 0])).pdf(x))


gmm = GaussianMixture(n_components=3)
ppc_fit = gmm.fit(bet_ppc.reshape(bet_ppc.shape[0],1))

pdf_ppc = lambda x: (ppc_fit.weights_[0] * sts.norm(loc=ppc_fit.means_[0,0],scale=np.sqrt(ppc_fit.covariances_[0,0,0])).pdf(x)
                     + ppc_fit.weights_[1] * sts.norm(loc=ppc_fit.means_[1,0],scale=np.sqrt(ppc_fit.covariances_[1,0,0])).pdf(x)
                     + ppc_fit.weights_[2] * sts.norm(loc=ppc_fit.means_[2, 0], scale=np.sqrt(ppc_fit.covariances_[2, 0, 0])).pdf(x))

gmm = GaussianMixture(n_components=3)
mst_fit = gmm.fit(bet_mst.reshape(bet_mst.shape[0],1))

pdf_mst = lambda x: (mst_fit.weights_[0] * sts.norm(loc=mst_fit.means_[0,0],scale=np.sqrt(mst_fit.covariances_[0,0,0])).pdf(x)
                     + mst_fit.weights_[1] * sts.norm(loc=mst_fit.means_[1,0],scale=np.sqrt(mst_fit.covariances_[1,0,0])).pdf(x)
                     + mst_fit.weights_[2] * sts.norm(loc=mst_fit.means_[2, 0], scale=np.sqrt(mst_fit.covariances_[2, 0, 0])).pdf(x))




xmin = min(bet_pfc.min(),bet_ppc.min(),bet_mst.min())
xmax = max(bet_pfc.max(),bet_ppc.max(),bet_mst.max())

xx = np.linspace(xmin,xmax,200)

plt.figure()
plt.plot(xx, pdf_mst(xx),'g')
plt.plot(xx, pdf_ppc(xx),'b')
plt.plot(xx, pdf_pfc(xx),'r')
plt.title('slope posterior distribution')
plt.xlabel('slope')
plt.ylabel('CDF')