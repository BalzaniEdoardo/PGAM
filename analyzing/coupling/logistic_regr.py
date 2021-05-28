#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:35:07 2021

@author: edoardo
"""
from copy import deepcopy
import sys
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
import numpy as np
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from gam_data_handlers import splineDesign
import scipy.stats as sts
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def pseudo_r2_compute(spk, family, modelX, params):#trans=False,vec=None,knots=None):
    
    if len(modelX.shape)==1:
        Y = deepcopy(modelX)
        mu = deepcopy(spk)
        spk = Y
    else:
        lin_pred = np.dot(modelX, params)
        mu = family.fitted(lin_pred)
    res_dev_t = family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t ** 2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, np.array([null_mu] * spk.shape[0]))

    null_deviance = np.sum(null_dev_t ** 2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2

moonkey='Schro'
keep_area = ['MST','PPC','PFC']


coupling_data = np.load('coupling_info.npy')

bruno_ppc_map = np.hstack(([np.nan],np.arange(1,9),[np.nan], np.arange(9,89), [np.nan], np.arange(89,97),[np.nan])).reshape((10,10))

consec_elect_dist_linear = 100
consec_elect_dist_utah = 400


electrode_map_dict = {
    'Schro': {'PPC': np.arange(1,49).reshape((8,6)), 'PFC': np.arange(49,97).reshape((8,6)),'MST':np.arange(1,25),'VIP':np.arange(1,25)},
    'Bruno': {'PPC': bruno_ppc_map},
    'Quigley':{'PPC':bruno_ppc_map,'MST':np.arange(1,25),'VIP':np.arange(1,25)}
    }

electrode_type_dict = {
    'Schro':{'MST':'linear','PPC':'utah','PFC':'utah','VIP':'linear'},
    'Quigley':{'PPC':'utah', 'MST':'linear','VIP':'linear'},
    'Marco':{'PFC':'linear'}
    
    }


def compute_dist(electrode_id_A,electrode_id_B,monkey,area):
    ele_type = electrode_type_dict[monkey][area]
    if ele_type == 'linear':
        distance = np.abs((electrode_id_A - electrode_id_B) * consec_elect_dist_linear)
    elif ele_type=='utah':
        
        x_pos_A,y_pos_A = np.where(electrode_map_dict[monkey][area] ==  electrode_id_A)
        x_pos_B,y_pos_B = np.where(electrode_map_dict[monkey][area] ==  electrode_id_B)
        
        distance = np.sqrt(((x_pos_A-x_pos_B)*consec_elect_dist_utah)**2 + ((y_pos_A-y_pos_B)*consec_elect_dist_utah)**2)
        
    return distance

sel = (coupling_data['monkey'] == 'Schro') * \
    (coupling_data['sender brain area'] == coupling_data['receiver brain area'])*\
        (coupling_data['manipulation type']=='all')

coupling_data = coupling_data[sel]

ba_sel = np.zeros(coupling_data.shape,dtype=bool) 
for ba in keep_area:
    ba_sel[coupling_data['sender brain area']==ba] = True

coupling_data = coupling_data[ba_sel]



Y = coupling_data['is significant']
X = np.zeros((Y.shape[0],4))
X[:, 0] = 1
X[coupling_data['sender brain area']=='MST', 1] = 1
X[coupling_data['sender brain area']=='PFC', 2] = 1
cc = 0
for row in coupling_data:
    X[cc, 3] = compute_dist(row['sender electrode id'],row['receiver electrode id'],row['monkey'],row['sender brain area'])
    cc+=1
    
true_dist = deepcopy(X[:, 3])
X[:, 3] = sts.zscore(X[:, 3])


# splint
all_tr = np.arange(X.shape[0],dtype=int)
train = all_tr[all_tr%10 != 0]
test = all_tr[all_tr%10 == 0]
model = Logit(Y[train],X[train])
logit_reg = model.fit()

pred = logit_reg.predict(X[test])
res = np.round(pred)

# GLM 
knots = np.linspace(np.min(X[:,3]),np.max(X[:,3]),5)
knots = np.hstack(([knots[0]]*3, knots,[knots[-1]]*3))
MX = splineDesign(knots, X[:,3], ord=4, der=0, outer_ok=False)
prep = (MX - MX.mean(axis=0))[:,:-1]

# fit reg
modelX = np.hstack((X[:,:-1], prep))
link = sm.families.links.logit()
family = sm.families.Binomial(link=link)
logit_spline = Logit(Y, modelX)
res_unreg = logit_spline.fit()


dist = np.linspace(np.min(X[:,3]),np.max(X[:,3])-0.0001,100)
zdist = (dist - np.mean(X[:,3]))/np.std(X[:,3])
MX_pred = splineDesign(knots, zdist, ord=4, der=0, outer_ok=False)
prep_pred = (MX_pred - MX.mean(axis=0))[:,:-1]

# mod ppc
modelPPC = np.zeros((100,3+prep_pred[1].shape[0]))
modelPPC[:,3:] = prep_pred
modelPPC[:,0] = 1

modelPFC = np.zeros((100,3+prep_pred[1].shape[0]))
modelPFC[:,3:] = prep_pred
modelPFC[:,0] = 1
modelPFC[:,2] = 1

modelMST = np.zeros((100,3+prep_pred[1].shape[0]))
modelMST[:,3:] = prep_pred
modelMST[:,0] = 1
modelMST[:,1] = 1


prob_ppc = res_unreg.predict(modelPPC)
prob_pfc = res_unreg.predict(modelPFC)
prob_mst = res_unreg.predict(modelMST)

import matplotlib.pylab as plt
srt = np.argsort(X[:,3])
pred_all = res_unreg.predict(modelX)
plt.figure()
ax = plt.subplot(111)
plt.plot(true_dist[srt][coupling_data[srt]['sender brain area']=='PPC'],
         pred_all[srt][coupling_data[srt]['sender brain area']=='PPC'],color='b',label='PPC')
plt.plot(true_dist[srt][coupling_data[srt]['sender brain area']=='MST'],
         pred_all[srt][coupling_data[srt]['sender brain area']=='MST'],color='g',label='MST')

plt.plot(true_dist[srt][coupling_data[srt]['sender brain area']=='PFC'],
         pred_all[srt][coupling_data[srt]['sender brain area']=='PFC'],color='r',label='PFC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('proba. connection')
plt.xlabel('$\mu$m')
plt.legend()
plt.savefig('logit_pred_prob.pdf')


logit_spline = Logit(Y[train], modelX[train])
fit_train = logit_spline.fit()
# print('test pr2',
#       pseudo_r2_compute(Y[test],family,modelX[test],fit_train.params))

# fit regul
# model = LogisticRegression(penalty='elasticnet',
#                             fit_intercept=False,solver='saga',l1_ratio=0.2)
#
# best_pr2 = -1
# func = lambda X,Y: pseudo_r2_compute(Y,family,X,fit_train.params)
# score = make_scorer(func, greater_is_better=True, needs_proba=True)
#
# parm_dict = {'C':np.logspace(-4,2,10),'l1_ratio':np.linspace(0,1,5)}
# clf = GridSearchCV(model, parm_dict,scoring=score)
# clf.fit(modelX, Y)
#
# result = clf.best_estimator_.fit(modelX, Y)
#
#

# coupling strength
Y = coupling_data['coupling strength']
X = np.zeros((Y.shape[0], 4))
X[:, 0] = 1
X[coupling_data['sender brain area'] == 'MST', 1] = 1
X[coupling_data['sender brain area'] == 'PFC', 2] = 1
cc = 0
for row in coupling_data:
    X[cc, 3] = compute_dist(row['sender electrode id'], row['receiver electrode id'], row['monkey'],
                            row['sender brain area'])
    cc += 1

true_dist = deepcopy(X[:, 3])
mean_dist = X[:,3].mean()
std_dist = X[:,3].std()

X[:, 3] = sts.zscore(X[:, 3])

# splint
all_tr = np.arange(X.shape[0], dtype=int)
train = all_tr[all_tr % 10 != 0]
test = all_tr[all_tr % 10 == 0]
model = OLS(Y[train], X[train])
logit_reg = model.fit()

pred = logit_reg.predict(X[test])
res = np.round(pred)

# GLM
knots = np.linspace(np.min(X[:, 3]), np.max(X[:, 3]), 5)
knots = np.hstack(([knots[0]] * 3, knots, [knots[-1]] * 3))
MX = splineDesign(knots, X[:, 3], ord=4, der=0, outer_ok=False)
prep = (MX - MX.mean(axis=0))[:, :-1]

# fit reg
modelX = np.hstack((X[:, :-1], prep))
link = sm.families.links.logit()
family = sm.families.Binomial(link=link)
logit_spline = OLS(Y, modelX)
res_unreg = logit_spline.fit()

dist = np.linspace(np.min(X[:, 3]), np.max(X[:, 3]) - 0.0001, 100)
zdist = (dist - np.mean(X[:, 3])) / np.std(X[:, 3])
MX_pred = splineDesign(knots, zdist, ord=4, der=0, outer_ok=False)
prep_pred = (MX_pred - MX.mean(axis=0))[:, :-1]

# mod ppc
modelPPC = np.zeros((100, 3 + prep_pred[1].shape[0]))
modelPPC[:, 3:] = prep_pred
modelPPC[:, 0] = 1

modelPFC = np.zeros((100, 3 + prep_pred[1].shape[0]))
modelPFC[:, 3:] = prep_pred
modelPFC[:, 0] = 1
modelPFC[:, 2] = 1

modelMST = np.zeros((100, 3 + prep_pred[1].shape[0]))
modelMST[:, 3:] = prep_pred
modelMST[:, 0] = 1
modelMST[:, 1] = 1

prob_ppc = res_unreg.predict(modelPPC)
prob_pfc = res_unreg.predict(modelPFC)
prob_mst = res_unreg.predict(modelMST)

import matplotlib.pylab as plt

srt = np.argsort(X[:, 3])
pred_all = res_unreg.predict(modelX)
plt.figure()
ax = plt.subplot(111)
plt.plot(true_dist[srt][coupling_data[srt]['sender brain area'] == 'PPC'],
         pred_all[srt][coupling_data[srt]['sender brain area'] == 'PPC'], color='b', label='PPC')
plt.plot(true_dist[srt][coupling_data[srt]['sender brain area'] == 'MST'],
         pred_all[srt][coupling_data[srt]['sender brain area'] == 'MST'], color='g', label='MST')

plt.plot(true_dist[srt][coupling_data[srt]['sender brain area'] == 'PFC'],
         pred_all[srt][coupling_data[srt]['sender brain area'] == 'PFC'], color='r', label='PFC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('coupling strength')
plt.xlabel('$\mu$m')
plt.legend()
plt.savefig('ols_pred_prob.pdf')

logit_spline = OLS(Y[train], modelX[train])
fit_train = logit_spline.fit()