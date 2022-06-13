#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:04:43 2022

@author: edoardo
"""

import numpy as np
import matplotlib.pylab as plt
import dill,os,re
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import scipy.stats as sts
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import seaborn as sbs

order = ['rad_vel','ang_vel','rad_acc','ang_acc','t_move','t_stop','t_flyOFF','rad_target',
         'ang_target','rad_path','ang_path','t_reward','eye_vert','eye_hori']

regr_res = np.load('/Users/edoardo/Dropbox/NoelBalzani_breifComm/Gain modulation summary stats/tuning_regression_LDwrtHD.npy',allow_pickle=True)
df = pd.DataFrame(regr_res)
df = df[df['fdr_pval'] < 0.005]
df = df.rename(columns = {'slope':'gain'}, inplace = False)

plt.figure(figsize=(14,4))
ax = plt.subplot(111)
sbs.pointplot(x='variable',y='gain',hue='brain_area',order=order,hue_order=['MST','PPC','PFC'],data=df,
              dodge=0.2,palette={'MST':'g','PPC':'b','PFC':'r'},linestyles='none',ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
xlim = ax.get_xlim()
ax.plot(xlim,[1,1],'--k')
plt.xticks(rotation=90)
plt.tight_layout()