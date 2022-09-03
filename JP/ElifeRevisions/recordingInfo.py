#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:37:01 2022

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import os,sys,re
import dill
# candidate units
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')
sys.path.append('/Users/edoardo/Work/Code/FF_dimReduction/Perturbation/PSTH/code/')
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sbs

import scipy.stats as sts
from scipy.io import loadmat
sys.path.append('/Users/edoardo/Work/Code/FF_dimReduction/process/')
from preprocClass import *
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as sts
from scipy.spatial.distance import cdist
import paramiko,scp

#%% loop over session and check the probetype x area

path = '/Volumes/Balsip HD/dataset_firefly'

dtype_dict = np.dtype([('session', 'U10'), ('num PFC', int),
                       ('num PPC', int),('num MST', int),
                       ('probe type PFC', 'U40'),
                       ('probe type PPC', 'U40'),('probe type MST', 'U40')])
table = np.zeros(0, dtype=dtype_dict)
#first = True
for fhname in os.listdir(path):
    if not re.match('^m\d+s\d+.mat$',fhname):
        continue
   
    
    session = fhname.split('.')[0]
    # print('session', session)
    # if session != 'm71s18' and ( first):
    #     continue
    # first=False
    print('session', session)
    dat = fireFly_dataPreproc(os.path.join(path, session+'.mat'))
    
    tmp = np.zeros(1, dtype=dtype_dict)
    tmp['session'] = session
    for area in ['PPC','PFC','MST']:
        tmp['num %s'%area] = np.sum(dat.spikes.brain_area==area)
        ptype = np.unique(dat.spikes.electrode_type[dat.spikes.brain_area==area])
        if len(ptype) == 0:
            ptype = ''
        elif len(ptype) == 1:
            ptype=ptype[0]
        else:
            ptype = 'unknown'
        tmp['probe type %s'%area] = ptype
    table = np.hstack((table,tmp))
np.save('/Users/edoardo/Work/Code/GAM_code/JP/ElifeRevisions/info_probe_session.npy',table)