#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:35:09 2022

@author: edoardo
"""
import numpy as np
import matplotlib.pylab as plt
import dill,os,re,sys
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import scipy.stats as sts

if os.path.exists('/scratch/jpn5/GAM_Repo'):
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    npz_path = '/scratch/jpn5/dataset_fiirefly/'
    base_fld = '/scratch/jpn5/GAM_fit_with_acc/'
    hist_length = 'long'

else:
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    npz_path = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/concatenation_with_accel/'
    base_fld = '/Volumes/WD_Edo 1/firefly_analysis/LFP_band/fit_longFilters/'
    hist_length = 'long'
from GAM_library import *
from data_handler import *
from gam_data_handlers import *

dict_type = {'names':
                 ('monkey',
                  'session',
                  'brain_area',
                  'unit_id',
                  'electrode_id',
                  'cluster_id',
                  'channel_id',
                  'manipulation_type',
                  'manipulation_value',
                  'is_significant',
                  'p_value',
                  'spike_hist_strength', 'inst_spike_hist_strength',
                  'filter_duration_ms',
                  'area_under_filter',
                  'log_det_cov',
                  'pseudo-r2',
                  'spike_hist_filter', 'spike_hist_mCI', 'spike_hist_pCI'),
             'formats':
                 ('U20',
                  'U20',
                  'U20',
                  int,
                  int,
                  int,
                  int,
                  'U30', float, bool, float,
                  float,
                  float, float, float, float, float, object, object, object)
             }

monkey_dict = {'m44': 'Quigley', 'm53': 'Schro', 'm91': 'Ody', 'm51': 'Bruno',
               'm72': 'Marco'}

for fld in os.listdir(base_fld):
    print(fld)
    if not fld.startswith('gam_'):
        continue
    session = fld.split('gam_')[1]

    fld_sess = base_fld + 'gam_%s/'%session


    dat = np.load(os.path.join(npz_path, session+'.npz'), allow_pickle=True)
    unit_info = dat['unit_info'].all()


    lst = os.listdir(fld_sess)
    info_sess = np.zeros(0,dtype=dict_type)
    for fhName in lst:
        if not re.match('^fit_results_m\d+s\d+_c\d+_all_1.0000.dill$',fhName):
            continue
        with open(os.path.join(fld_sess,fhName), 'rb') as fh:
            gam_res = dill.load(fh)
            full = gam_res['full']


            unit_info = dat['unit_info'].all()

        unit_id = int(fhName.split('_')[3].split('c')[1])

        info_neu = np.zeros(1,dtype=dict_type)
        info_neu['session'] = session
        info_neu['monkey'] = monkey_dict[session.split('s')[0]]
        info_neu['unit_id'] = unit_id
        info_neu['electrode_id'] = unit_info['electrode_id'][unit_id-1]
        info_neu['cluster_id'] = unit_info['cluster_id'][unit_id-1]
        info_neu['channel_id'] = unit_info['channel_id'][unit_id-1]
        info_neu['brain_area'] = unit_info['brain_area'][unit_id-1]

        info_neu['manipulation_type'] = 'all'
        info_neu['manipulation_value'] = True
        info_neu['pseudo-r2'] = gam_res['p_r2_coupling_full']
        ii = np.where(full.covariate_significance['covariate'] == 'spike_hist')[0]
        info_neu['p_value'] = full.covariate_significance[ii]['p-val']
        info_neu['is_significant'] = full.covariate_significance[ii]['p-val'] < 0.001
        cov_beta = full.cov_beta[full.index_dict['spike_hist'], :]
        cov_beta = cov_beta[:, full.index_dict['spike_hist']]
        eig = np.linalg.eigh(cov_beta)[0]
        info_neu['log_det_cov'] = np.log(eig).sum()
        if hist_length=='short':
            impulse = np.zeros(13)
            impulse[6] = 1
            fX,fXm,fXp = full.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
            info_neu['area_under_filter'] = 0.006 * (fX[7:11] - fX[0]).sum()

            fX = fX - fX[0]
            fX = fX[7:11]
            fXm = fXm[7:11]
            fXp = fXp[7:11]
        else:
            impulse = np.zeros(301)
            impulse[150] = 1
            fX,fXm,fXp = full.smooth_compute([impulse], 'spike_hist', perc=0.99, trial_idx=None)
            info_neu['area_under_filter'] = 0.006 * (fX[151:251] - fX[0]).sum()
            fX = fX - fX[0]
            fX = fX[151:251]
            fXm = fXm[151:251]
            fXp = fXp[151:251]

        info_neu['spike_hist_strength'] = np.linalg.norm(fX)
        info_neu['inst_spike_hist_strength'] = fX[0:3].mean()
        info_neu['filter_duration_ms'] = fX.shape[0] * 6
        info_neu['spike_hist_mCI'][0] = fXm
        info_neu['spike_hist_pCI'][0] = fXp
        info_neu['spike_hist_filter'][0] = fX

        info_sess = np.hstack((info_sess,info_neu))
    np.save('spike_hist_info_%s.npy'%session,info_sess)
    
first = True
for name in os.listdir('/Users/edoardo/Work/Code/GAM_code/analyzing/example_spike_hist/'):
    if not name.endswith('.npy'):
        continue
    session = name.split('_')[-1].split('.')[0]
    spkh = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/example_spike_hist/'+name,allow_pickle=True)
    if first:
        spkhist = deepcopy(spkh)
        first=False
    else:
        spkhist = np.hstack((spkhist, spkh))
from scipy.io import savemat

savemat('/Users/edoardo/Work/Code/GAM_code/analyzing/spike_hist_long.mat', mdict={'spkhist':spkhist})

