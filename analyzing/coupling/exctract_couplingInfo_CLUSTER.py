#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:18:23 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
# thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
# sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
# sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
# sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')

if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
    cluster = False
    JOB = -1
else:
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library/')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc/')
    cluster = True
    JOB = int(sys.argv[1])

from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
from scipy.io import loadmat
import numpy as np
#import matplotlib.pylab as plt
import statsmodels.api as sm
import dill
import pandas as pd
import scipy.stats as sts
import scipy.linalg as linalg
from time import perf_counter
#from seaborn import heatmap
from path_class import get_paths_class
from knots_constructor import knots_cerate
from copy import deepcopy
from numba import njit
from bisect import bisect_left,bisect_right

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None


def unpack_name(name):
    session = re.findall('m\d+s\d+',name)[0]
    unitID = int(re.findall('_c\d+_',name)[0].split('_')[1].split('c')[1])
    man_type = re.findall('_c\d+_[a-z]+_',name)[0].split('_')[2]
    man_val = float(re.findall('\d+.\d\d\d\d',name)[0])
    return session,unitID,man_type,man_val


if not cluster:
    path_to_gam = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'
    npz_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

else:
    path_to_gam = '/scratch/jpn5/mutual_info/'
    npz_path = '/scratch/jpn5/dataset_firefly/'

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno',
                   'm72':'Marco'}

path_gen = get_paths_class()


cond_type = 'all'
cond_value = 1

dict_type = {'names':
             ('monkey','session', 
              'sender brain area', 'receiver brain area',
              'sender unit id', 'receiver unit id', 
              'sender electrode id', 'receiver electrode id',
              'sender cluster id','receiver cluster id',
              'sender channel id', 'receiver channel id',
              'manipulation type','manipulation value',
              'is significant','p-value',
              'coupling strength','instantaneous coupling strength',
              'filter duration (ms)','area under filter','filter dur [ms]','log |cov|','pseudo-r2'),
             'formats':
                 ('U20','U20',
                  'U20','U20',
                  int,int,
                  int,int,
                  int,int,
                  int,int,
                  'U30',float,bool,float,
                  float,float,
                  float,float,float,float,float)
             }


# build session list
sess_list = []
for root, dirs, files in os.walk(path_to_gam):
    print(root)
    if not re.match('^.+/gam_m\d+s\d+$', root):
        continue
    session = os.path.basename(root).split('_')[1]
    if session != 'm53s130':
        sess_list += [session]

print(len(sess_list))


pattern = '^fit_results_m\d+s\d+_c\d+_[a-z]+_\d+.\d\d\d\d.dill$'

info_coupling = np.zeros(0,dtype=dict_type)
done_id = ['']*info_coupling.shape[0]

done_id = np.sort(np.array(done_id))
done_id_min = []
for name in done_id:
    try:
        if done_id_min[-1]!=name:
            done_id_min += [name]
    except IndexError:
         print('err')  
         done_id_min += [name]                     
done_id = done_id_min


orig_done = deepcopy(info_coupling)
previous_sess  = ''
impulse = np.zeros(201)
impulse[100] = 1
extr_first=False
skp_unitl = False
session = sess_list[JOB]
for name in os.listdir(os.path.join(path_to_gam,'gam_%s'%session)):



    if not re.match(pattern, name):
        continue
    session,receiver_ID,man_type,man_val = unpack_name(name)
    item = '%s_%d_%s_%s'%(session,receiver_ID,man_type,man_val)
    # blchk = ((orig_done['receiver unit id'] == receiver_ID) &
    #          (orig_done['session'] == session) &
    #          (orig_done['manipulation type'] == man_type) &
    #          (orig_done['manipulation value'] == man_val))
    # idxLast = bisect_right(done_id,item)
    idxFirst = bisect_left(done_id,item)
    # assert(check_done <= 1)
    # if check_done >= 1:
    #     print('skip')
        # continue
    if len(done_id) == 0:
        pass
    else:
        try:
            if done_id[idxFirst] == item:
                # rmv_index = np.ones(done_id.shape,dtype=bool)
                # rmv_index[idxFirst:idxLast] = False
                # done_id = done_id[rmv_index]
                print('skip')
                continue
        except IndexError:
            pass
    try:
    # open fits
        with open(os.path.join(path_to_gam,'gam_%s'%session, name), 'rb') as fh:
            gam_res = dill.load(fh)
            full = gam_res['full']
            reduced = gam_res['reduced']
            pr2 = gam_res['p_r2_coupling_reduced']
            del gam_res
    except:
        print('BAD dill',session,receiver_ID,man_type,man_val)
        continue

    if session != previous_sess:
        print('session',session)
        previous_sess = session
        dat = np.load(os.path.join(npz_path, session+'.npz'), allow_pickle=True)
        unit_info = dat['unit_info'].all()



    count_coupl = 0
    for var in full.var_list:
        if var.startswith('neu_'):
            count_coupl += 1

    info_neu = np.zeros(count_coupl,dtype=dict_type)
    info_neu['session'] = session
    info_neu['monkey'] = monkey_dict[session.split('s')[0]]
    info_neu['receiver unit id'] = receiver_ID
    info_neu['receiver electrode id'] = unit_info['electrode_id'][receiver_ID-1]
    info_neu['receiver cluster id'] = unit_info['cluster_id'][receiver_ID-1]
    info_neu['receiver channel id'] = unit_info['channel_id'][receiver_ID-1]
    info_neu['receiver brain area'] = unit_info['brain_area'][receiver_ID-1]

    info_neu['manipulation type'] = man_type
    info_neu['manipulation value'] = man_val

    cc = 0
    ii = 0
    for var in full.covariate_significance['covariate']:
        if not var.startswith('neu_'):
            ii += 1
            continue

        sender_ID = int(var.split('_')[1])
        info_neu['sender unit id'][cc] = sender_ID
        info_neu['sender electrode id'][cc] = unit_info['electrode_id'][sender_ID-1]
        info_neu['sender cluster id'][cc] = unit_info['cluster_id'][sender_ID-1]
        info_neu['sender channel id'][cc] = unit_info['channel_id'][sender_ID-1]
        info_neu['sender brain area'][cc] = unit_info['brain_area'][sender_ID-1]
        info_neu['pseudo-r2'][cc] = pr2
        if full.covariate_significance[ii]['p-val'] < 0.001:
            jj = np.where(reduced.covariate_significance['covariate'] == var)[0]
            info_neu['is significant'][cc] = reduced.covariate_significance[jj]['p-val'] < 0.001
            info_neu['p-value'][cc] = reduced.covariate_significance[jj]['p-val']

            cov_beta = reduced.cov_beta[reduced.index_dict[var],:]
            cov_beta = cov_beta[:,reduced.index_dict[var]]
            eig = np.linalg.eigh(cov_beta)[0]
            info_neu['log |cov|'][cc] = np.log(eig).sum()
        else:
            info_neu['is significant'][cc] = False
            info_neu['p-value'][cc] = full.covariate_significance[ii]['p-val']
            cov_beta = full.cov_beta[full.index_dict[var], :]
            cov_beta = cov_beta[:, full.index_dict[var]]
            eig = np.linalg.eigh(cov_beta)[0]
            info_neu['log |cov|'][cc] = np.log(eig).sum()

        if info_neu['receiver brain area'][cc] == info_neu['sender brain area'][cc]:
            impulse = np.zeros(13)
            impulse[6] = 1
            fX = full.smooth_compute([impulse], var, perc=0.99,trial_idx=None)[0]
            info_neu['area under filter'][cc] = 0.006 * (fX[7:11] - fX[0]).sum()
            info_neu['filter dur [ms]'][cc] = 6 * 4

            fX = fX - fX[0]
            fX = fX[7:11]


        else:
            impulse = np.zeros(301)
            impulse[150] = 1
            fX = full.smooth_compute([impulse], var, perc=0.99,trial_idx=None)[0]
            info_neu['area under filter'][cc] = 0.006 * (fX[151:251] - fX[0]).sum()
            info_neu['filter dur [ms]'][cc] = 6 * (251 - 151)
            fX = fX - fX[0]
            fX = fX[151:251]


        if var == 'neu_24':
            xxx=10
        info_neu['coupling strength'][cc] = np.linalg.norm(fX)
        info_neu['instantaneous coupling strength'][cc] = fX[0]
        info_neu['filter duration (ms)'][cc] = fX.shape[0]*6


        cc += 1
        ii += 1
    extr_first = True
    info_coupling = np.hstack((info_coupling,info_neu))


np.save('coupling_info_%s.npy'%session,info_coupling)
            
        