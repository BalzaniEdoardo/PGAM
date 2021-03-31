#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:35:51 2021

@author: edoardo
"""

testMode=False
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))

if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
    folder_fh = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'
    done_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info'
    folder_incomplete = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info/incomplete'
    sess_list = os.listdir(folder_fh)
    JOB = 12
    is_clust = False
else:
    JOB = int(sys.argv[1]) - 1
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc/')
    folder_fh = '/scratch/jpn5/mutual_info/'
    done_folder = '/scratch/jpn5/mutual_info/done'
    folder_incomplete = '/scratch/jpn5/mutual_info/incomplete'
    is_clust = True

from spline_basis_toolbox import *
from utils_loading import unpack_preproc_data, add_smooth

from GAM_library import *
from data_handler import *
from gam_data_handlers import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import dill
import pandas as pd
import scipy.stats as sts
import scipy.linalg as linalg
from time import perf_counter
# from seaborn import heatmap
from path_class import get_paths_class
from knots_constructor import knots_cerate
from copy import deepcopy
from time import sleep
path_gen = get_paths_class()

import signal
from matplotlib import cm

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result

# model prediction
def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        sm_x[sel] = np.convolve(x[sel],filter,mode='same')
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk


from sklearn.decomposition import PCA
import seaborn as sbn
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
filtwidth = 20

t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h / np.sum(h)


fh_list = os.listdir(folder_fh)
num_sess = []
sess_list = []
for name in fh_list:
    if re.match('^gam_m\d+s\d+$',name):
        sess_list += [name.split('_')[1]]
        num_sess += [int(name.split('s')[1].split('.')[0])+float(name.split('_m')[1].split('s')[0])/100.]

sess_list = np.array(sess_list)[np.argsort(num_sess)]

# sess_list = list(sess_list)
# for fh in os.listdir(done_folder):
#     if not re.match("^mutual_info_and_tunHz_m\d+s\d+.dill$",fh):
#         continue
#     sess = fh.split('tunHz_')[1].split('.')[0]

#     try:
#         sess_list.remove(sess)
#     except:
#         pass
sess_list = np.array(sess_list)

session = 'm53s86'#sess_list[JOB]

print('SESSION: ',session)

if not is_clust:
    dat = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/%s.npz' % session,
                  allow_pickle=True)
else:
    dat = np.load('/scratch/jpn5/dataset_firefly/%s.npz' % session,
                  allow_pickle=True)

def pseudo_r2_compute(spk, family, modelX, params):

    lin_pred = np.dot(modelX, params)
    mu = family.fitted(lin_pred)
    res_dev_t = family.resid_dev(spk, mu)
    resid_deviance = np.sum(res_dev_t ** 2)

    null_mu = spk.sum() / spk.shape[0]
    null_dev_t = family.resid_dev(spk, [null_mu] * spk.shape[0])

    null_deviance = np.sum(null_dev_t ** 2)

    pseudo_r2 = (null_deviance - resid_deviance) / null_deviance
    return pseudo_r2

class empty_container(object):
    def __init__(self):
        pass

def predicet_rate(sm_handler,spikes_all,modelX,beta,cov_beta,index_dict,
                filter_trials,smooth_info,time_bin=0.006,skip_vars=[]):
    tuning_Hz = empty_container()
    # compute mutual info
    # compute the mu in log space
    mutual_info = {}
    mu = np.dot(modelX[filter_trials, :], beta)
    if len(skip_vars):
        keep_cols = []
        for var in index_dict.keys():
            if var in skip_vars:
                continue
            keep_cols = np.hstack((keep_cols, index_dict[var]))
        keep_cols = np.array(keep_cols, dtype=int)
        CC = cov_beta[keep_cols,:]
        CC = CC[:, keep_cols]
        MM = modelX[:, keep_cols]
        U, H = np.linalg.eigh(CC)
        ff = U < 5
        H = H[:, ff]
        U = U[ff]
        CC = np.dot(np.dot(H, np.diag(U)), H.T)

        sigma2 = np.einsum('ij,jk,ik->i', MM[filter_trials, :], CC, MM[filter_trials, :],
                           optimize=True)
        del MM,CC
    else:

        U,H = np.linalg.eigh(cov_beta)
        print(np.sort(U)[-10:],H.max(),H.min(),filter_trials.sum())
        ff = U < 5
        H = H[:,ff]
        U = U[ff]
        cov_beta = np.dot(np.dot(H,np.diag(U)),H.T)
        sigma2 = np.einsum('ij,jk,ik->i', modelX[filter_trials, :], cov_beta, modelX[filter_trials, :],
                           optimize=True)


    # convert to rate space
    lam_s = np.exp(mu + sigma2 * 0.5)
    return lam_s
    

def compute_predicted_rate(spikes_neu, sm_handler, modelX, trial_type, trial_idx, index_dict, neuron,
                        folder_fh, session, cond_dict,monkey_dict,brain_area,time_bin=0.006):
    
    # monkey = monkey_dict[session.split('s')[0]]
    # mi_dtype = {
    #     'names':('monkey', 'session', 'neuron', 'brain_area', 'variable',
    #              'manipulation_type',
    #              'manipulation_value',
    #              'mutual_info','significance','pseudo-r2'),
    #     'formats':('U30','U30',int,'U30','U30','U30',float,float,bool,float)
    #     }
    # tun_dtype = {
    #     'names':('monkey', 'session', 'neuron', 'brain_area','manipulation_type',
    #              'manipulation_value','variable', 'x',
    #              'y_model','y_raw','y_var_model'),
    #     'formats':('U30','U30',int,'U30','U30',float,'U30',object,object,object,object)
    #     }
    # mi_info = np.zeros(0,dtype=mi_dtype)
    # tuning_Hz = np.zeros(0,dtype=tun_dtype)
    print(cond_dict)
    for cond in cond_dict.keys():
        for value in cond_dict[cond]:
            fhName = os.path.join(folder_fh,'gam_%s'%session,
                    'fit_results_%s_c%d_%s_%.4f.dill'%(session,neuron,cond,value))
            # print(cond,value)
            # skip if fit is not done
            if not os.path.exists(fhName):
                continue
            
            with open(fhName,'rb') as fh:
                res = dill.load(fh)
                full = res['full']
                pr2 = res['p_r2_coupling_full']
                del res

            # if cond == 'controlgain' and value == 1:
            #     print('sele gain 1')
            #     modelX = modelX_dict['gain=1']
            #     index_dict = index_dict_cond['gain=1']
            #
            # if cond == 'controlgain' and value == 1.5:
            #     print('sele gain 1.5')
            #     modelX = modelX_dict['gain=1.5']
            #     index_dict = index_dict_cond['gain=1.5']
            #
            # else:
            #     print('sele gain other')
            #     modelX = modelX_dict['other']
            #     index_dict = index_dict_cond['other']

            if cond != 'ptb' and 't_ptb' in full.index_dict.keys():
                keep_var = [0]
                len_skip = 0
                for key in full.index_dict.keys():
                    if key != 't_ptb':
                        keep_var = np.hstack((keep_var,full.index_dict[key]))
                        full.index_dict[key] = full.index_dict[key] - len_skip
                    else:
                        len_skip = len(full.index_dict[key])
                keep_var = np.array(keep_var,dtype=int)
                full.index_dict.pop('t_ptb')
                full.beta = full.beta[keep_var]
                full.cov_beta = full.cov_beta[:,keep_var]
                full.cov_beta = full.cov_beta[ keep_var,:]

            beta_reorg = np.zeros(full.beta.shape[0])
            idx_sort = np.zeros(full.beta.shape[0],dtype=int)

            # remove variables that are in the new index_dict but not in the fit (like t_ptb, sometimes)
            rm_from_dict = list(set(index_dict.keys()).difference(set(full.index_dict.keys())))
            tmp = {}
            len_skip = 0
            model_matrix_filter = np.ones(modelX.shape[1], dtype=bool)
            for key in index_dict.keys():
                if key in rm_from_dict:
                    len_skip += len(index_dict[key])
                    model_matrix_filter[index_dict[key]] = False
                else:
                    tmp[key] = index_dict[key] - len_skip

            index_dict_use = deepcopy(tmp)

            for key in index_dict_use.keys():
                # print(key,len(full.index_dict[key]),len(index_dict_use[key]))
                idx_sort[index_dict_use[key]] = full.index_dict[key]
                beta_reorg[index_dict_use[key]] = full.beta[full.index_dict[key]]

            beta_reorg[0] = full.beta[0]
            cov_reorg = full.cov_beta[idx_sort,:]
            cov_reorg = cov_reorg[:,idx_sort]
            # print(cond, beta_reorg.shape,cov_reorg.shape)
            keep = np.zeros(spikes_neu.shape, dtype=bool)
            if cond != 'odd':
                for tr in np.where(trial_type[cond]==value)[0]:
                    keep[trial_idx==tr] = True
            else:
                all_trs = np.arange(trial_type.shape[0])
                all_trs = all_trs[trial_type['all']==1]
                if value == 1:
                    idx_subselect = all_trs[1::2]
                else:
                    idx_subselect = all_trs[::2]
                for tr in idx_subselect:
                    keep[trial_idx==tr] = True

            if cond =='reward' and value == 0:
                skip_vars = ['t_reward','rad_target']

            elif cond =='t_ptb' and value == 0 and ('t_ptb' in full.keys()):
                skip_vars = ['t_ptb']
            elif cond != 'ptb' and 't_ptb' in full.index_dict.keys():
                skip_vars = ['t_ptb']
            else:
                skip_vars = []
            rate = predicet_rate(sm_handler,spikes_neu, modelX[:,model_matrix_filter], beta_reorg, cov_reorg, index_dict_use,
                   keep,full.smooth_info,time_bin=0.006,skip_vars=skip_vars)

            # mi_info_tmp = np.zeros(len(mi.keys()),dtype=mi_dtype)
            # tuning_Hz_tmp = np.zeros(len(mi.keys()),dtype=tun_dtype)
            # cc = 0
            # for var in mi.keys():
            #     pval = full.covariate_significance['p-val'][full.covariate_significance['covariate']==var][0]
            #     mi_info_tmp['monkey'][cc] = monkey
            #     mi_info_tmp['session'][cc] = session
            #     mi_info_tmp['brain_area'][cc] = brain_area
            #     mi_info_tmp['neuron'][cc] = neuron
            #     mi_info_tmp['manipulation_type'][cc] = cond
            #     mi_info_tmp['manipulation_value'][cc] = value
            #     mi_info_tmp['variable'][cc] = var
            #     mi_info_tmp['mutual_info'][cc] = mi[var]
            #     mi_info_tmp['significance'][cc] = pval < 0.001
            #     mi_info_tmp['pseudo-r2'][cc] = pr2
                
            #     try:
            #         func = getattr(tun, var)
            #         tuning_Hz_tmp['monkey'][cc] = monkey
            #         tuning_Hz_tmp['session'][cc] = session
            #         tuning_Hz_tmp['brain_area'][cc] = brain_area
            #         tuning_Hz_tmp['neuron'][cc] = neuron
            #         tuning_Hz_tmp['manipulation_type'][cc] = cond
            #         tuning_Hz_tmp['manipulation_value'][cc] = value
            #         tuning_Hz_tmp['variable'][cc] = var
            #         tuning_Hz_tmp['x'][cc] = func.x
            #         tuning_Hz_tmp['y_model'][cc] = func.y_model
            #         tuning_Hz_tmp['y_raw'][cc] = func.y_raw
            #         tuning_Hz_tmp['y_var_model'][cc] = func.y_var_model
            #     except:
            #         tuning_Hz_tmp['monkey'][cc] = monkey
            #         tuning_Hz_tmp['session'][cc] = session
            #         tuning_Hz_tmp['brain_area'][cc] = brain_area
            #         tuning_Hz_tmp['neuron'][cc] = neuron
            #         tuning_Hz_tmp['manipulation_type'][cc] = cond
            #         tuning_Hz_tmp['manipulation_value'][cc] = value
            #         tuning_Hz_tmp['variable'][cc] = var
            #         tuning_Hz_tmp['x'][cc] = np.zeros((15,))*np.nan
            #         tuning_Hz_tmp['y_model'][cc] = np.zeros((15,))*np.nan
            #         tuning_Hz_tmp['y_raw'][cc] = np.zeros((15,))*np.nan
            #         tuning_Hz_tmp['y_var_model'][cc] = np.zeros((15,))*np.nan
            #         print('no tuning funcitons')
                
            #     cc+=1
                
                
            
            # mi_info = np.hstack((mi_info, mi_info_tmp))
            # tuning_Hz = np.hstack((tuning_Hz, tuning_Hz_tmp))
    return rate,pr2
            

# from numba import njit

# session = 'm53s113'
monkey_dict = {'m53':'Schro','m44':'Quigley','m72':'Marco','m51':'Bruno',
               'm71':'Viktor'}
# postprocessing extract mutual information

# load and filter

fit_unit = 1

# Unpack all the variables

# GAM model inputs
concat = dat['data_concat'].all()
X = concat['Xt']
spikes = concat['Yt']
var_names = dat['var_names']
trial_idx = concat['trial_idx']
lfp_beta = concat['lfp_beta']
lfp_alpha = concat['lfp_alpha']
lfp_theta = concat['lfp_theta']
trial_type = dat['info_trial'].all().trial_type

del concat

if testMode:
    keeptr = np.where(trial_type['all'])[0][::15]
    sel = np.zeros(trial_idx.shape,dtype=bool)
    for tr in keeptr:
        sel[trial_idx==tr]=True
    X = X[sel]
    spikes = spikes[sel]
    trial_idx = trial_idx[sel]
    lfp_beta = lfp_beta[sel]
    lfp_alpha = lfp_alpha[sel]
    lfp_theta = lfp_theta[sel]
    # trial_type = trial_type[keeptr]
    del sel,keeptr

## Info regarding the units:
# quality metric for the units
unit_info = dat['unit_info'].all()
unit_type = unit_info['unit_type']
isiV = unit_info['isiV'] # % of isi violations 
cR = unit_info['cR'] # contamination rate
presence_rate = unit_info['presence_rate'] # measure of stability of the firing in time

# std filters for unit quality
cont_rate_filter = (cR < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate > 0.9
isi_v_filter = isiV < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)

brain_area = unit_info['brain_area'][combine_filter]
cluster_id = unit_info['cluster_id'][combine_filter]
electrode_id = unit_info['electrode_id'][combine_filter]
channel_id = unit_info['channel_id'][combine_filter]
unit_type = unit_type[combine_filter]

spikes =spikes[:,combine_filter]
lfp_alpha = lfp_alpha[:,combine_filter]
lfp_beta = lfp_beta[:,combine_filter]
lfp_theta = lfp_theta[:,combine_filter]
keep_unit = np.arange(1,1+combine_filter.shape[0])[combine_filter]
# fit_unit = keep_unit[fit_unit]

##  truncate ang dist (this variable become noisy when the monkey is close to the target)
ang_idx = np.where(np.array(var_names) == 'ang_target')[0][0]
X[np.abs(X[:, ang_idx]) > 50, ang_idx] = np.nan


link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)


cond_type = 'all'
cond_value = 1

print(' condition', cond_type, cond_value)


# numfit+=1

if cond_type == 'odd':
    all_trs = np.arange(trial_type.shape[0])
    all_trs = all_trs[trial_type['all']==1]
    if cond_value == 1:
        idx_subselect = all_trs[1::2]
    else:
        idx_subselect = all_trs[::2]
else:
    idx_subselect = np.where(trial_type[cond_type] == cond_value)[0]

test_trials = idx_subselect[::10]
train_trials = np.sort(list(set(idx_subselect).difference(set(idx_subselect[::10]))))


# take the train trials
keep = []
for ii in train_trials:
    keep = np.hstack((keep, np.where(trial_idx == ii)[0]))
    
keep_test = []
for ii in test_trials:
    keep_test = np.hstack((keep_test, np.where(trial_idx == ii)[0]))


print(' condition', cond_type, cond_value)


keep = np.array(keep, dtype=int)
trial_idx_train = trial_idx[keep]

keep_test = np.array(keep_test, dtype=int)
trial_idx_test = trial_idx[keep_test]


cond_knots = cond_type

cond_dict = {'all':[True]}
    
# create a smooth handler
time_bin = 0.006
hand_vel_temp = True
mi_dtype = {
       'names':('monkey', 'session', 'neuron', 'brain_area', 'variable',
            'manipulation_type',
            'manipulation_value',
            'mutual_info','significance','pseudo-r2'),
       'formats':('U30','U30',int,'U30','U30','U30',float,float,bool,float)}

tun_dtype = {
        'names':('monkey', 'session', 'neuron', 'brain_area','manipulation_type',
             'manipulation_value','variable', 'x',
             'y_model','y_raw','y_var_model'),
        'formats':('U30','U30',int,'U30','U30',float,'U30',object,object,object,object)
    }

if os.path.exists(os.path.join(folder_incomplete,'mutual_info_and_tunHz_%s.dill'%session)):
    with open(os.path.join(folder_incomplete,'mutual_info_and_tunHz_%s.dill'%session),'rb') as fh:
        incomp = dill.load(fh)
        mi_info = incomp['mutual_info']
        tuning_Hz = incomp['tuning_Hz']
else:

    mi_info = np.zeros(0,dtype=mi_dtype)
    tuning_Hz = np.zeros(0,dtype=tun_dtype)


if 'controlgain' in cond_dict.keys():
    ctrl_cond = cond_dict.pop('controlgain')
else:
    ctrl_cond = []

counter_save = 0
#### COMPUTE CONTROL GAIN STUFF

# create rate matrix
rate_matrix = np.zeros((keep.shape[0],spikes.shape[1]))
brain_area_res = np.zeros(spikes.shape[1],dtype='U30')
neurin_ids = np.zeros(spikes.shape[1],dtype=int)
spikes_raw = np.zeros((keep.shape[0],spikes.shape[1]))
pr2_matrix =  np.zeros(spikes.shape[1],dtype=float)
ccn = 0

sel_trial = trial_idx_train < np.unique(trial_idx_train)[22]
spk = spikes[keep]
flt = pop_spike_convolve(spk[sel_trial],trial_idx_train[sel_trial],h)

brain_area_coupl = np.zeros(spk.shape[1],dtype='U30')
predict = np.zeros((sel_trial.sum(),spk.shape[1]),dtype=float)
p_r2_coupling = np.zeros(spk.shape[1],dtype=float)


for fhname in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/coupling_vs_spike_history/'):
    if re.match('^compare_hist_filt_coupling_%s_c\d+.dill$'%session,fhname) is None:
        continue
    
    neuron = int(fhname.split('%s_c'%session)[1].split('.')[0])
    path = os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/coupling_vs_spike_history/',fhname)
    results = dill.load(open(path,'rb'))
    pr2_matrix[ccn] = results['p_r2_coupling']
    full = results['full_coupling']
    X_list = []
    idx_neu = np.where(keep_unit == neuron)[0][0]
   
    #
    for var in full.var_list:
   
        # for now skip
        
        if var=='hand_vel1' or var == 'hand_vel2':
            continue
        
        if var in ['lfp_beta', 'lfp_alpha', 'lfp_theta']:
            is_cyclic = True
        else:
            is_cyclic = False
    
        if var == 'lfp_theta':
            x = lfp_theta[keep, idx_neu]
            x_test = lfp_theta[keep_test, idx_neu]
    
        elif var == 'lfp_beta':
            x = lfp_beta[keep, idx_neu]
            x_test = lfp_beta[keep_test, idx_neu]
    
    
        elif var == 'lfp_alpha':
            x = lfp_alpha[keep, idx_neu]
            x_test = lfp_alpha[keep_test,idx_neu]
    
    
        elif var == 'spike_hist':
            tmpy = spikes[keep, idx_neu]
            x = tmpy
            
            x_test = spikes[keep_test, idx_neu]
    
            # x = np.hstack(([0], tmpy[:-1]))
        elif var.startswith('neu_'):
            
            tmpy = spikes[keep, np.where(keep_unit == int(var.split('neu_')[1]))[0][0]]
            x = tmpy
            
    
        else:
            cc = np.where(var_names == var)[0][0]
            x = X[keep, cc]
            
            x_test = X[keep_test, cc]
        
        X_list += [[x[sel_trial]]]
    predict[:,ccn] = full.predict(X_list,full.var_list)
    brain_area_coupl[ccn] = results['brain_area']
    p_r2_coupling[ccn] = results['p_r2_coupling']
    
    ccn+=1
    
    
    
    

# with open('rate_matrix_tunHz_%s.dill'%session,'wb') as fh:

#     vv = {'rate_model':rate_matrix, 'brain_area': brain_area_res,
#           'neuron_id':neurin_ids,
#           'spikes_raw':spikes_raw, 'pseudor2':pr2_matrix}
#     fh.write(dill.dumps(vv))

 


blues = cm.get_cmap('Blues', 256)
greens = cm.get_cmap('Greens', 256)
reds = cm.get_cmap('Reds', 256)

newcolors = np.vstack((greens(np.linspace(0, 1, 256)),
                        blues(np.linspace(0, 1, 256)),
                        reds(np.linspace(0, 1, 256))))

newcmp = ListedColormap(newcolors)


cm_dict = {'MST':cm.Greens,'PPC':cm.Blues,'PFC':cm.Reds}
kk=1
collev = 0

# # 30 sec
sm_predict = pop_spike_convolve(predict,trial_idx_train[sel_trial],h)
smAll_plot = flt/0.006
smModel_plot = sm_predict/0.006

smAll_plot = smAll_plot[:,pr2_matrix>0.05]
smModel_plot = smModel_plot[:,pr2_matrix>0.05]
brain_area_coupl_flt = brain_area_coupl[pr2_matrix>0.05]

modY = smModel_plot
model = PCA(1)
pcares = model.fit(modY)

srt_pca = np.argsort(pcares.components_[0,:].flatten())

smModel_plot[:,brain_area_coupl_flt=='MST'] = np.clip(smModel_plot[:,brain_area_coupl_flt=='MST'],0,50)
smModel_plot[:,brain_area_coupl_flt=='PPC'] =  np.clip(51 + smModel_plot[:,brain_area_coupl_flt=='PPC'],50,100)
smModel_plot[:,brain_area_coupl_flt=='PFC'] =  np.clip(101 + smModel_plot[:,brain_area_coupl_flt=='PFC'],100,150)


smAll_plot[:,brain_area_coupl_flt=='MST'] = np.clip(smAll_plot[:,brain_area_coupl_flt=='MST'],0,50)
smAll_plot[:,brain_area_coupl_flt=='PPC'] =  np.clip(51+ smAll_plot[:,brain_area_coupl_flt=='PPC'],50,100)
smAll_plot[:,brain_area_coupl_flt=='PFC'] =  np.clip(101 + smAll_plot[:,brain_area_coupl_flt=='PFC'],100,150)

smAll_plot = smAll_plot[:,srt_pca]
smModel_plot = smModel_plot[:,srt_pca]
ba_res = brain_area_coupl_flt[srt_pca]

idxsrt = np.hstack((np.where(ba_res=='MST')[0],
                    np.where(ba_res=='PPC')[0],
                    np.where(ba_res=='PFC')[0]))

plt.figure(figsize=(10,5))

ax1 = plt.subplot(2,1,1)

smAll_plot 
sbn.heatmap(smAll_plot[:,idxsrt[::-1]].T,cmap=newcmp,ax=ax1,vmax=150)

plt.xticks([])
plt.yticks([])
ax2 = plt.subplot(2,1,1+1)
sbn.heatmap(smModel_plot[:,idxsrt[::-1]].T,cmap=newcmp,ax=ax2,vmax=150)
plt.xticks([])
plt.yticks([])

  
plt.tight_layout()  
# plt.savefig('%s_60sec_raw_vs_model.tiff'%session)
    
