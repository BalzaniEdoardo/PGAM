#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:34:01 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc/')
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
from seaborn import heatmap
from path_class import get_paths_class
from knots_constructor import knots_cerate
from copy import deepcopy
from time import sleep
from spline_basis_toolbox import *
from bisect import bisect_left
from statsmodels.distributions import ECDF
import venn

npz_folder = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/'

dat = np.load('/Users/edoardo/Work/Code/GAM_code/analyzing/extract_tuning/eval_matrix_and_info.npz',allow_pickle=True)
info = dat['info']

monkey = 'Schro'


condition_dict = {
    'controlgain':[[1],[1.,2]],
    'ptb':[[0],[1]],
    'density':[[0.005],[0.0001]]
    }

monkey_dict = {'m44':'Quigley','m53':'Schro','m91':'Ody','m51':'Bruno'}


dtype_dict = {'names':['monkey','session','brain area','channel id','electrode id','cluster id','condition','value','unit','rate [Hz]'],
                               'formats':['U30','U30','U30',int,int,int,'U30',float, int,float]}


firing_rate = np.load('firing_rate_x_cond.npy')


for cond in condition_dict.keys():
    fr_sele = firing_rate[(firing_rate['condition'] == cond) * 
                          (firing_rate['monkey'] == monkey)]
    
    
    for ba in np.unique(fr_sele['brain area']):
        fr_ba = fr_sele[fr_sele['brain area'] == ba]
        
        # get the condition
        cond_x = condition_dict[cond][0][0]
        
        fr_x = []
        fr_y = {}
        for key in condition_dict[cond][1]:
            fr_y[key] = []
        
        for session in np.unique(fr_ba['session']):
            fr_sess = fr_ba[fr_ba['session']==session]
            skip_sess = False
            for cond_y in condition_dict[cond][1]:
                if (fr_sess['value'] == cond_y).sum() == 0:
                    skip_sess = True
                    break
            if skip_sess:
                continue
            
            for cond_y in condition_dict[cond][1]:
                fr_y[cond_y] = np.hstack((fr_y[cond_y], fr_sess['rate [Hz]'][fr_sess['value'] == cond_y]))
            fr_x = np.hstack((fr_x, fr_sess['rate [Hz]'][fr_sess['value'] == cond_x]))
        
        if len(fr_x) == 0:
            continue
        
        fig = plt.figure()
        ax = plt.subplot(111,aspect='equal')

        plt.title('Firing rate - %s %s'%(ba,cond))
        for cond_y in fr_y.keys():
            scat = plt.scatter(fr_x, fr_y[cond_y], label='%f'%cond_y)
            lreg = sts.linregress(fr_x, fr_y[cond_y])
            color = scat.get_facecolor()[0][:3]
            xx = np.linspace(np.min(fr_x),np.max(fr_x),100)
            plt.plot(xx,lreg.intercept + lreg.slope*xx,color=color)
            
        plt.plot(xx,xx,color='r',label='y=x')
        plt.legend()
        plt.xlabel('firing rate %s = %f'%(cond,cond_x))
        plt.ylabel('firing rate %s'%(cond))
        plt.savefig('Firing_Figs/firing_rate_%s_%s.png'%(cond,ba))
        
for cond in condition_dict.keys():
    fr_sele = firing_rate[(firing_rate['condition'] == cond) * 
                          (firing_rate['monkey'] == monkey)]
    
    
    for ba in np.unique(fr_sele['brain area']):
        fr_ba = fr_sele[fr_sele['brain area'] == ba]
        
        # get the condition
        cond_x = condition_dict[cond][0][0]
        
        fr_x = []
        fr_y = {}
        for key in condition_dict[cond][1]:
            fr_y[key] = []
        
        for session in np.unique(fr_ba['session']):
            fr_sess = fr_ba[fr_ba['session']==session]
            skip_sess = False
            for cond_y in condition_dict[cond][1]:
                if (fr_sess['value'] == cond_y).sum() == 0:
                    skip_sess = True
                    break
            if skip_sess:
                continue
            
            for cond_y in condition_dict[cond][1]:
                fr_y[cond_y] = np.hstack((fr_y[cond_y], fr_sess['rate [Hz]'][fr_sess['value'] == cond_y]))
            fr_x = np.hstack((fr_x, fr_sess['rate [Hz]'][fr_sess['value'] == cond_x]))
        
        if len(fr_x) == 0:
            continue
        
        fig = plt.figure()
        ax = plt.subplot(111)

        plt.title('Firing rate - %s %s\nbaseline %s=%f'%(ba,cond,cond,cond_x))
        for cond_y in fr_y.keys():
            scat = plt.hist( fr_y[cond_y]-fr_x, label='%f'%cond_y,bins=20,alpha=0.4,density=True)
            color = scat[2][0].get_facecolor()[:3]
            
            plt.plot([(fr_y[cond_y]-fr_x).mean()]*2,[0,np.max(scat[0])],'--',color=color)
            lreg = sts.linregress(fr_x, fr_y[cond_y])
            # color = scat.get_facecolor()[0][:3]
            # xx = np.linspace(np.min(fr_x),np.max(fr_x),100)
            # plt.plot(xx,lreg.intercept lreg.slope*xx,color=color)
            
        # plt.plot(xx,xx,color='r',label='y=x')
        plt.legend()
        plt.xlabel('delta firing rate')
        plt.ylabel('density')
        plt.savefig('Firing_Figs/firing_rate_delta_%s_%s.png'%(cond,ba))
        
        
for cond in condition_dict.keys():
    fr_sele = firing_rate[(firing_rate['condition'] == cond) * 
                          (firing_rate['monkey'] == monkey)]
    
    
    for ba in np.unique(fr_sele['brain area']):
        fr_ba = fr_sele[fr_sele['brain area'] == ba]
        
        # get the condition
        cond_x = condition_dict[cond][0][0]
        
        fr_x = []
        fr_y = {}
        for key in condition_dict[cond][1]:
            fr_y[key] = []
        
        for session in np.unique(fr_ba['session']):
            fr_sess = fr_ba[fr_ba['session']==session]
            skip_sess = False
            for cond_y in condition_dict[cond][1]:
                if (fr_sess['value'] == cond_y).sum() == 0:
                    skip_sess = True
                    break
            if skip_sess:
                continue
            
            for cond_y in condition_dict[cond][1]:
                fr_y[cond_y] = np.hstack((fr_y[cond_y], fr_sess['rate [Hz]'][fr_sess['value'] == cond_y]))
            fr_x = np.hstack((fr_x, fr_sess['rate [Hz]'][fr_sess['value'] == cond_x]))
        
        if len(fr_x) == 0:
            continue
        
        fig = plt.figure()
        ax = plt.subplot(111)

        plt.title('Firing rate - %s %s\nbaseline %s=%f'%(ba,cond,cond,cond_x))
        for cond_y in fr_y.keys():
            
            perc = 2*(fr_y[cond_y]-fr_x) / (fr_y[cond_y]+fr_x)
            scat = plt.hist( perc, label='%f'%cond_y,bins=20,alpha=0.4,density=True)
            color = scat[2][0].get_facecolor()[:3]
            
            plt.plot([(perc).mean()]*2,[0,np.max(scat[0])],'--',color=color)
            lreg = sts.linregress(fr_x, fr_y[cond_y])
            # color = scat.get_facecolor()[0][:3]
            # xx = np.linspace(np.min(fr_x),np.max(fr_x),100)
            # plt.plot(xx,lreg.intercept lreg.slope*xx,color=color)
            
        # plt.plot(xx,xx,color='r',label='y=x')
        plt.legend()
        plt.xlabel('delta firing rate')
        plt.ylabel('density')
        plt.savefig('Firing_Figs/firing_rate_percIncrease_%s_%s.png'%(cond,ba))
        
        