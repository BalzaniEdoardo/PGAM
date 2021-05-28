#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:56:35 2021

@author: edoardo
"""
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
from spline_basis_toolbox import *
from GAM_library import *
from data_handler import *
from gam_data_handlers import *
import numpy as np
import scipy.stats as sts
from copy import deepcopy
import matplotlib.pylab as plt
from utils_loading import unpack_preproc_data, add_smooth


dat_old = np.load('/Users/edoardo/Work/Code/GAM_code/TMP/rad_path_comp/rad_path_checkTuning.npz')

x = dat_old['x']
fX_matrix_old = dat_old['fX_matrix']
fX_pci_matrix_old = dat_old['fX_pci_matrix']
fX_mci_matrix_old = dat_old['fX_mci_matrix']


dat_new = np.load('/Users/edoardo/Work/Code/GAM_code/TMP/rad_path_comp/rad_path_from_xy_checkTuning.npz')
fX_matrix_new = dat_new['fX_matrix']
fX_pci_matrix_new = dat_new['fX_pci_matrix']
fX_mci_matrix_new = dat_new['fX_mci_matrix']

par_list = [ 'cR', 'presence_rate', 'isiV',
            'unit_type']
fhName = '/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s113.npz'

( cont_rate_filter, presence_rate_filter, isi_v_filter, unit_type) = unpack_preproc_data(fhName, par_list)

cont_rate_filter = (cont_rate_filter < 0.2) | (unit_type == 'multiunit')
presence_rate_filter = presence_rate_filter > 0.9
isi_v_filter = isi_v_filter < 0.2
combine_filter = (cont_rate_filter) * (presence_rate_filter) * (isi_v_filter)
unit_list = np.arange(cont_rate_filter.shape[0])+1
unit_list = unit_list[combine_filter]


for j in range(4):
    plt.figure(figsize=(10,8))
    ccplot = 1
    for k in range(j*20, (j+1)*20):
        plt.subplot(4,5,ccplot)
        plt.title('unit %d'%unit_list[k])
        plt.plot(x,fX_matrix_old[k,:],color='k',label='old rad path')
        plt.fill_between(x,fX_mci_matrix_old[k,:],fX_pci_matrix_old[k,:],color='k',
                         alpha=0.4)
        
        
        plt.plot(x,fX_matrix_new[k,:],color='r',label='new rad path')
        plt.fill_between(x,fX_mci_matrix_new[k,:],fX_pci_matrix_new[k,:],color='r',
                         alpha=0.4)
        ccplot+=1
    
    plt.tight_layout()
    
        
    