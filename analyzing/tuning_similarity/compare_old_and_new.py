#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:40:41 2021

@author: edoardo
"""
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import matplotlib.pylab as plt
import dill,sys,os
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from seaborn import *
from GAM_library import *
from scipy.integrate import simps
from spectral_clustering import *
from basis_set_param_per_session import *
from spline_basis_toolbox import *
from scipy.cluster.hierarchy import linkage,dendrogram


old_dat = np.load('/Users/edoardo/Work/Code/Angelaki-Savin/NIPS_Analysis/coupling_x_similarity/pairwise_L2_dist.npz',allow_pickle=True)
info_dict_old = old_dat['info_dict'].all()
beta_dict_old = old_dat['beta_dict'].all()
npz_old = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/oldConcatExample/m53s98.npz',allow_pickle=True)

dat = np.load('pairwise_L2_dist.npz',allow_pickle=True)
info_dict = dat['info_dict'].all()
beta_dict = dat['beta_dict'].all()
npz_new = np.load('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel/m53s98.npz',allow_pickle=True)

