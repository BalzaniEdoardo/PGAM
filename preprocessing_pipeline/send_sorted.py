#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:39:49 2020

@author: edoardo
"""


import numpy as np
import sys,os,re
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(main_dir,'preprocessing_pipeline','util_preproc'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from path_class import get_paths_class
from scipy.io import loadmat
from data_handler import *
from extract_presence_rate import *
from scipy.io import loadmat
from path_class import get_paths_class
path_user = get_paths_class()
import matplotlib.pylab as plt
from GAM_library import *
import subprocess
from time import sleep

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True)
    # Read stdout from subprocess until the buffer is empty !
    lineList = []
    for line in iter(p.stdout.readline, b''):
        if line: # Don't print blank lines
            lineList += [line.decode('utf-8').rstrip()]
            
    # This ensures the process has completed, AND sets the 'returncode' attr
    while p.poll() is None:                                                                                                                                        
        sleep(.1) #Don't waste CPU-cycles
    # Empty STDERR buffer
    err = p.stderr.read()
    if p.returncode != 0:
       # The run_command() function is responsible for logging STDERR 
       return(False,"Error: " + str(err))
    else:
        return(True,lineList)
     
    
list_copy = ['spike_times_master_clock.npy','spike_times.npy','spike_clusters.npy',
             'spike_templates.npy',
             'amplitudes.npy',
             'templates.npy',
             'whitening_mat_inv.npy',
             'channel_map.npy']
             # 'pc_features.npy',
             # 'pc_feature_ind.npy']
        
regex = '^m[0-9]+s[0-9]+.npz$'

sshPass = 'sshpass'

commandMkDir = sshPass + ' -p ' + ' "' + 'francimartaedoSA22!!' + '" ssh  eb162@prince.hpc.nyu.edu mkdir %s'
commandCopy = sshPass + ' -p ' + ' "' + 'francimartaedoSA22!!' + '" scp %s  eb162@prince.hpc.nyu.edu:%s'
commandLs = sshPass + ' -p ' + ' "' + 'francimartaedoSA22!!' + '" ssh  eb162@prince.hpc.nyu.edu ls %s'

dest_path = '/scratch/eb162/sorted/'

list_session = []
for root, dirs, files in os.walk("/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET"):
    for fh in files:
        if re.match(regex,fh):
            session = fh.split('.')[0]
            if  not 'm44' in session:
                continue
              # make dir
            run_command(commandMkDir%(os.path.join(dest_path,session)))
            run_command(commandMkDir%(os.path.join(dest_path,session+'_PLEXON')))
            print(session)
            _,lst_array = run_command(commandLs%os.path.join(dest_path,session))
            _,lst_linear = run_command(commandLs%os.path.join(dest_path,session+'_PLEXON'))
            
            if type(lst_array) == str:
                lst_array = []
                
            if type(lst_array) == str:
                lst_linear = []
                
            sorted_fold = path_user.get_path('server_data',session)
            
            
            for send_fh in list_copy:
                if send_fh in  lst_array:
                    continue
                if os.path.exists(os.path.join(sorted_fold,send_fh)):
                    send_path = os.path.join(sorted_fold,send_fh)
                    send_path = '\ '.join(send_path.split(' '))
                    # send
                    dest_array = os.path.join(dest_path,session)
                    
                    run_command(commandCopy%(send_path, dest_array))
            
            if not 'Utah' in sorted_fold:
                sorted_fold = path_user.get_path('cluster_array_data',session)
                for send_fh in list_copy:
                    if send_fh in  lst_linear:
                        continue
                    if os.path.exists(os.path.join(sorted_fold,send_fh)):
                        send_path = os.path.join(sorted_fold,send_fh)
                        send_path = '\ '.join(send_path.split(' '))
                        # send
                        dest_array = os.path.join(dest_path,session+'_PLEXON')
                        
                        run_command(commandCopy%(send_path, dest_array))
                        
                    