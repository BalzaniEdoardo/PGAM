#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:59:17 2020

@author: edoardo
"""
import numpy as np
import sys,os
folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
sys.path.append(os.path.join(main_dir,'GAM_library'))
sys.path.append(os.path.join(main_dir,'firefly_utils'))
sys.path.append(os.path.join(folder_name,'util_preproc'))
from path_class import get_paths_class

user_paths = get_paths_class()


path_GAM_library = os.path.join(main_dir,'GAM_library')
path_fitting_fld = os.path.join(main_dir,'fitting')
path_ff_utils = os.path.join(main_dir,'firefly_utils')
path_util_preproc = os.path.join(folder_name,'util_preproc')
path_jpn5_base = os.path.dirname(os.path.dirname(user_paths.get_path('data_hpc')))
path_to_fit_fld = os.path.join(path_jpn5_base,'jp_final_gam_fit_coupling')

# send gam_fit.py
os.system('sshpass -p "%s" scp %s jpn5@prince.hpc.nyu.edu:%s' % ('savin123!', os.path.join(path_fitting_fld,'gam_fit.py'),path_to_fit_fld))


# send any condition_list file
print('sending cond list:\n=================')
for root, dirs, files in os.walk(main_dir):
    if '/.' in root or '__pycache__' in root:
        continue
    for fhname in files:
        if 'condition_list_' in fhname:
            fh_path = os.path.join(root,fhname)
            # print('sending: \n',fh_path)
            os.system('sshpass -p "%s" scp %s jpn5@prince.hpc.nyu.edu:%s' % ('savin123!', fh_path, path_to_fit_fld))

            session = fhname.split('.')[0].split('condition_list_')[1]
            with open(os.path.join(main_dir,'sh_template.sh'),'r') as fh:
                sh_text = fh.read()
                fh.close()
                
            sh_text = sh_text.replace('template',session)
            with open(os.path.join(main_dir,'gam_fit_%s.sh'%session),'w') as fh:
                fh.write(sh_text)
                fh.close()
            
            os.system('sshpass -p "%s" scp %s jpn5@prince.hpc.nyu.edu:%s' % ('savin123!', os.path.join(main_dir,'gam_fit_%s.sh'%session), path_fitting_fld))
            os.remove(os.path.join(main_dir,'gam_fit_%s.sh'%session))
            
print('=================\n')


# send basis set 
path_to_basisscript = os.path.join(path_util_preproc,'basis_set_param_per_session.py')
dest_fld = os.path.join(path_jpn5_base,'GAM_Repo','preprocessing_pipeline','util_preproc')
os.system('sshpass -p "%s" scp %s jpn5@prince.hpc.nyu.edu:%s' % ('savin123!', path_to_basisscript, dest_fld))


