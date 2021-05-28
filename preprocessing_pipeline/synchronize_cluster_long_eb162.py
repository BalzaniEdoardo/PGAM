#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:59:17 2020

@author: edoardo
"""
import numpy as np
import sys,os

send_to = 'eb162'
PP = 'francimartaedoSA24!'



folder_name = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(folder_name)
if os.path.exists(os.path.join(main_dir,'GAM_library')):
    sys.path.append(os.path.join(main_dir,'GAM_library'))
    sys.path.append(os.path.join(main_dir,'firefly_utils'))
    sys.path.append(os.path.join(folder_name,'util_preproc'))
else:
    sys.path.append('/scratch/%s/GAM_Repo/GAM_library/'%send_to)
    sys.path.append('/scratch/%s/GAM_Repo/firefly_utils/'%send_to)
    sys.path.append('/scratch/%s/GAM_Repo/preprocessing_pipeline/util_preproc/'%send_to)


from path_class import get_paths_class

user_paths = get_paths_class()


path_GAM_library = os.path.join(main_dir,'GAM_library')
path_fitting_fld = os.path.join(main_dir,'fitting')
path_ff_utils = os.path.join(main_dir,'firefly_utils')
path_util_preproc = os.path.join(folder_name,'util_preproc')
path_jpn5_base = '/scratch/%s/'%send_to#os.path.dirname(os.path.dirname(user_paths.get_path('data_hpc')))
path_to_fit_fld = os.path.join(path_jpn5_base,'fit_longFilters')#'/scratch/%s/fit_ptb_as_variable'%send_to

# send gam_fit.py
print('\nsending:')
print(os.path.join(path_fitting_fld,'gam_fit.py'))


fhname = '/Users/edoardo/Work/Code/GAM_code/fitting/full_model_fit_long_filters.py'
with open(fhname,'r') as fh:
    string = fh.read()
    
if send_to == 'jpn5':
    string=string.replace('eb162', 'jpn5')
else:
    string=string.replace('jpn5', 'eb162')

with open(fhname,'w') as fh:
    fh.write(string)
    
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % ( PP, '/Users/edoardo/Work/Code/GAM_code/fitting/full_model_fit_long_filters.py',send_to,path_to_fit_fld))
path_to_script = os.path.join(path_ff_utils,'knots_constructor.py')
dest_fld = os.path.join(path_jpn5_base,'GAM_Repo','firefly_utils')
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, path_to_script, send_to,dest_fld))


# send any condition_list file
print('sending cond list:\n=================')
for root, dirs, files in os.walk(main_dir):
    if '/.' in root or '__pycache__' in root:
        continue
    for fhname in files:
        if 'condition_list_' in fhname:
            # if not 'm53s51' in fhname:
            #     continue
            fh_path = os.path.join(root,fhname)
            print('sending: \n',fh_path)
            os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, fh_path, send_to, path_to_fit_fld))
            print('sent')
            session = fhname.split('.')[0].split('condition_list_')[1]
            with open(os.path.join(main_dir,'sh_template.sh'),'r') as fh:
                sh_text = fh.read()
                fh.close()
            
            
            sh_text = sh_text.replace('template',session)
            sh_text=sh_text.replace('gam_fit.py','full_model_fit_long_filters.py')
            sh_text=sh_text.replace('jp_final_gam_fit_coupling','fit_longFilters')
            sh_text=sh_text.replace('jpn5@nyu.edu','eb162@nyu.edu')
            if send_to == 'eb162':
                old_activate = 'source /scratch/jpn5/select_hand_vel/venv/bin/activate'
                new_activate = 'source /scratch/eb162/venv/bin/activate'
                sh_text = sh_text.replace(old_activate, new_activate)
                
                
            with open(os.path.join(main_dir,'gam_fit_%s.sh'%session),'w') as fh:
                fh.write(sh_text)
                fh.close()
            print('sending: \n',os.path.join(main_dir,'gam_fit_%s.sh'%session))
            os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, os.path.join(main_dir,'gam_fit_%s.sh'%session),send_to, path_to_fit_fld))
            print('sent')
            os.remove(os.path.join(main_dir,'gam_fit_%s.sh'%session))
            
# # # print('=================\n')


# send basis set 
path_to_basisscript = os.path.join(path_util_preproc,'basis_set_param_per_session.py')
print('\nsending:')
print(path_to_basisscript)
dest_fld = os.path.join(path_jpn5_base,'GAM_Repo','preprocessing_pipeline','util_preproc')
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, path_to_basisscript, send_to, dest_fld))

path_to_script = os.path.join(path_GAM_library,'der_wrt_smoothing.py')
dest_fld = os.path.join(path_jpn5_base,'GAM_Repo','GAM_library')
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, path_to_script, send_to, dest_fld))

path_to_script = os.path.join(path_GAM_library,'GAM_library.py')
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, path_to_script,send_to, dest_fld))

path_to_script = os.path.join(path_GAM_library,'der_wrt_smoothing.py')
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, path_to_script, send_to, dest_fld))

path_to_script = os.path.join(path_GAM_library,'gam_data_handlers.py')
os.system('sshpass -p "%s" scp %s %s@greene.hpc.nyu.edu:%s' % (PP, path_to_script,send_to, dest_fld))


