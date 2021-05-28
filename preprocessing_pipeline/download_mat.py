import numpy as np
import os,sys,re

fold_list = [
    '/Volumes/server/Data/Monkey2_newzdrive/Marco/U-probe'
]

fh_list = []
lst = os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/concatenation_with_accel')

monk = ['m71','m72']
for fh in lst:
    if not re.match('^m\d+s\d+.npz$',fh):
        continue
    if fh.split('s')[0] in monk:
        fh_list += [fh.split('.')[0]]

dict_fld = {}
for fld in fold_list:
    for root, dirs, files in os.walk(fld,topdown=True):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        print(root)
        if not 'Pre-processing X E' in root:
            continue

        for fh in files:
            sess = re.findall('m\d+s\d+',fh)
            assert(len(sess)==1)
            sess = sess[0]
            dict_fld[sess] = root
            if sess in fh_list:
                continue
            if sess == 'm72s7':
                continue
            if fh == 'm72s11.mat' or fh == 'lfp_beta_m72s11.mat':
                continue



            print('copynig %s'%fh)
            # dict_fld[sess] = root
            # dr = root.replace(' ','\ ')
            # os.system('cp %s /Volumes/WD_Edo/firefly_analysis/LFP_band/DATASET_accel' % (os.path.join(dr,fh)))



