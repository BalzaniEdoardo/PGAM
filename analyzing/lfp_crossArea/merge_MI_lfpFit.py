import numpy as np
import os, dill, re


fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info_LFP/'
pattern = '^mutual_info_and_tunHz_m\d+s\d+.dill$'

lst_fh = os.listdir(fld)
first = True
for fh in lst_fh:
    if not re.match(pattern,fh):
        continue
    fhName = os.path.join(fld,fh)
    res = dill.load(open(fhName,'rb'))

    if first:
        first = False
        MI = res['mutual_info']
        tun = res['tuning_Hz']
    else:
        MI = np.hstack((MI,res['mutual_info']))
        tun = np.hstack((tun, res['tuning_Hz']))
np.savez('mutual_info.npz',tuning_Hz=tun, mutual_info=MI)
