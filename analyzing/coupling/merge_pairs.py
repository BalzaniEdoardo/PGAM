import numpy as np
import os,re
from copy import deepcopy

lst = os.listdir(os.getcwd())
first=True
for fh in lst:
    if not re.match('^m\d+s\d+c.npz$',fh):
        continue
    print(fh)
    dat = np.load(fh)
    if first:
        first = False
        info = deepcopy(dat['info'])
        tunings = deepcopy(dat['tunings'])
        cond_list = deepcopy(dat['cond_list'])
    else:
        info = np.hstack((info,dat['info']))
        tunings = np.vstack((tunings,dat['tunings']))

np.savez('paired_coupling_flt.npz',tunings=tunings,info=info,cond_list=cond_list)