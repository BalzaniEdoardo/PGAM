import numpy as np
import os,re
from copy import deepcopy

pattern = '^tuning_info_m\d+s\d+.npy$'

first = True
for name in os.listdir('.'):
    if not re.match(pattern,name):
        continue
    dat = np.load(name)
    bl = dat['variable'] != ''
    dat = dat[bl]
    if first:
        result = deepcopy(dat)
        first = False
    else:
        result = np.hstack((result, dat))
np.save('tuning_info.npy', result)