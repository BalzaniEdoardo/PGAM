import numpy as np
import re,os

pattern = '^coupling_info_m\d+s\d+.npy$'
first = True
for name in os.listdir('/scratch/jpn5/coupling_info'):
    if not re.match(pattern,name):
        continue
    if first:
        info = np.load(name)
        first=False
    else:
        info = np.hstack((info, np.load(name)))
np.save('coupling_info.npy',info)
