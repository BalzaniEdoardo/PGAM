import numpy as np
import os
cnt = 0
dir_list = ['F','S','N']
for bsdir in dir_list:
    for root, dirs, files in os.walk(bsdir, topdown=False):
        for fhName in files:
            if fhName.startswith('gam_fit_useCoup'):
                cnt += 1
print('tot fits: ',cnt)

