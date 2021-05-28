import numpy as np
import os,re


fh_folder = '.'
first = True
for fh in os.listdir(fh_folder):
    if not re.match('^m\d+s\d+_LFP_coherence.npz$',fh):
        continue
    dat = np.load(os.path.join(fh_folder,fh))
    if first:
        first = False
        info = dat['info']
        choerence = dat['choerence']
        freq = dat['freq']
    else:
        try:
            info = np.hstack((info, dat['info']))
            choerence = np.vstack((choerence, dat['choerence']))
            assert(all(freq == dat['freq']))
        except:
            raise ValueError

np.savez('lfp_results.npz',info=info,choerence=choerence,freq=freq)
