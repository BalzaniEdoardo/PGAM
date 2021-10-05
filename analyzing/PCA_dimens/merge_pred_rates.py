import numpy as np
import os,re,sys
if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    session = 'm53s113'
else:
    sess_list = []
    for fhn in os.listdir('/scratch/jpn5/dataset_firefly'):
        if re.match('^m\d+s\d+.npz$', fhn):
            sess_list += [fhn.split('.')[0]]
    JOB = int(sys.argv[1]) - 1
    session = sess_list[JOB]

dat_PPC = np.load('meanFR_%s_%s.npz'%(session,'PPC'))
dat_PFC = np.load('meanFR_%s_%s.npz'%(session,'PFC'))
dat_MST = np.load('meanFR_%s_%s.npz'%(session,'MST'))

fr = np.hstack((dat_PPC['meanFr'],dat_PFC['meanFr'],dat_MST['meanFr']))
fr_noCP = np.hstack((dat_PPC['meanFr_noCP'],dat_PFC['meanFr_noCP'],dat_MST['meanFr_noCP']))
fr_noHist = np.hstack((dat_PPC['meanFr_noHist'],dat_PFC['meanFr_noHist'],dat_MST['meanFr_noHist']))
fr_noInt = np.hstack((dat_PPC['meanFr_noInt'],dat_PFC['meanFr_noInt'],dat_MST['meanFr_noInt']))


pr2 = np.hstack((dat_PPC['pseudo_r2'],dat_PFC['pseudo_r2'],dat_MST['pseudo_r2']))
nid = np.hstack((dat_PPC['neuron_fit'],dat_PFC['neuron_fit'],dat_MST['neuron_fit']))

srt_idx = np.argsort(nid)
fr = fr[:, srt_idx]
fr_noInt = fr_noInt[:,srt_idx]
fr_noCP = fr_noCP[:,srt_idx]
fr_noHist = fr_noHist[:,srt_idx]

pr2 = pr2[srt_idx]
nid = nid[srt_idx]

np.savez('meanFR_%s.npz'%(session),meanFr=fr,meanFr_noHist=fr_noHist,meanFr_noCP=fr_noCP,meanFr_noInt=fr_noInt,pseudo_r2=pr2,neuron_fit=nid)

os.remove("meanFR_%s_PFC.npz"%(session))
os.remove("meanFR_%s_PPC.npz"%(session))
os.remove("meanFR_%s_MST.npz"%(session))
