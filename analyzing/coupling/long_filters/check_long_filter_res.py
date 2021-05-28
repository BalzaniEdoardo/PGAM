import os,sys,re
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
# from data_handler import *
from GAM_library import *
import dill
import matplotlib.pylab as plt
import numpy as np

pr2_input = []
pr2_coupling = []
pr2_history = []
for fh in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/coupling_vs_spike_history/'):
    if not fh.endswith('.dill'):
        continue
    res = dill.load(open(
        '/Volumes/WD_Edo/firefly_analysis/LFP_band/coupling_vs_spike_history/'+fh,
        'rb'))

    pr2_history += [res['p_r2_spike_hist']]
    pr2_coupling += [res['p_r2_coupling']]
    pr2_input += [res['p_r2_input']]


plt.figure()
plt.scatter(pr2_input,pr2_history, marker='o',s=10,color='k',label='history')
plt.scatter(pr2_input,pr2_coupling,marker='o',s=10,color='y',label='history+coupling')
plt.plot([0,0.14],[0,0.14],'k')
plt.legend()

# plt.figure()
# plt.hist(np.log(np.array(pr2_input)),color='k',alpha=0.4,label='history',density=True)
# plt.hist(np.log(np.array(pr2_history)),color='b',alpha=0.4,label='history',density=True)
# plt.hist(np.log(np.array(pr2_coupling)),color='y',alpha=0.4,label='coupling',density=True)


plt.figure()
plt.scatter(pr2_history,pr2_coupling,marker='o',s=10,color='y')
plt.plot([0,0.2],[0,0.2],'k')

plt.xlabel('history')
plt.ylabel('history+coupling')