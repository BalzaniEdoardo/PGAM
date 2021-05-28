import numpy as np
import dill,sys,os


sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library/')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils/')
from GAM_library import *
import matplotlib.pylab as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = "Arial"

sess_list = ['m53s40','m53s41','m53s93','m53s31','m53s36']

pr2_red = []
pr2_ful = []
for sess in sess_list:
    lst = os.listdir(os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/','gam_%s'%sess))
    for fh in lst:
        if not '_all_' in fh:
            continue
        res = dill.load(open(os.path.join('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/','gam_%s'%sess,fh),'rb'))

        pr2_ful += [res['p_r2_coupling_full']]
        pr2_red += [res['p_r2_coupling_reduced']]

pr2_red = np.array(pr2_red)
pr2_ful = np.array(pr2_ful)
ii = (pr2_red>0) & (pr2_ful > 0)
pr2_red = pr2_red[ii]
pr2_ful = pr2_ful[ii]
plt.figure()
ax = plt.subplot(111)
plt.scatter(pr2_ful,pr2_red,color=(0.5,)*3,s=8)
plt.scatter([np.median(pr2_ful)],[np.median(pr2_red)],marker='o',s=80,facecolors='none',edgecolors='k',lw=2)
plt.plot([0,0.25],[0,0.25],'k')

plt.xlabel('full model')
plt.ylabel('reduced model')
plt.title('cross-validated pseudo-R$^2$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('reduced_model_pr2.pdf')