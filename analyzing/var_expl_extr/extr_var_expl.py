import numpy as np
import dill,os,sys

if os.path.exists('/Users/edoardo/Work/Code/GAM_code/GAM_library'):
    sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
else:
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library')

from GAM_library import *
dt_dict = {'names':('session','unit','brain_area','pseudo-r2','var_expl'),'formats':('U20',int,'U3',float,float)}
var_expl_table = np.zeros(0,dtype=dt_dict)
for fld in os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'):
    lst_fh = os.listdir('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'+fld)
    cc=1
    print(fld)
    for fh in lst_fh:
        if cc % 100 == 1:

            print('%d/%d'%(cc,len(lst_fh)))
        if not '_all_' in fh:
            continue

        res = dill.load(open('/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/'+
            fld+'/'+fh,
                'rb'))

        session = fh.split('results_')[1].split('_')[0]
        unit = int(fh.split('_c')[1].split('_')[0])
        tmp = np.zeros(1,dtype=dt_dict)
        tmp['brain_area'] = res['brain_area']
        tmp['session'] = session
        tmp['pseudo-r2'] = res['p_r2_coupling_full']
        tmp['var_expl'] = res['full'].var_expl
        var_expl_table = np.hstack((var_expl_table,tmp))

        cc+=1

import matplotlib.pylab as plt

ax = plt.subplot(111)
plt.hist(var_expl_table['var_expl'],range=(-1,1),density=True,bins=20,facecolor='none',edgecolor='k',lw=1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('R$^2$')
plt.ylabel('frequency')
