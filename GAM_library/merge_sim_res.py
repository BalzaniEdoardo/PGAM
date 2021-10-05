import numpy as np
import os

dtype_table_all = {'names':('sim_num','seed')+('session', 'neuron', 'variable', 'significant', 'fit_type'),
                  'formats':(int,int,'U20',int,'U20',bool,'U20')}
table_all = np.zeros(0, dtype=dtype_table_all)
tunRes_dict = {}
tunResWO_dict = {}
first = False
for k in range(100):
    fl_path = '/Volumes/WD_Edo/firefly_analysis/LFP_band/simSciPaper/sim_%d_res_simul.npz'%(k+1)
    if not os.path.exists(fl_path):
        print(fl_path)
        continue
    dat = np.load(fl_path)
    table = dat['table']
    tunRes = dat['tunRes']
    tunResWO = dat['tunResWO']
    groundTruth = dat['groundTruth']
    seed = dat['seed']
    tunRes_dict[k] = tunRes
    tunResWO_dict[k] = tunResWO

    tmp = np.zeros(table.shape[0], dtype=dtype_table_all)
    for name in table.dtype.names:
        tmp[name] = table[name]

    tmp['seed'] = seed
    tmp['sim_num'] = k
    table_all = np.hstack((table_all,tmp))

np.savez('/Volumes/WD_Edo/firefly_analysis/LFP_band/simSciPaper/stackedSim.npz',
         tunRes=tunRes_dict,tunResWO=tunResWO_dict,table=table_all,groundTruth=groundTruth)


table = table_all[table_all['variable']=='rad_target']

## inclusions
false_pos = 1 - (table['significant'] & (table['fit_type'] == 'all_vars')).sum()/(table['fit_type'] == 'all_vars').sum()
false_neg = 1 - (np.array(~table['significant'],dtype=bool) & (table['fit_type'] == 'wo_rad_target')).sum()/(table['fit_type'] == 'wo_rad_target').sum()
