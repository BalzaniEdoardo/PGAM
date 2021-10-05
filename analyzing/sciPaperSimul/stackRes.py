import numpy as np
import os

fh_list = os.listdir('/Users/edoardo/Work/Code/GAM_code/analyzing/sciPaperSimul/res')
first =  True
for fh in fh_list:
    res = np.load(os.path.join('/Users/edoardo/Work/Code/GAM_code/analyzing/sciPaperSimul/res',fh),allow_pickle=True)
    if first:
        first = False
        table = res['table']
    else:
        table = np.hstack((table,res['table']))

table = table[table['variable'] == 'rad_target']

false_neg = (table[table['fit_type']=='all_vars']['significant'] == False).sum() / (table['fit_type']=='all_vars').sum()
false_pos = (table[table['fit_type']=='wo_rad_target']['significant'] == True).sum() / (table['fit_type']=='all_vars').sum()


