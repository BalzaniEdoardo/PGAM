import numpy as np
import matplotlib.pylab as plt
import dill
import os, re
from copy import deepcopy
import pandas as pd
import seaborn as sbn
from time import perf_counter
from matplotlib.patches import PathPatch
from statsmodels.formula.api import ols
import statsmodels.api as sm

import sys
sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
from GAM_library import *
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.family':'Arial'})

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


color_dict = {'PPC': 'b', 'PFC': 'r', 'MST': 'g', 'VIP': 'k'}
fld_file = '/Volumes/WD_Edo/firefly_analysis/LFP_band/processed_data/mutual_info/'
lst_done = os.listdir(fld_file)
# mutual_info_and_tunHz_m53s42.dill
fld_fit = []
first = True
dtype_dict = {'names':('session','unit','par_full','par_reduced'),'formats':('U30',int,int,int)}
table = np.zeros(0,dtype=dtype_dict)
for fhName in lst_done:

    if not re.match('^mutual_info_and_tunHz_m\d+s\d+.dill$', fhName):
        continue
    sess = fhName.split('mutual_info_and_tunHz_')[1].split('.')[0]
    print(sess)
    with open(os.path.join(fld_file, fhName), 'rb') as fh:
        res = dill.load(fh)
        mi = res['mutual_info']
        # tun = res['tuning_Hz']
    units = np.unique(mi['neuron'])
    table_tmp = np.zeros(units.shape[0], dtype=dtype_dict)
    table_tmp['session'] = sess

    fld = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'%sess
    cc = 0
    for un in units:
        try:
            with open(os.path.join(fld,'fit_results_%s_c%d_all_1.0000.dill'%(sess,un)),'rb') as ffh:
                res = dill.load(ffh)
        except:
            continue

        par_full = 0
        par_reduced = 0

        for var in res['full'].var_list:
            npar = res['full'].smooth_info[var]['knots'][0].shape[0]
            par_full += npar
            if res['reduced'] is None:
                continue
            if var in res['reduced'].var_list:
                par_reduced += npar

        table_tmp['unit'][cc] = un
        table_tmp['par_full'][cc] = par_full
        table_tmp['par_reduced'][cc] = par_reduced

        cc+=1
    table = np.hstack((table,table_tmp))
        # for var in res['reduced'].var_list:
        #     par_reduced += res['reduced'].smooth_info[var]['knots'][0].shape[0]

        # par_full_param = np.sum(list(dict_var_knot_size.values()))



table = table[table['par_full'] != 0]

np.save('param_count.npy',table)

plt.figure()
ax = plt.subplot(111)
plt.ylabel('counts',fontsize=12)
plt.xlabel('# of parameters',fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(table['par_full'],bins=20,alpha=0.60,label='full')
plt.hist(table['par_reduced'],bins=20,alpha=0.60,label='reduced')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.savefig('counts.pdf')



plt.figure()
ax = plt.subplot(111)
plt.ylabel('reduced',fontsize=12)
plt.xlabel('full model',fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.scatter(table['par_full'],table['par_reduced'],color='k',s=60)
xlim = plt.xlim()
ylim = plt.ylim(xlim)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

plt.savefig('scatter_counts.pdf')

#     with open(os.path.join(fld_file, fh), 'rb') as fh:
#         res = dill.load(fh)
#         mi = res['mutual_info']
#         tun = res['tuning_Hz']
#     if first:
#         mutual_info = deepcopy(mi)
#         tuning = deepcopy(tun)
#         first = False
#     else:
#         mutual_info = np.hstack((mutual_info, mi))
#         tuning = np.hstack((tuning, tun))
#
# # filter only density manip
# keep_sess = np.unique(mutual_info['session'][mutual_info['manipulation_type'] == 'density'])
# filt_sess = np.zeros(mutual_info.shape, dtype=bool)
# for sess in keep_sess:
#     filt_sess[mutual_info['session'] == sess] = True
#
# filt = (mutual_info['manipulation_type'] == 'all') & (mutual_info['pseudo-r2'] > 0.01) & \
#        (~np.isnan(mutual_info['mutual_info'])) & filt_sess
#
# tuning = tuning[filt]
# mutual_info = mutual_info[filt]
