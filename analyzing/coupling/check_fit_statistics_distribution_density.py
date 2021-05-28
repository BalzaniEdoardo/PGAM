import numpy as np
# import matplotlib.pylab as plt
# import seaborn as sbn
# import pandas as pd
# from sklearn.model_selection import GridSearchCV
# from sklearn.mixture import GaussianMixture
# from scipy.interpolate import griddata
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import os,dill,sys,inspect

if os.path.exists('/scratch/jpn5'):
    sys.path.append('/scratch/jpn5/GAM_Repo/GAM_library')
    sys.path.append('/scratch/jpn5/GAM_Repo/preprocessing_pipeline/util_preproc')
    sys.path.append('/scratch/jpn5/GAM_Repo/firefly_utils')
    fld_dill = '/scratch/jpn5/mutual_info//gam_%s'
    JOB = int(sys.argv[1])-1
else:
    thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
    sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
    JOB=1

def select_var(dat,sender,receiver,var):
    sel = (dat['sender brain area'] == sender) & (dat['receiver brain area'] == receiver)
    return dat[var][sel]

# select some significant coupling cross-area
dat = np.load('coupling_info.npy', allow_pickle=True)
dat = dat[(dat['pseudo-r2']>=0.01)]

# monkey:
dat = dat[dat['monkey'] == 'Schro']
dat = dat[dat['session'] !='m53s127']

# dat = dat[dat['session'] !='m53s127']

# select the density only session
density_sess = np.unique(dat['session'][dat['manipulation type']=='density'])
bl_session = np.zeros(dat.shape[0],dtype=bool)
for sess in density_sess:
    bl_session[dat['session']==sess] = True
dat = dat[bl_session]
del bl_session


dat = dat[(dat['manipulation type'] == 'all') | (dat['manipulation type'] == 'odd') | (dat['manipulation type'] == 'density')]

sess_list = np.unique(dat['session'])

dat_tmp = dat[dat['session'] == sess_list[JOB]]

# select the significant


fmt = []
for kk in range(len(dat_tmp.dtype)):
    fmt+=[dat_tmp.dtype[kk]]
fmt+=[object,object]
names = np.hstack((dat_tmp.dtype.names, ['t'],['y']))

res_mat = np.zeros(dat_tmp.shape[0],dtype={'names':names,'formats':fmt})

cntRow = 0
for row in dat_tmp:
    print(cntRow,dat_tmp.shape[0])
    for nm in dat_tmp.dtype.names:
        res_mat[cntRow][nm] = row[nm]



    unit_rec = row['receiver unit id']

    unit_sen = row['sender unit id']
    session = row['session']
    cond = row['manipulation type']
    value = row['manipulation value']


    with open(os.path.join(fld_dill % session,
                           'fit_results_%s_c%d_%s_%.4f.dill' % (session, unit_rec, cond, value)), 'rb') as fh:
        res_dict = dill.load(fh)

    full_gam = res_dict['full']
    red_gam = res_dict['reduced']

    var = 'neu_%d' % unit_sen
    dim_kern = full_gam.smooth_info[var]['basis_kernel'].shape[0]
    knots_num = full_gam.smooth_info[var]['knots'][0].shape[0]
    ord_ = full_gam.smooth_info[var]['ord']
    idx_select = np.arange(0, dim_kern, (dim_kern + 1) // knots_num)

    impulse = np.zeros(dim_kern)
    impulse[(dim_kern - 1) // 2] = 1
    xx = 0.006 * np.linspace(-(dim_kern - 1) / 2, (dim_kern - 1) / 2, dim_kern)
    fX, fX_p_ci, fX_m_ci = full_gam.smooth_compute([impulse], var, perc=0.99,
                                                   trial_idx=None)  #

    fX = fX[xx>0] - fX[xx==0]
    res_mat[cntRow]['t'] = xx[xx>0]
    res_mat[cntRow]['y'] = fX

    cntRow += 1

# create the within area coupling

sel = res_mat['sender brain area'] == res_mat['receiver brain area']

dict_tun = {}
dict_tun[('odd', 0)] = res_mat[sel][(res_mat[sel]['manipulation type'] == 'odd') & (res_mat[sel]['manipulation value'] == 0)]
dict_tun[('odd', 1)] = res_mat[sel][(res_mat[sel]['manipulation type'] == 'odd') & (res_mat[sel]['manipulation value'] == 1)]
dict_tun[('density',0.005)] = res_mat[sel][(res_mat[sel]['manipulation type'] == 'density') &
                                         (res_mat[sel]['manipulation value'] == 0.005)]

dict_tun[('density', 0.0001)] = res_mat[sel][(res_mat[sel]['manipulation type'] == 'density') &
                                         (res_mat[sel]['manipulation value'] == 0.0001)]

tmp = res_mat[sel]
tmp = tmp[tmp['manipulation type'] == 'all']

names = np.array(names)
sel2 = (names!='manipulation type') & (names !='manipulation value') & (names!='t') & (names!='y')

names_sub = np.hstack((names[sel2],['is sign density 0.0001','is sign density 0.0050',
                       'is sign odd 1','is sign odd 0']))
fmt_sub = np.hstack((np.array(fmt)[sel2],[bool]*4))
info = np.zeros(tmp.shape[0],dtype={'names':names_sub,'formats':fmt_sub})
cond_list = [('all',True),('odd', 0), ('odd',1), ('density',0.0001),('density',0.005)]
tunings = np.zeros((info.shape[0],5,6)) * np.nan

cc = 0
for row in tmp:
    print(cc,tmp.shape[0])
    sender = row['sender unit id']
    receiver = row['receiver unit id']
    for nm in info.dtype.names:
        if 'is sign ' in nm:
            continue
        info[cc][nm] = row[nm]

    tunings[cc,0,:] = row['y']
    kk = 1
    for cond,val in cond_list[1:]:
        bl = (dict_tun[(cond,val)]['sender unit id'] == sender) & (dict_tun[(cond,val)]['receiver unit id'] == receiver)
        SUM = bl.sum()
        if SUM == 0:
            continue
        elif SUM > 1:
            raise ValueError
        tunings[cc, kk, :] = dict_tun[(cond,val)]['y'][bl][0]

        kk += 1

        if cond =='density':
            ff = '%.4f'
        else:
            ff = '%d'
        info[cc]['is sign %s '%cond + ff%val] = dict_tun[(cond,val)][bl]['is significant']
    cc += 1
np.save('%s_density_coupling_filters.npy'%session,res_mat)


non_nan = ~np.isnan(tunings.sum(axis=(1,2)))
tunings = tunings[non_nan]
info = info[non_nan]

np.savez('%s_paied_coupliing_filt.npz'%session,tunings=tunings,info=info,cond_list=cond_list)

# sgn = tunings[info['is significant']]
# nsgn = tunings[~info['is significant']]
# un = 15
# ax_1 = plt.subplot(121)
# ax_2 = plt.subplot(122)
# for k in range(5):
#
#     ax_1.plot(sgn[un,k,:])
#     ax_1.set_title('significant')
#     ax_2.plot(nsgn[un,k,:])
#     ax_2.set_title('non significant')
