import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from scipy.interpolate import griddata
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import os,dill,sys,inspect

thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))


def select_var(dat,sender,receiver,var):
    sel = (dat['sender brain area'] == sender) & (dat['receiver brain area'] == receiver)
    return dat[var][sel]

# select some significant coupling cross-area
dat = np.load('coupling_info.npy', allow_pickle=True)
dat = dat[(dat['pseudo-r2']>=0.008)]

# monkey:
dat = dat[dat['monkey'] == 'Schro']
dat = dat[dat['session'] !='m53s127']

dat = dat[dat['session'] !='m53s127']

# select the density only session
density_sess = np.unique(dat['session'][dat['manipulation type']=='density'])
bl_session = np.zeros(dat.shape[0],dtype=bool)
for sess in density_sess:
    bl_session[dat['session']==sess] = True
dat = dat[bl_session]
del bl_session


# dat = dat[dat['manipulation type'] == 'all']
dat = dat[dat['sender brain area'] != dat['receiver brain area']]
plt.figure()
plt.subplot(121)
cs = dat['area under filter']
cs = np.log(cs + 1 - np.min(cs))
# Y + 1 - min(Y)
plt.hist(cs,range=(np.nanpercentile(cs,2), np.nanpercentile(cs,98)),bins=40,density=True)

plt.subplot(122)
cov = dat['log |cov|']
plt.hist(cov,range=(np.nanpercentile(cov,0.1), np.nanpercentile(cov,99.9)),bins=40,density=True)



plt.figure()
plt.subplot(121)
plt.title('coupling strength')
cs_all = dat['area under filter']#np.log(dat['area under filter'] + 1 - np.min(dat['area under filter']))

cs = cs_all[dat['is significant']]
# Y + 1 - min(Y)
plt.hist(cs,range=(np.nanpercentile(cs,2), np.nanpercentile(cs,98)),bins=40,density=True,label='sign',alpha=0.4)

cs = cs_all[~dat['is significant']]
# Y + 1 - min(Y)
plt.hist(cs,range=(np.nanpercentile(cs,2), np.nanpercentile(cs,98)),bins=40,density=True,label='not sign',alpha=0.4)


plt.subplot(122)
plt.title('$\log |cov_{\\beta}|$')
cov = dat['log |cov|'][dat['is significant']]

plt.hist(cov,range=(np.nanpercentile(cov,0.1), np.nanpercentile(cov,99.9)),bins=40,density=True,label='sign',alpha=0.4)
cov = dat['log |cov|'][~dat['is significant']]

plt.hist(cov,range=(np.nanpercentile(cov,0.1), np.nanpercentile(cov,99.9)),bins=40,density=True,label='not sign',alpha=0.4)
plt.legend()

plt.savefig('effect_testing.pdf')
# # significant coupling only
# dat = dat[dat['is significant']]


tmp= dat[dat['is significant']]
tmp = tmp[tmp['log |cov|']>-75]
X = np.zeros((tmp['is significant'].sum(),2))
X[:, 0] = tmp['area under filter'][tmp['is significant']]
X[:, 1] = tmp['log |cov|'][tmp['is significant']]

dict_params = {'n_components':[10,20,25, 30,100]}
# model = GaussianMixture()
# clf = GridSearchCV(model,param_grid=dict_params)
# cv_res = clf.fit(X)


#
# H,xe,ye = np.histogram2d(X[:,0],X[:,1],range=[[np.percentile(X[:,0],2),np.percentile(X[:,0],99)],
#                                     [np.percentile(X[:,1],1),np.percentile(X[:,1],99)]],bins=100,density=True)
#
# Xe,Ye = np.meshgrid(xe[:-1],ye[:-1])
# plt.plot()
# # Z = np.dot(cv_res.best_estimator_.predict_proba(mXY),cv_res.best_estimator_.weights_)
# # griddata(mXY, Z, xi, method='linear', fill_value=nan, rescale=False)
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1,projection='3d')
# surf = ax.plot_surface(Xe, Ye, H, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

xe = np.linspace(np.percentile(X[:,0],2),np.percentile(X[:,0],99),6)
ye = np.linspace(np.percentile(X[:,1],2),np.percentile(X[:,1],99),6)

plt.figure()
plt.subplot(121)
plt.hist(X[:,0],bins=100,range=[np.nanpercentile(X[:,0],1),np.nanpercentile(X[:,0],99)])
plt.plot(xe,[0]*len(xe),'or')
plt.subplot(122)
plt.hist(X[:,1],bins=100,range=[np.nanpercentile(X[:,1],1),np.nanpercentile(X[:,1],99)])
plt.plot(ye,[0]*len(xe),'or')

dat_tmp = tmp[tmp['is significant']]
for k in range(xe.shape[0]-1):
    for h in range(ye.shape[0] - 1):
        x0 = xe[k]
        x1 = xe[k+1]

        y0 = ye[h]
        y1 = ye[h + 1]
        sel = (dat_tmp['area under filter'] >= x0) & (dat_tmp['area under filter'] < x1)
        sel = sel & (dat_tmp['log |cov|'] >= y0) & (dat_tmp['log |cov|'] < y1)

        dat_cond = dat_tmp[sel]
        dat_cond = dat_cond[np.random.permutation(dat_cond.shape[0])]

        plt.figure(figsize=(12, 8))
        plt.suptitle('CS: %.3f-%.3f    VAR: %.3f-%.3f'%(x0,x1,y0,y1))
        for kk in range(1, min(25,dat_cond.shape[0])):

            unit_rec = dat_cond[kk-1]['receiver unit id']
            unit_sen = dat_cond[kk-1]['sender unit id']
            session = dat_cond[kk-1]['session']
            cond = 'all'
            value = 1

            plt.subplot(5, 5, kk)
            fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
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
                                                           trial_idx=None)  # full_gam.['neu_%d'%unit_sen]

            idx = np.where(xx >= -0.024)[0]

            i_0 = np.where(red_gam.covariate_significance['covariate']==var)[0][0]
            pp, = plt.plot(-xx[idx][::-1], fX[idx][::-1],label='%.3f'%red_gam.covariate_significance['p-val'][i_0])
            plt.fill_between(-xx[idx][::-1], fX_m_ci[idx][::-1], fX_p_ci[idx][::-1], color=pp.get_color(), alpha=0.4)
            plt.title('%s %d->%d' % (dat_cond[kk-1]['session'], dat_cond[kk-1]['sender unit id'],
                                     dat_cond[kk - 1]['receiver unit id']), fontsize=8)
            if kk > 20:
                plt.xlabel('time [sec]')

            if kk % 5 == 1:
                plt.ylabel('gain')
        # plt.suptitle('PFC->MST coupling filter examples')
            plt.legend(fontsize=6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('/Users/edoardo/Work/Code/GAM_code/analyzing/coupling/coupling_category/strength_%d_var_%d.pdf'%(h,k))

        plt.close('all')


fmt = []
for kk in range(len(dat_tmp.dtype)):
    fmt+=[dat_tmp.dtype[kk]]
fmt+=[object,object]
names = np.hstack((dat_tmp.dtype.names, ['t'],['y']))

res_mat = np.zeros(dat_tmp.shape[0],dtype={'names':names,'formats':fmt})

cntRow = 0
for row in dat_tmp:
    for nm in dat_tmp.dtype.names:
        res_mat[cntRow][nm] = row[nm]



    unit_rec = row['receiver unit id']

    unit_sen = row['sender unit id']
    session = row['session']
    cond = 'all'
    value = 1

    fld_dill = '/Volumes/WD_Edo/firefly_analysis/LFP_band/GAM_fit_with_acc/gam_%s'
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
    fX, fX_p_ci, fX_m_ci = red_gam.smooth_compute([impulse], var, perc=0.99,
                                                   trial_idx=None)  #

    fX = fX[xx>0] - fX[xx==0]
    res_mat[cntRow]['t'] = xx[xx>0]
    res_mat[cntRow]['y'] = fX

    cntRow += 1


np.save('between_area_coupling_filters.npy',res_mat)