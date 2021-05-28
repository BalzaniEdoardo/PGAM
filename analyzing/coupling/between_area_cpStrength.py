import numpy as np
import matplotlib.pylab as plt
from statsmodels.distributions.empirical_distribution import ECDF
import os,inspect,sys,re
print (inspect.getfile(inspect.currentframe()))
thisPath = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(os.path.dirname(thisPath),'GAM_library'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'util_preproc'))
sys.path.append(os.path.join(os.path.dirname(thisPath),'firefly_utils'))
# from spline_basis_toolbox import *
from GAM_library import *
import dill
import pandas as pd
import seaborn as sbn
from copy import deepcopy
from scipy.io import savemat


def select_var(dat,sender,receiver,var):
    sel = (dat['sender brain area'] == sender) & (dat['receiver brain area'] == receiver)
    return dat[var][sel]

# select some significant coupling cross-area
dat = np.load('coupling_info.npy', allow_pickle=True)
dat = dat[(dat['pseudo-r2']>=0.008)]

# monkey:
dat = dat[dat['monkey'] == 'Schro']
dat = dat[dat['session'] !='m53s127']

# select the density only session
density_sess = np.unique(dat['session'][dat['manipulation type']=='density'])
bl_session = np.zeros(dat.shape[0],dtype=bool)
for sess in density_sess:
    bl_session[dat['session']==sess] = True
dat = dat[bl_session]
del bl_session

# significant coupling only
dat = dat[dat['is significant']]

cs_mst_pfc = select_var(dat,'MST','PFC','area under filter')
cs_mst_ppc = select_var(dat,'MST','PPC','area under filter')
cs_ppc_mst = select_var(dat,'PPC','MST','area under filter')
cs_ppc_pfc = select_var(dat,'PPC','PFC','area under filter')
cs_pfc_ppc = select_var(dat,'PFC','PPC','area under filter')
cs_pfc_mst = select_var(dat,'PFC','MST','area under filter')

cdf_mst_pfc = ECDF(cs_mst_pfc)
cdf_mst_ppc = ECDF(cs_mst_ppc)
cdf_ppc_mst = ECDF(cs_ppc_mst)
cdf_ppc_pfc = ECDF(cs_ppc_pfc)
cdf_pfc_ppc = ECDF(cs_pfc_ppc)
cdf_pfc_mst = ECDF(cs_pfc_mst)

xmin = min(np.nanpercentile(cs_mst_pfc,2),np.nanpercentile(cs_mst_ppc,2),
           np.nanpercentile(cs_ppc_mst,2),np.nanpercentile(cs_ppc_pfc,2),
           np.nanpercentile(cs_pfc_ppc,2),np.nanpercentile(cs_pfc_mst,2))
xmax = max(np.nanpercentile(cs_mst_pfc,98),np.nanpercentile(cs_mst_ppc,98),
           np.nanpercentile(cs_ppc_mst,98),np.nanpercentile(cs_ppc_pfc,98),
           np.nanpercentile(cs_pfc_ppc,98),np.nanpercentile(cs_pfc_mst,98))

xx = np.linspace(xmin,xmax,100)


plt.figure()
plt.title('coupling strength')
plt.plot(xx,cdf_mst_pfc(xx),label='MST->PFC')
plt.plot(xx,cdf_mst_ppc(xx),label='MST->PPC')
plt.plot(xx,cdf_ppc_pfc(xx),label='PPC->PFC')
plt.plot(xx,cdf_ppc_mst(xx),label='PPC->MST')
plt.plot(xx,cdf_pfc_mst(xx),label='PFC->MST')
plt.plot(xx,cdf_pfc_ppc(xx),label='PFC->PPC')

plt.legend()

plt.plot([0,0],[0,1],'--k')
plt.plot([xx.min(),xx.max()],[0.5,0.5],'--k')
plt.ylabel('cdf')
plt.xlabel('coupling strength')
plt.savefig('coupling_strength_cdf.pdf')


tot_shape = cs_mst_pfc.shape[0] + cs_mst_ppc.shape[0] + cs_ppc_mst.shape[0] + cs_ppc_pfc.shape[0] + \
        cs_pfc_ppc.shape[0] + cs_pfc_mst.shape[0]

table = np.zeros(tot_shape,dtype={'names':('sender','receiver','coupling strength'),'formats':('U30','U30',float)})
cc = 0
table['sender'][cc: cc+cs_mst_pfc.shape[0]] = 'MST'
table['receiver'][cc: cc+cs_mst_pfc.shape[0]] = 'PFC'
table['coupling strength'][cc: cc+cs_mst_pfc.shape[0]] = cs_mst_pfc
cc+=cs_mst_pfc.shape[0]

table['sender'][cc: cc+cs_mst_ppc.shape[0]] = 'MST'
table['receiver'][cc: cc+cs_mst_ppc.shape[0]] = 'PPC'
table['coupling strength'][cc: cc+cs_mst_ppc.shape[0]] = cs_mst_ppc
cc+=cs_mst_ppc.shape[0]


table['sender'][cc: cc+cs_ppc_mst.shape[0]] = 'PPC'
table['receiver'][cc: cc+cs_ppc_mst.shape[0]] = 'MST'
table['coupling strength'][cc: cc+cs_ppc_mst.shape[0]] = cs_ppc_mst
cc+=cs_ppc_mst.shape[0]

table['sender'][cc: cc+cs_ppc_pfc.shape[0]] = 'PPC'
table['receiver'][cc: cc+cs_ppc_pfc.shape[0]] = 'PFC'
table['coupling strength'][cc: cc+cs_ppc_pfc.shape[0]] = cs_ppc_pfc
cc+=cs_ppc_pfc.shape[0]

table['sender'][cc: cc+cs_pfc_mst.shape[0]] = 'PFC'
table['receiver'][cc: cc+cs_pfc_mst.shape[0]] = 'MST'
table['coupling strength'][cc: cc+cs_pfc_mst.shape[0]] = cs_pfc_mst
cc+=cs_pfc_mst.shape[0]

table['sender'][cc: cc+cs_pfc_ppc.shape[0]] = 'PFC'
table['receiver'][cc: cc+cs_pfc_ppc.shape[0]] = 'PPC'
table['coupling strength'][cc: cc+cs_pfc_ppc.shape[0]] = cs_pfc_ppc

df = pd.DataFrame(table)

plt.figure()
ax=plt.subplot(111)
sbn.pointplot(x="sender",
              y="coupling strength",
              data=df,hue='receiver',estimator=np.median,
              palette={'MST':'g','PPC':'b','PFC':'r'},
              hue_order=['MST','PFC','PPC'],linestyles='none',
              dodge=0.2,ax=ax)
plt.ylabel("coupling strength",fontsize=12)
plt.xlabel('sender',fontsize=12)
plt.title('coupling strength',fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('coupling_strength_pointplot.pdf')


df = pd.DataFrame(table)
df['coupling strength'] = np.abs(df['coupling strength'] )
plt.figure()
ax=plt.subplot(111)
sbn.pointplot(x="sender",
              y="coupling strength",
              data=df,hue='receiver',estimator=np.median,
              palette={'MST':'g','PPC':'b','PFC':'r'},
              hue_order=['MST','PFC','PPC'],linestyles='none',
              dodge=0.2,ax=ax)
plt.ylabel("coupling strength",fontsize=12)
plt.xlabel('sender',fontsize=12)
plt.title('coupling strength',fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('abs_coupling_strength_pointplot.pdf')



plt.figure()
df = pd.DataFrame(table)
df = df[df['coupling strength'] > 0]
ax=plt.subplot(211)

sbn.pointplot(x="sender",
              y="coupling strength",
              data=df,hue='receiver',estimator=np.median,
              palette={'MST':'g','PPC':'b','PFC':'r'},
              hue_order=['MST','PFC','PPC'],linestyles='none',
              dodge=0.2,ax=ax)
plt.ylabel("coupling strength",fontsize=12)
plt.xlabel('sender',fontsize=12)
plt.title('excitatory coupling',fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax._remove_legend(True)
df = pd.DataFrame(table)
df = df[df['coupling strength'] < 0]
ax=plt.subplot(212)

sbn.pointplot(x="sender",
              y="coupling strength",
              data=df,hue='receiver',estimator=np.median,
              palette={'MST':'g','PPC':'b','PFC':'r'},
              hue_order=['MST','PFC','PPC'],linestyles='none',
              dodge=0.2,ax=ax)
plt.ylabel("coupling strength",fontsize=12)
plt.xlabel('sender',fontsize=12)
plt.title('inhibitory coupling',fontsize=12)
ax._remove_legend(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('excit_inhib_coupling.pdf')



plt.figure()
this_table = deepcopy(table)
for k in range(this_table.shape[0]):
    if this_table[k]['coupling strength'] > 0:
        this_table[k]['receiver'] = this_table[k]['receiver'] + ' +'
    else:
        this_table[k]['receiver'] = this_table[k]['receiver'] + ' -'

df = pd.DataFrame(this_table)
ax=plt.subplot(111)

sbn.pointplot(x="sender",
              y="coupling strength",
              data=df,hue='receiver',estimator=np.median,
              palette={'MST +':'g','PPC +':'b','PFC +':'r','MST -':'g','PPC -':'b','PFC -':'r'},
              hue_order=['MST +','MST -','PFC +','PFC -','PPC +','PPC -'],linestyles='none',
              dodge=0.1,ax=ax)
plt.ylabel("coupling strength",fontsize=12)
plt.xlabel('sender',fontsize=12)
plt.title('excitatory coupling',fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax._remove_legend(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('excit_inhib_coupling_joint.pdf')


savemat('coupling_strength.mat',mdict={'coupling_strength':table})