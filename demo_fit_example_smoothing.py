import scipy.stats as sts
import numpy as np
import matplotlib.pylab as plt
import sys,inspect,os
path = os.path.join(os.path.dirname( inspect.getfile(inspect.currentframe())),'GAM_library')
sys.path.append(path)
print(path)
from GAM_library import *
from gam_data_handlers import *
import dill
import statsmodels.api as sm
from spline_basis_toolbox import *
from copy import deepcopy
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


np.random.seed(5)



with open('spat_and_temp_filt.dill','rb') as fh:
    dict_tuning = dill.load(fh)



## inputs parameters
num_events = 5000
time_points = 1*10**5 # 30 mins at 0.006 ms resolution
rate = 1.5 * 0.006 # Hz rate of the final kernel
variance = 5. # spatial input and nuisance variance
corr = 0.7 # spatial input and nuisance correlation
int_knots_num = 20 # num of internal knots for the spline basis
order = 4 # spline order

## create temporal input
idx = np.random.choice(np.arange(time_points),num_events,replace=False)
events = np.zeros(time_points)
events[idx] = 1

# create spatial input
a = variance * (1 + corr)
b = variance * (1 - corr)
D = np.diag([a, b])
R = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])

COV = np.dot(np.dot(R, D), R.T)
rv = sts.multivariate_normal(mean=[0,0], cov=COV)
samp = rv.rvs(time_points)
XT = samp[:, 0]
XN = samp[:,1]

# truncate X to avoid jumps in the resp function
sele_idx = np.abs(XT) < 5
XT = XT[sele_idx]
XN = XN[sele_idx]
while XT.shape[0] < time_points:
    tmpX = rv.rvs(10 ** 4)
    sele_idx = np.abs(tmpX[:,0]) < 5
    tmpX = tmpX[sele_idx, :]

    XT = np.hstack((XT, tmpX[:,0]))
    XN = np.hstack((XN, tmpX[:, 1]))
XT = XT[:time_points]
XN = XN[:time_points]
print('correlation true vs nusiance',sts.pearsonr(XT,XN)[0])


# set firing rate
filter_used_conv = dict_tuning['temporal']['zeropad'] # temporal filter (vector)
resp_func = dict_tuning['spatial'] # funciton

log_mu0 =  resp_func(XT)#np.convolve(events, filter_used_conv, mode='same') +
# set mean rate
const = np.log(np.mean(np.exp(log_mu0))/rate)
log_mu0 = log_mu0 - const

# generate spikes
spk = np.random.poisson(np.exp(log_mu0))


# create handler of P-splines

kern_dir = -1 # post event causality
int_knots = np.linspace(0., 15, int_knots_num) # internal knots
# sm_handler.add_smooth('temporal', [events], ord=order, knots_num=30,
#                       penalty_type='der', der=2, kernel_length=165,
#                       kernel_direction=kern_dir,trial_idx=np.ones(time_points),
#                       is_temporal_kernel=True, time_bin=0.006,
#                       event_input=True,lam=5*10**(-8))


# add spatial variable and nuisance




int_knots =np.linspace(-5,5,int_knots_num)

cmap = plt.get_cmap('Greys')

lambdas = 10**np.array([-6,-3,-2,10**-1,-0.5,0,0.5,1,2,3,6])
clint  = np.linspace(0.1,1,len(lambdas))
dict_response = {}

cccp = 0
first = True
for lam in lambdas:
    plt.figure(figsize=[6,6*0.83])
    
    
    
    ax4 = plt.subplot(111)
    sm_handler = smooths_handler()

    sm_handler.add_smooth('spatial', [XT], ord=4, knots=[int_knots],
                              penalty_type='der', der=2,
                              is_temporal_kernel=False,lam=lam)
    # sm_handler.add_smooth('spatial_nuis', [XN], ord=4, knots=[int_knots],
    #                       penalty_type='der', der=2,
    #                       is_temporal_kernel=False,lam=5*10**(-6))
    
    
    print('firing hz:',spk.mean()/0.006)
    
    
    
    # Fit a gam
    
    
    link = deriv3_link(sm.genmod.families.links.log())
    poissFam = sm.genmod.families.family.Poisson(link=link)
    family = d2variance_family(poissFam)
    
    gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,spk,
                                       poissFam,fisher_scoring=False)
    
    
    reduced = gam_model.optim_gam(sm_handler.smooths_var,
                                                  smooth_pen=None, max_iter=0, tol=10 ** (-8),
                                                  conv_criteria='deviance',
                                                  initial_smooths_guess=False,
                                                  method='L-BFGS-B',
                                                  gcv_sel_tol=10 ** (-13),
                                                  use_dgcv=True,
                                                  fit_initial_beta=True,
                                                  perform_PQL=False,
                                                 
                                                  )
    
    xx = np.linspace(-5,5, 1000)
    fX,fX_p_ci,fX_m_ci = reduced.smooth_compute([xx],'spatial',perc=0.99)
    interc1 = reduced.beta[0]
   
    
    plt.plot(xx,fX-interc1,color='k', label='lam %e'%lam,lw=4)
    
    # plt.fill_between(xx, fX_m_ci - interc1, fX_p_ci- interc1,alpha=0.3,color='r')
    cccp+=1
    # nuisance
    
    
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    plt.xlabel('$x$',fontsize=20)
    plt.ylabel('$f(x)$',fontsize=20)
    plt.yticks([])
    plt.xticks([])
    if first:
        first=False
        ylim = plt.ylim()
    plt.ylim(ylim)
    dict_response[lam] = {'x':xx,'fX':fX,'fX_p_ci':fX_p_ci,'fX_m_ci':fX_m_ci,'interc':interc1}
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig('/Users/edoardo/Desktop/tmp_figs/smoothing_constant_%e.pdf'%lam)

# sm = plt.cm.ScalarMappable(cmap=cmap)
# sm.set_array([])
# cbar = plt.colorbar(sm, ticks=np.linspace(0,1,2), 
#              )
# cbar.ax.set_yticklabels(['0','$\infty$'],fontsize=20)

plt.figure()
for lam in 10.**np.array([-6,1,2,6]):
    plt.plot(dict_response[lam]['x'],dict_response[lam]['fX'])
    
lam = 10 
cmap = plt.get_cmap('RdYlBu_r')
plt.figure(figsize=[8,6])
ax = plt.subplot(111)
levs = np.linspace(0,1,reduced.beta.shape[0]-3)
for kk in range(3,reduced.beta.shape[0]-3):
    ek = np.zeros(reduced.beta.shape[0])
    ek[kk]=1
    sm_handler_tst = smooths_handler()

    sm_handler_tst.add_smooth('spatial', [xx], ord=4, knots=[int_knots],
                              penalty_type='der', der=2,
                              is_temporal_kernel=False,lam=lam)
    X = sm_handler_tst['spatial'].X.toarray()
    cols = np.array(cmap(levs[kk]))
    cols[:-1] = 0.8*cols[:-1]
    plt.plot(xx,np.dot(X,ek),color=cols,lw=2)

plt.xlabel('$x$',fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/Users/edoardo/Desktop/tmp_figs/nice_colors_basis.pdf')