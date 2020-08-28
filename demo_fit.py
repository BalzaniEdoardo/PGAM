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
np.random.seed(4)



with open('spat_and_temp_filt.dill','rb') as fh:
    dict_tuning = dill.load(fh)



## inputs parameters
num_events = 6000
time_points = 3*10**5 # 30 mins at 0.006 ms resolution
rate = 5.* 0.006 # Hz rate of the final kernel
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

log_mu0 = np.convolve(events, filter_used_conv, mode='same') + resp_func(XT)
# set mean rate
const = np.log(np.mean(np.exp(log_mu0))/rate)
log_mu0 = log_mu0 - const

# generate spikes
spk = np.random.poisson(np.exp(log_mu0))


# create handler of P-splines
sm_handler = smooths_handler()

kern_dir = -1 # post event causality
int_knots = np.linspace(0., 15, int_knots_num) # internal knots
sm_handler.add_smooth('temporal', [events], ord=order, knots=[int_knots],
                      penalty_type='der', der=2, kernel_length=165,
                      kernel_direction=kern_dir,trial_idx=np.ones(time_points),
                      is_temporal_kernel=True, time_bin=0.006,
                      event_input=True,lam=10)


# add spatial variable and nuisance
int_knots =np.linspace(-5,5,int_knots_num)
sm_handler.add_smooth('spatial', [XT], ord=4, knots=[int_knots],
                          penalty_type='der', der=2,
                          is_temporal_kernel=False,lam=10)
sm_handler.add_smooth('spatial_nuis', [XN], ord=4, knots=[int_knots],
                      penalty_type='der', der=2,
                      is_temporal_kernel=False,lam=10)





print('firing hz:',spk.mean()/0.006)



# Fit a gam
link = deriv3_link(sm.genmod.families.links.log())
poissFam = sm.genmod.families.family.Poisson(link=link)
family = d2variance_family(poissFam)

gam_model = general_additive_model(sm_handler,sm_handler.smooths_var,spk,
                                   poissFam,fisher_scoring=False)
full,reduced = gam_model.fit_full_and_reduced(sm_handler.smooths_var,th_pval=0.001,
                                              smooth_pen=None, max_iter=10 ** 3, tol=10 ** (-8),
                                              conv_criteria='deviance',
                                              initial_smooths_guess=False,
                                              method='L-BFGS-B',
                                              gcv_sel_tol=10 ** (-13),
                                              use_dgcv=True,
                                              fit_initial_beta=True,
                                              trial_num_vec=np.ones(time_points))

plt.figure(figsize=[8,6])

# plot the basis set
ax1 = plt.subplot(221)
# temporal basis evaluation convolve x with the basis
xx = np.zeros(165)
xx[84] = 1
time = np.arange(-82,83)*0.006
fX = reduced.eval_basis([xx],'temporal',trial_idx=None).toarray()
plt.title('temporal basis')
plt.plot(time, fX)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = plt.subplot(222)
xx = np.linspace(-5,5,1000)
fX = reduced.eval_basis([xx],'spatial').toarray()
plt.title('spatial basis')
plt.plot(xx, fX)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)



# plot results
ax3 = plt.subplot(223)
plt.title('temporal')
filter_used = dict_tuning['temporal']['filteronly']
time = np.arange(len(filter_used)) * (-0.006)
time = time[::-1] + 0.5
keep = time < 0
kernel_length = 165
impulse = np.zeros(kernel_length)
impulse[(kernel_length-1)//2] = 1
fX,fX_p_ci,fX_m_ci = reduced.smooth_compute([impulse],'temporal',perc=0.99)
fX = fX[keep]
fX_p_ci = fX_p_ci[keep]
fX_m_ci = fX_m_ci[keep]
filter_used = filter_used[keep]
time = time[keep]
fX = fX[:-1]
fX_p_ci = fX_p_ci[:-1]
fX_m_ci = fX_m_ci[:-1]
filter_used = filter_used[:-1]
time = time[:-1]
interc1 = np.nanmedian(fX[-30:]-filter_used[-30:])
plt.plot(time,filter_used,'k',label='True')
plt.plot(time,fX-interc1,color='r', label='GAM')
plt.fill_between(time, fX_m_ci - interc1, fX_p_ci - interc1,alpha=0.3,color='r')
plt.xlabel('time(ms)')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)


ax4 = plt.subplot(224)
xx = np.linspace(-5,5, 1000)
fX,fX_p_ci,fX_m_ci = reduced.smooth_compute([xx],'spatial',perc=0.99)
interc1 = np.nanmedian(fX-resp_func(xx))
plt.plot(xx,resp_func(xx),'k',label='True')

plt.plot(xx,fX-interc1,color='r', label='GAM')
plt.fill_between(xx, fX_m_ci - interc1, fX_p_ci- interc1,alpha=0.3,color='r')

plt.title('spatial')
# nuisance
fX, fX_p_ci, fX_m_ci = full.smooth_compute([xx], 'spatial_nuis', perc=0.99)

plt.plot(xx, fX, color=(125 / 255.,) * 3, label='nuisance')
plt.fill_between(xx, fX_m_ci, fX_p_ci, color=(125 / 255.,) * 3, alpha=0.3)

plt.legend(frameon=False)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
plt.xlabel('x')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])



