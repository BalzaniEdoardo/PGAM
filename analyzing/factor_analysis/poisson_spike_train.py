import numpy as np
import scipy.stats as sts
from factor_analysis import *
import sys
from scipy.optimize import minimize
np.random.seed(4)
from copy import deepcopy

sys.path.append('/Users/edoardo/Work/Code/GAM_code/GAM_library')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/preprocessing_pipeline/util_preproc')
sys.path.append('/Users/edoardo/Work/Code/GAM_code/firefly_utils')
import matplotlib.pylab as plt

def spike_smooth(x,trials_idx,filter):
    sm_x = np.zeros(x.shape[0])*np.nan
    for tr in np.unique(trials_idx):
        sel = trials_idx == tr
        tmp = np.convolve(x[sel], filter, mode='same')
        if tmp.shape[0] == sel.sum():
            sm_x[sel] = tmp
    return sm_x

def pop_spike_convolve(spike_mat,trials_idx,filter):
    sm_spk = np.zeros(spike_mat.shape)
    for neu in range(spike_mat.shape[1]):
        sm_spk[:,neu] = spike_smooth(spike_mat[:,neu],trials_idx,filter)
    return sm_spk


D = 10
M = 100

C = np.random.normal(size=(M, D)) ** 2
C = C / 10
x = np.random.normal(size=(D, 55000))
R = np.eye(M)

ix, iy = np.diag_indices(50)
R[ix[:M // 2], iy[:M // 2]] = 2.
y = np.zeros((x.shape[1], C.shape[0]))

mean_y = np.random.uniform(low=1,high=80,size=M) * 0.018
# mean_y = np.random.normal(size=M)
log_FR = np.dot(C, x)
log_FR_adj = np.zeros(log_FR.shape)
C_scaled = np.zeros(C.shape)
for neu in range(M):
    fr = np.exp(log_FR[neu])
    target_fr = mean_y[neu]
    # func = lambda alpha : (np.mean(fr**alpha) - target_fr)**2
    # res = minimize(func, 0.0001, method='Nelder-Mead',tol=10**-8)
    # C_scaled[neu,:] = C[neu,:] * res.x[0]

    ## SCALE MODEL
    const = np.log(np.mean(fr) / target_fr)
    log_FR_adj[neu] = np.log(fr) - const

    ## SCALE C
    # func = lambda alpha : (np.mean(fr**alpha) - target_fr)**2
    # res = minimize(func, 0.0001, method='Nelder-Mead',tol=10**-8)
    # C_scaled[neu,:] = C[neu,:] * res.x[0]
    # log_FR_adj[neu] = np.dot(C_scaled[neu,:],x)

## check the firing properties of the population
# print(np.max(np.exp(log_FR_adj))/0.018)
# print(np.median(np.exp(log_FR_adj))/0.018)
#
# plt.figure(figsize=(12,4))
# lst = []
# for vv in log_FR_adj:
#     lst += [np.exp(vv) / 0.018]
#
# plt.boxplot(lst,showfliers=False)


filtwidth = 5
t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
h = np.exp(-t ** 2 / (2 * filtwidth ** 2))
h = h - h[0]
h = h / np.sum(h)


D = 10

prd_error = np.zeros((5,20))
for k in range(20):
    yt = np.random.poisson(np.exp(log_FR_adj))
    sm_spike = pop_spike_convolve(np.sqrt(yt.T), np.ones(yt.shape[1]), h)
    idx_endTrain = int(0.9*sm_spike.shape[0])
    print(k,'ITER')
    cc=0
    for DD in [2,6,10,14,18]:
        print('latent dim',DD)
        model = FactorAnalysis(n_components=DD)

        fit = model.fit(sm_spike[:idx_endTrain])
        M = yt.shape[0]
        loglike = lambda R, C: sts.multivariate_normal.logpdf(
            sm_spike,
            mean=np.zeros(M), cov=(np.dot(C, C.T) + R)).mean()
        # print('SKL', loglike(np.diag(fit.noise_variance_), fit.components_.T), 'EM', ll_iter[-1])
        pred_mu_skl, pred_sigma_skl, predict_error_skl = mean_yj_given_ymj(
            sm_spike[idx_endTrain:],
            fit.components_.T,
            fit.noise_variance_,
            np.sqrt(yt[:,idx_endTrain:]).T,subtract_mean=True)
        prd_error[cc,k] = deepcopy(predict_error_skl)
        cc+=1
        # ee = np.linalg.eig(np.dot(fit.components_,fit.components_.T))[0]
        # plt.plot(np.sort(ee)[::-1],'-o')

dim_vec = [2,6,10,14,18]


p, = plt.plot( dim_vec,prd_error.mean(axis=1),'-ob')
plt.xlabel('latent dimension')
plt.ylabel('prediction error')
plt.title('cv latent dimensionality estimation')
plt.savefig('cv_estim_latent_dim.png')

# plt.fill_between(dim_vec, prd_error.mean(axis=1)-prd_error.std(axis=1),prd_error.mean(axis=1)+prd_error.std(axis=1),
#                  color=p.get_color(),alpha=0.4)
# print('FILT WIDTH %d'%filtwidth)
#
#
# plt.plot(sm_spike[:400,8])
# plt.plot(pred_mu_skl[:400,8])