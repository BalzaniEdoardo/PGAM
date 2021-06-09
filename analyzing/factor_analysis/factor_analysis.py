import numpy as np
import scipy.stats as sts
import matplotlib.pylab as plt
from statsmodels.api import OLS

from sklearn.decomposition import FactorAnalysis

def posterior_mean_and_cov(y, C, diagR):
    """

    :param y: T x N
    :param C: N x p
    :param diagR: N
    :return:
    """
    CTRinv = np.dot(C.T, np.diag(1/diagR))
    IpCTRinvC = np.linalg.pinv(np.eye(C.shape[1]) + np.dot(CTRinv,C))
    beta = np.dot(CTRinv,np.eye(C.shape[0]) - np.dot(np.dot(C, IpCTRinvC), CTRinv))
    # covInv = np.linalg.pinv(np.dot(C,C.T) + np.diag(diagR))
    # beta = np.dot(np.dot(C.T, np.diag(1/diagR)),np.eye(C.shape[0])-) #np.dot(C.T, covInv)
    mu = np.dot(beta, y.T)
    cov = np.eye(C.shape[1]) - np.dot(beta, C)
    return mu, cov

def init_param(y, N, k):
    S,D,V = np.linalg.svd(np.cov(y.T))
    C = np.dot(np.dot(S[:, : k], np.diag(D[:k])), V[:k,:k])
    return C, np.ones(N)


def EM_step(y, k, epsi=10**-8, maxiter=10**3):
    C, diagR = init_param(y, y.shape[1], k)
    R = np.diag(diagR)
    N = y.shape[1]
    S = np.cov(y.T)
    loglike = lambda R, C: sts.multivariate_normal.logpdf(y, mean=np.zeros(N), cov=(np.dot(C,C.T) + R)).mean()
    DL = 1
    ll0 = loglike( R, C)
    ll_iter = [ll0]
    itr = 0
    C = C + 0.01
    while DL > epsi and itr < maxiter:

        mu, cov = posterior_mean_and_cov(y, C, diagR)
        eigcov,U = np.linalg.eigh(cov)
        if np.min(eigcov) < 0.001:
            cov = np.dot( U * (eigcov+0.001), U.T)
        if itr % 100 == 0:
            print('iter', itr, DL, ll0)
        delta = np.dot(y.T, mu.T)
        gamma = np.einsum('kt,ht->kh',mu, mu) + y.shape[0]*cov#np.dot(mu, mu.reshape(k, 1)) + N * cov
        C = np.dot(delta, np.linalg.pinv(gamma))
        diagR = np.diag(S) - np.diag(np.dot(C, delta.T/y.shape[0]))
        ll1 = loglike(np.diag(diagR), C)
        DL = np.abs(ll0 - ll1)
        ll0 = ll1
        ll_iter += [ll0]
        itr += 1
    return C, diagR, mu, cov, ll_iter



def mean_yj_given_ymj(y, C, diagR, sqrty,subtract_mean=True):
    predict_mean = np.zeros(y.shape)
    predict_var = np.zeros(y.shape)
    sigma_y = np.dot(C, C.T) + np.diag(diagR)
    if subtract_mean:
        mean_ = y.mean(axis=0)
    else:
        mean_ = np.zeros(y.shape[1])
    M = y.shape[1]
    for j in range(y.shape[1]):
        B_mj = np.eye(M)
        B_mj[j, j] = 0
        B_mj = B_mj[:, np.arange(M) != j]
        B_j = np.zeros((M, 1))
        B_j[j] = 1

        R0j = np.eye(M)
        idx = np.zeros(M, dtype=int)
        idx[:j] = np.arange(1, j + 1)
        idx[j] = 0
        idx[j + 1:] = np.arange(j + 1, M)
        R0j = R0j[idx, :]

        y_j_mj = np.hstack((np.dot(y, B_j), np.dot(y, B_mj)))
        sigma_y_j_mj = np.dot(np.dot(R0j.T, sigma_y), R0j)

        Binv = np.linalg.pinv(sigma_y_j_mj[1:, 1:])
        predict_mean[:, j] = np.dot(np.dot(sigma_y_j_mj[0, 1:], Binv), y_j_mj[:, 1:].T) + mean_[j]
        predict_var[:, j] = sigma_y_j_mj[0,0] - np.dot(np.dot(sigma_y_j_mj[0, 1:], Binv), sigma_y_j_mj[0, 1:])
    predict_error = np.mean((sqrty - predict_mean)**2)
    return predict_mean,predict_var, predict_error



if __name__ == '__main__':

    # p(y) = N(0, Sigma)
    # B
    # y = [y_j,y_mj]


    D = 10
    M = 100





    C = np.random.normal(size=(M,D))**2
    x = np.random.normal(size=(D,15000))
    R = np.eye(M)

    ix,iy = np.diag_indices(50)
    R[ix[:M//2],iy[:M//2]] = 3
    y = np.zeros((x.shape[1],C.shape[0]))
    mean_y = np.random.normal(size=M)
    for t in range(x.shape[1]):
        if t %1000==0:
            print(t,x.shape[1])
        y[t,:] = np.random.multivariate_normal(np.dot(C,x[:,t]), R)

    y = y + mean_y
    CC, diagR, mu, cov,ll_iter = EM_step(y-mean_y, x.shape[0], epsi=10**-6, maxiter=5000)



    S_tr,D,V_tr = np.linalg.svd(C)
    S, _ , V = np.linalg.svd(CC)


    C_tilde = np.dot(np.dot(np.dot(S_tr,np.dot(S.T,CC)),V.T),V_tr)
    R_tilde = np.diag(diagR)

    D = x.shape[0]
    Proj = np.zeros((D,D))
    for k in range(D):
        model = OLS(x[k,:],mu.T)
        fit = model.fit()
        Proj[:,k] = fit.params

    mu_tilde = np.dot(Proj.T,mu)


    plt.figure(figsize=(10,4))
    plt.subplot(131)
    plt.plot(ll_iter)
    plt.title('Log-Likelihood')
    plt.xlabel('iter')
    plt.ylabel('LL')

    plt.subplot(132)
    plt.plot(mu_tilde[2,:100])
    plt.plot(x[2,:100])

    plt.title('Reconstructed Latent')
    plt.xlabel('time')
    plt.ylabel('latent')



    model = FactorAnalysis(n_components=D)
    fit = model.fit(y)

    xx = fit.transform(y)
    Proj2 = np.zeros((D,D))
    for k in range(D):
        model = OLS(x[k,:],xx)
        fitregr = model.fit()
        Proj2[:,k] = fitregr.params

    mu_2 = np.dot(Proj2.T,xx.T)

    plt.subplot(133)
    plt.plot(diagR)
    plt.plot(np.diag(R))
    plt.plot(fit.noise_variance_)

    plt.title('Private Noise')
    plt.xlabel('neuron')
    plt.ylabel('private noise')
    plt.tight_layout()

    loglike = lambda R, C: sts.multivariate_normal.logpdf(y, mean=np.zeros(M), cov=(np.dot(C, C.T) + R)).mean()

    print('SKL',loglike(np.diag(fit.noise_variance_),fit.components_.T), 'EM',ll_iter[-1])

    pred_mu, pred_sigma, predict_error = mean_yj_given_ymj(y, CC, diagR, y)
    plt.figure()
    plt.plot(y[:40,3],label='True')
    p, = plt.plot(pred_mu[:40,3],label='fit')
    plt.fill_between(range(40), pred_mu[:40,3]-1.96*pred_sigma[:40,3],pred_mu[:40,3]+1.96*pred_sigma[:40,3],color=p.get_color(),alpha=0.3)


# predictioi