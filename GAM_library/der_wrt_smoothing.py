import sys,inspect,os

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
sys.path.append(path)

import numpy as np
import scipy.linalg as linalg
from copy import deepcopy
import statsmodels.api as sm
import scipy.stats as sts
from scipy.special import erfinv
import inspect
from scipy.optimize import minimize
from time import perf_counter
from deriv_det_Slam import *
from newton_optim import *
from numpy.core.umath_tests import inner1d
from gam_data_handlers import *
try:
    import fast_summations
except:
    pass
from opt_einsum import contract
import matplotlib.pylab as plt
# from GAM_library import GAM_result,general_additive_model




class d2variance_family(sm.genmod.families.Family):
    def __init__(self,family,run_tests=True):
        self.__class__ = family.__class__

        for name,method in inspect.getmembers(family):
            if name.startswith('__') and name.endswith('__'):
                continue
            self.__setattr__(name,method)
            if name == 'variance':
                self.variance.deriv2 = lambda x: variance_deriv2(self,x)
                self.variance.deriv3 = lambda x: variance_deriv3(self,x)

        self.__link = family.link
        self.deriv3 = lambda x: link_deriv3(self,x)
        epsi = 10 ** -4
        func = lambda x: self.variance.deriv(x)
        func1 = lambda x: self.variance.deriv2(x)
        self.variance.approx_deriv2 = lambda x: (func(x + epsi) - func(x - epsi)) / (2 * epsi)
        if run_tests:
            if isinstance(self,sm.genmod.families.family.Tweedie):
                x = np.random.uniform(-1,1,size=20)
            else:
                x = np.random.uniform(0.1,0.9,size=10)
            approx_der = (func(x + epsi) - func(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der, self.variance.deriv2(x)):
                print('\n2rd order variance deriv possibly wrong for %s\n' % self.__class__)
            else:
                print('2nd order deriv of variance function is ok!')
            approx_der2 = (func1(x + epsi) - func1(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der2, self.variance.deriv3(x)):
                print('\n3rd order variance deriv possibly wrong for %s\n' % self.__class__)
            else:
                print('3rd order deriv of variance function is ok!')


#


class deriv3_link(sm.genmod.families.links.Link):
    def __init__(self, link, run_tests=True):
        self.__class__ = link.__class__

        for name, method in inspect.getmembers(link):
            if name.startswith('__') and name.endswith('__'):
                continue
            self.__setattr__(name, method)

        self.__link = link
        self.deriv3 = lambda x: link_deriv3(self, x)
        self.deriv4 = lambda x: link_deriv4(self, x)

        epsi = 10 ** -4
        func = lambda x: self.deriv2(x)
        func1 = lambda x: self.deriv3(x)
        self.approx_deriv3 = lambda x: (func(x + epsi) - func(x - epsi)) / (2 * epsi)
        if run_tests:
            x = np.random.uniform(0.1, 0.9, size=10)
            approx_der = (func(x + epsi) - func(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der, self.deriv3(x)):
                print('\n3rd order deriv possibly wrong for %s\n' % self.__class__)
            else:
                print('3rd order deriv of link function is ok!')
            approx_der1 = (func1(x + epsi) - func1(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der1, self.deriv4(x)):
                print('\n4th order deriv possibly wrong for %s\n' % self.__class__)
            else:
                print('4th order deriv of link function is ok!')

    def __call__(self, x):
        return self.__link(x)

def link_deriv3(self,x):
    if isinstance(self,sm.genmod.families.links.identity):
        return np.zeros(shape=x.shape)
    elif isinstance(self,sm.genmod.families.links.log):
        return 2/x**3
    elif isinstance(self,sm.genmod.families.links.logit):
        return -((-2 + 6 * x - 6 * x**2) /((1 - x)**3 * x**3))
    elif isinstance(self,sm.genmod.families.links.inverse_power):
        return -6/x**4
    elif isinstance(self,sm.genmod.families.links.probit):
        return 2 * np.sqrt(2) * (np.pi**(3/2)) * np.exp(3 * erfinv(-1 + 2 * x)**2) * (1 + 4 * erfinv(-1 + 2*x)**2)
    else:
        raise NotImplementedError('deriv3 not implemented for %s'%self.__class__)

def link_deriv4(self,x):
    if isinstance(self,sm.genmod.families.links.identity):
        return np.zeros(shape=x.shape)
    elif isinstance(self,sm.genmod.families.links.log):
        return -6/x**4
    elif isinstance(self,sm.genmod.families.links.logit):
        return 6*(4*x**3 -6*x**2 + 4*x -1) / (((1-x)**4) * x**4)
    elif isinstance(self,sm.genmod.families.links.inverse_power):
        return 24/x**5
    elif isinstance(self,sm.genmod.families.links.probit):
        return 4 * np.sqrt(2) * (np.pi**2) * np.exp(4*erfinv(2*x-1)**2) * erfinv(2*x-1)*(12*erfinv(2*x-1)**2+7)
    else:
        raise NotImplementedError('deriv3 not implemented for %s'%self.__class__)


def variance_deriv2(family_object, x):
    if isinstance(family_object,sm.genmod.families.family.Poisson):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Gamma):
        return 2*np.ones(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Binomial):
        return -2*np.ones(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Gaussian):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.InverseGaussian):
        return 6*x
    elif isinstance(family_object,sm.genmod.families.family.Tweedie):
        k = family_object.var_power
        res = k*(k - 1) * np.fabs(x) ** (k - 2)
        return res
    else:
        raise NotImplementedError('deriv3 not implemented for %s'%family_object.__class__)

def variance_deriv3(family_object, x):
    if isinstance(family_object,sm.genmod.families.family.Poisson):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Gamma):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Binomial):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Gaussian):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.InverseGaussian):
        return 6*np.ones(shape=x.shape)
    elif isinstance(family_object,sm.genmod.families.family.Tweedie):
        k = family_object.var_power
        res = k*(k - 1)*(k-2) * np.fabs(x) ** (k - 3)
        ii = np.flatnonzero(x < 0)
        res[ii] = -1*res[ii]
        return res
    else:
        raise NotImplementedError('deriv3 not implemented for %s'%family_object.__class__)


def create_Slam(rho, sm_handler, var_list):
    S_list = compute_Sjs(sm_handler,var_list)
    S_tens = np.zeros((len(S_list),)+S_list[0].shape)
    S_tens[:,:,:] = S_list
    Slam = np.einsum('i,ikl->kl',np.exp(rho),S_tens)
    return Slam

# def compute_Sjs(sm_handler,var_list):
#
#     S_all = []
#     tot_dim = 1
#     ii = 0
#     if len(var_list)> 1:
#         ii = 1
#     for var in var_list:
#         tot_dim += sm_handler[var].X.shape[1] - ii
#
#     cc = 1
#     for var in var_list:
#         dim = sm_handler[var].dim
#
#         for k in range(dim):
#
#             S = np.zeros((tot_dim,tot_dim))
#             Sk = sm_handler[var].S_list[k]
#             shapeS = Sk.shape[0]
#             Sk = Sk[:shapeS-ii,:shapeS-ii]
#             S[cc: cc+Sk.shape[0], cc:cc+Sk.shape[0]] = Sk
#         S_all += [S]
#         cc += Sk.shape[0]
#
#     return S_all

def compute_Sall(sm_handler, var_list):
    Sall = []
    for var_name in var_list:
        S_list = sm_handler[var_name].S_list
        for S in S_list:
            Sall += [S]


    return Sall

def penalty_ll(rho,beta,sm_handler,var_list,phi_est):
    Slam = create_Slam(rho, sm_handler, var_list)
    penalty = -0.5 / phi_est * np.dot(np.dot(beta,Slam),beta)
    return penalty

def penalty_ll_Slam(Slam,beta,phi_est):
    # Slam = create_Slam(rho, sm_handler, var_list)
    penalty = -0.5 / phi_est * np.dot(np.dot(beta,Slam),beta)
    return penalty

def dbeta_penalty_ll(rho,beta,sm_handler,var_list,phi_est):
    Slam = create_Slam(rho, sm_handler, var_list)
    grad = -1/phi_est * np.dot(Slam,beta)
    return grad

def dbeta_penalty_ll_Slam(Slam,beta,phi_est):
    grad = -1/phi_est * np.dot(Slam,beta)
    return grad

def d2beta_penalty_ll(rho,beta,sm_handler,var_list,phi_est):
    Slam = create_Slam(rho, sm_handler, var_list)
    grad = -1/phi_est * Slam
    return grad

def d2beta_penalty_ll_Slam(Slam,beta,phi_est):
    grad = -1/phi_est * Slam
    return grad


def unpenalized_ll(beta,y,X,family,phi_est,omega=1):
    mu = family.link.inverse(np.dot(X,beta))
    ll = family.loglike(y, mu, var_weights=omega, scale=phi_est)
    return ll

def dbeta_unpenalized_ll(beta,y,X,family,phi_est):
    mu = family.link.inverse(np.dot(X,beta))
    vector = (y-mu)/(family.link.deriv(mu) * family.variance(mu))
    grad_unp_ll = 1./phi_est * np.dot(vector,X)
    return grad_unp_ll


def d2beta_unpenalized_ll(beta,y,X,family,phi_est):
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, beta))
    dy = y - mu
    corr = family.variance.deriv(mu) / family.variance(mu) + family.link.deriv2(mu) / family.link.deriv(mu)
    alpha = 1 + dy * np.clip(corr, FLOAT_EPS, np.inf)
    mu = family.link.inverse(np.dot(X, beta))
    dmu_deta = np.clip(1/family.link.deriv(mu) ,FLOAT_EPS,np.inf)
    w = alpha * dmu_deta** 2 /  family.variance(mu)
    if any(np.abs(w[w<0]) > 10**-15):
        raise ValueError('w takes negative values')
    else:
        w = np.clip(w,0,np.inf)
    WX = (np.sqrt(w)*X.T).T
    ## test
    # wX1 = np.dot(np.diag(np.sqrt(w)),X)
    hess = -1/phi_est * np.dot(WX.T,WX)
    return hess

def alpha_mu(y,mu,family):
    """
    alpha as a funciton of mu
    :return:
    """
    FLOAT_EPS = np.finfo(float).eps
    dy = y - mu
    corr = family.variance.deriv(mu) / family.variance(mu) + family.link.deriv2(mu) / family.link.deriv(mu)
    if not any(np.abs(corr[corr<0]) > 10**-15):
        corr = np.clip(corr, FLOAT_EPS, np.inf)
    alpha = 1 + dy * corr
    return alpha

def alpha_deriv(y,mu,family):
    """
    Derivative of alpha wrt mu (model specific derivatives)
    """
    FLOAT_EPS = np.finfo(float).eps
    term1 = -(family.variance.deriv(mu) / family.variance(mu) + family.link.deriv2(mu)/family.link.deriv(mu))
    dy = (y - mu)
    add1 = (family.variance.deriv2(mu)*family.variance(mu) - family.variance.deriv(mu)**2)/family.variance(mu)**2
    add2 = (family.link.deriv3(mu)*family.link.deriv(mu) - family.link.deriv2(mu)**2)/np.clip(family.link.deriv(mu)**2, FLOAT_EPS, np.inf)
    return term1 + dy *(add1 + add2)



def d3beta_unpenalized_ll(beta,y,X,family,phi_est):
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, beta))
    dalpha_dmu = alpha_deriv(y,mu,family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    # d3beta_ll = -1/phi_est * np.einsum('i,i,im,ir,il->mrl',dalpha_dmu,dmu_deta,X,X,X)
    t0 = perf_counter()
    temp = -1/phi_est*contract('i,i,im,ir,il->mrl',dalpha_dmu,dmu_deta,X,X,X)
    t1 = perf_counter()
    print('optim einsum:',t1-t0)
    t0 = perf_counter()
    try:
        d3beta_ll = -1/phi_est * fast_summations.d3beta_unpenalized_ll_summation(X,dalpha_dmu,dmu_deta)
    except:
        d3beta_ll =  -1/phi_est*np.einsum('i,i,im,ir,il->mrl', dalpha_dmu, dmu_deta, X, X, X)
    t1 = perf_counter()
    print('fastsum:', t1 - t0)
    return d3beta_ll

def ll_MLE_rho(rho,y,X,family,sm_handler,var_list,phi_est,conv_criteria='deviance',max_iter=10**3,tol=1e-10,returnMLE=False):
    mu = family.starting_mu(y)
    # mu = y + delta
    # eta = self.family.link(mu)
    converged = False
    old_conv_score = -100
    n_obs = y.shape[0]
    iteration = 0
    smooth_pen = np.exp(rho)
    sm_handler.set_smooth_penalties(smooth_pen,var_list)
    print('Start optim: criteria', conv_criteria, 'smoothing par', smooth_pen)
    # get full penalty matrix
    pen_matrix = sm_handler.get_penalty_agumented(var_list)
    # agument X
    Xagu = np.vstack((X, pen_matrix))
    yagu = np.zeros(Xagu.shape[0])
    wagu = np.ones(Xagu.shape[0])

    while not converged:
        # get parameters w and z
        FLOAT_EPS = np.finfo(float).eps
        alpha = alpha_mu(y,mu,family)
        dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
        w = alpha * dmu_deta ** 2 / family.variance(mu)
        lin_pred = family.predict(mu)
        z = lin_pred + family.link.deriv(mu) * (y - mu) / alpha
        # agument variables
        yagu[:n_obs] = z
        wagu[:n_obs] = w
        # fit WLS
        model = sm.WLS(yagu, Xagu, wagu)
        fit_OLS = model.fit()
        lin_pred = np.dot(X[:n_obs, :], fit_OLS.params)
        # get new mu
        mu = family.fitted(lin_pred)

        # check convergence
        conv_score = convergence_score(y, model, family, eta=lin_pred, criteria=conv_criteria)
        converged = abs(conv_score - old_conv_score) < tol * conv_score
        old_conv_score = conv_score
        if iteration >= max_iter:
            break
        iteration += 1
    if returnMLE:
        return fit_OLS.params
    S_all = compute_Sall(sm_handler,var_list)
    ll_beta_hat = dbeta_unpenalized_ll(fit_OLS.params, y, X, family, phi_est) + penalty_ll(rho, fit_OLS.params, S_all, phi_est)
    return ll_beta_hat

def convergence_score(y, model, family,criteria='gcv',eta=None):
    if criteria == 'gcv':
        return compute_gcv_convergence(y,model)
    if criteria == 'deviance':
        return compute_deviance(y,eta,family)

def compute_gcv_convergence(y,model):
    res = sm.OLS(model.wendog, model.wexog).fit()
    n_obs = y.shape[0]
    hat_diag = res.get_influence().hat_matrix_diag[:n_obs]
    trA = hat_diag.sum()
    # print('trA form old',trA)
    rsd = model.wendog[:n_obs] - res.fittedvalues[:n_obs]
    rss = np.sum(np.power(rsd, 2))
    sig_hat = rss / (n_obs - trA)
    gcv = sig_hat * n_obs / (n_obs - trA)
    return gcv

def compute_deviance(y,eta,family):
    mu = family.link.inverse(eta)
    return family.deviance(y, mu)

def mle_gradient_bassed_optim(rho,sm_handler, var_list,y,X,family,phi_est = 1, method='Newton-CG',num_random_init=1,beta_zero=None,tol=10**-8):
    # two possible methods:
    # method = 'Newton-CG'
    # method='L-BFGS-B'
    # should not depend on phi_est (as seen in the EM WLS solution)
    Slam = create_Slam(rho, sm_handler, var_list)
    func = lambda beta: -1*(unpenalized_ll(beta,y,X,family,phi_est,omega=1) + penalty_ll_Slam(Slam,beta,phi_est))
    grad_func = lambda beta: -1*(dbeta_unpenalized_ll(beta,y,X,family,phi_est) + dbeta_penalty_ll_Slam(Slam,beta,phi_est))
    if method == 'Newton-CG':
        # much more reliable estimate
        hess_func = lambda  beta: -1*(d2beta_penalty_ll_Slam(Slam,beta,phi_est) + d2beta_unpenalized_ll(beta,y,X,family,phi_est))
    else:
        hess_func = None
    # res = minimize(gcv_func,rho0,method='Newton-CG',jac=gcv_grad,hess=gcv_hess,tol=10**-8)
    if beta_zero is None:
        curr_min = np.inf
        for kk in range(num_random_init):
            beta0 = np.random.normal(0, 0.1, X.shape[1])
            tmp = minimize(func, beta0, method=method, jac=grad_func, hess=hess_func, tol=tol)
            if tmp.fun < curr_min:
                res = tmp
                curr_min = tmp.fun
                beta_zero = beta0.copy()
    #else:
    res = minimize(func, beta_zero, method=method, jac=grad_func, hess=hess_func, tol=tol)

    return res.x,res,beta_zero

def Vbeta_rho(rho, b_hat, y, X, family,sm_handler, var_list,phi_est,inverse=False,compute_grad = False):
    ## numerically stable method to compute Vb
    if compute_grad:
        b_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
                                                 num_random_init=10)[0]
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, b_hat))
    alpha = alpha_mu(y, mu, family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta ** 2 / family.variance(mu)
    if any(np.abs(w[w<0]) > 10**-15):
        raise ValueError('w takes negative values')
    else:
        w = np.clip(w,0,np.inf)
    WX = (np.sqrt(w) * X.T).T

    Q, R = np.linalg.qr(WX, mode='reduced')
    sm_handler.set_smooth_penalties(np.exp(rho),var_list)
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    _, s, V_T = linalg.svd(np.vstack((R, B)))

    # remove low val singolar values
    i_rem = np.where(s < 10 ** (-8) * s.max())[0]

    # remove cols
    s = np.delete(s, i_rem, 0)
    V_T = np.delete(V_T, i_rem, 0)

    # compute the diag matrix with the singolar vals
    di = np.diag_indices(s.shape[0])
    D = np.zeros((s.shape[0], s.shape[0]))  # Dinv * Dinv
    if inverse:
        D[di] = phi_est / (s) ** 2
    else:
        D[di] = (s)**2 / phi_est

    D = np.matrix(D)

    # transform everything needed in matrix
    D, V_T = matrix_transform(D, V_T)
    sum_hes_inv = -V_T.T * D * V_T

    sumfunc = lambda rho: -H_rho(rho, y, X, S_all, family, phi_est) - create_Slam(rho, sm_handler, var_list)/phi_est
    # sum_hes2 = sumfunc(rho)

    return sum_hes_inv

def dbeta_hat(rho,b_hat,S_all,sm_handler, var_list,y,X,family,phi_est=1,compute_gradient=False,method='Newton-CG'):

    if compute_gradient:
        beta = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method=method,
                                                 num_random_init=10,tol=10**-12)[0]
    else:
        beta = b_hat
    sum_hes_inv = Vbeta_rho(rho, beta, y, X, family,sm_handler, var_list,phi_est,inverse=True)

    # use broadcasting for multiply Sj * lam[j]
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    Slam_tensor = (Slam_tensor.T*np.exp(rho)).T / phi_est
    # use einsum to perform desired combinaiton
    P1 = np.einsum('kij,j->ki', Slam_tensor, beta)
    true_grad = np.einsum('li,ki->kl',sum_hes_inv,P1)
    # true_grad = np.einsum('li,kij,j->kl',sum_hes_inv,Slam_tensor,beta)

    return true_grad

def d2beta_hat(rho,b_hat,S_all,sm_handler, var_list,y,X,family,phi_est=1):
    """
    This function compute the hessian of the MLE \hat{\beta} wrt to the smoothing
    paramters (the formula in the paper and in the book is wrong! check out overleaf)
    """
    # compute gradient of H
    dH_drho = grad_H_drho(rho, b_hat, y, X, sm_handler, var_list, family,S_all, phi_est)
    # compute gradient of Slam
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    exp_rho = np.exp(rho)
    dSlam_drho = (Slam_tensor.T * exp_rho).T / phi_est

    # compute -np.linalg.inv(H + Slam)
    neg_sum_inv = np.array(
        Vbeta_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=True, compute_grad=False))

    grad_neg_sum = -(dH_drho + dSlam_drho)
    grad_beta = dbeta_hat(rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est)





    if np.isfortran(neg_sum_inv):
        neg_sum_inv = np.array(neg_sum_inv,order='C')
    if np.isfortran(grad_neg_sum):
        grad_neg_sum = np.array(grad_neg_sum,order='C')
    if np.isfortran(dSlam_drho):
        dSlam_drho = np.array(dSlam_drho,order='C')
    if np.isfortran(b_hat):
        b_hat = np.array(b_hat,order='C')
    if np.isfortran(grad_beta):
        grad_beta = np.array(grad_beta,order='C')

    try:
        add1 = fast_summations.d2beta_hat_summation_1(neg_sum_inv, grad_neg_sum, dSlam_drho, b_hat) # add1 = np.einsum('ij,hjl,lr,krp,p->hki', neg_sum_inv, grad_neg_sum, -neg_sum_inv, dSlam_drho, b_hat)
        add2 = fast_summations.d2beta_hat_summation_2(neg_sum_inv, dSlam_drho, grad_beta)
    except:
        add1 = np.einsum('ij,hjl,lr,krp,p->hki', neg_sum_inv, grad_neg_sum, -neg_sum_inv, dSlam_drho, b_hat,optimize=True)
        tmp = np.einsum('kjl,hl->hkj', dSlam_drho, grad_beta, optimize=True)
        add2 = np.einsum('hkj,ij->hki', tmp, neg_sum_inv, optimize=True)
    di1, di2 = np.diag_indices(rho.shape[0])
    hes_beta = add1 + add2
    hes_beta[di1, di2] = hes_beta[di1, di2] + np.einsum('ij,hjl,l->hi', neg_sum_inv, dSlam_drho, b_hat) # equiv np.einsum('ij,kjl,hl->hki', neg_sum_inv, dSlam_drho, grad_beta)

    return hes_beta

def w_mu(mu,y,family):
    FLOAT_EPS = np.finfo(float).eps
    alpha = alpha_mu(y, mu, family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta ** 2 / family.variance(mu)
    return w

def w_deriv(mu,y,family):
    """
    Derivatives of the wrt mu (model specific, useful to compute dH/drho
    """
    FLOAT_EPS = np.finfo(float).eps
    alpha_prime = alpha_deriv(y,mu,family)
    g_prime = family.link.deriv(mu)
    g_2prime = family.link.deriv2(mu)
    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    alpha = alpha_mu(y,mu,family)
    w_prime = (alpha_prime * g_prime * V - alpha*(2*g_2prime*V + V_prime*g_prime)) / (np.clip(g_prime**3, FLOAT_EPS, np.inf) * V**2)
    return w_prime

def w_2deriv(mu,y,family):
    """
    Derivatives of the wrt mu (model specific, useful to compute dH/drho
    """
    FLOAT_EPS = np.finfo(float).eps
    alpha = alpha_mu(y,mu,family)
    alpha_prime = alpha_deriv(y,mu,family)
    alpha_2prime = alpha_deriv2(y,mu,family)
    g_prime = np.clip(family.link.deriv(mu),FLOAT_EPS,np.inf)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)
    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)

    NUM = alpha_prime * g_prime * V - alpha*(2*g_2prime*V + V_prime*g_prime)
    NUM_prime = alpha_2prime*g_prime*V + alpha_prime*g_2prime*V + alpha_prime*g_prime*V_prime \
                - alpha_prime * (2*g_2prime*V + g_prime*V_prime)\
                - alpha *(2*g_3prime * V + 2*g_2prime*V_prime + V_2prime*g_prime + V_prime*g_2prime)


    tmp_DEN_prime = 3 * g_2prime*V + 2* g_prime * V_prime
    w_2prime = (NUM_prime*(g_prime*V) - NUM*tmp_DEN_prime)/(g_prime**4*V**3)
    return w_2prime


def dw_dbeta(beta,y,family,X):
    mu = family.link.inverse(np.dot(X, beta))
    w_prime = w_deriv(mu,y,family)
    g_prime = np.clip(family.link.deriv(mu),np.finfo(float).eps,np.inf)
    grad_w = (w_prime/g_prime)*X.T
    return grad_w

def w_rho(rho,beta,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False,method='Newton-CG'):
    if compute_grad:
        beta_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method=method,
                                         num_random_init=10,tol=10**-12)[0]
    else:
        beta_hat = beta
    mu = family.link.inverse(np.dot(X, beta_hat))
    w = w_mu(mu,y,family)
    return w

def dw_drho(rho,beta,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False,method='Newton-CG'):
    if compute_grad:
        np.random.seed(4)
        beta_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method=method,
                                         num_random_init=10,tol=10**-14)[0]
    else:
        beta_hat = beta
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, beta_hat))
    w_prime = w_deriv(mu, y, family)
    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est)
    # g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_prime = family.link.deriv(mu)
    dw = np.zeros((rho.shape[0],X.shape[0]))
    for h in range(rho.shape[0]):
        for k in range(X.shape[0]):
            dw[h,k] = np.dot(dB[h, :], X[k, :])*w_prime[k]/g_prime[k]

    # dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    # dw_dB = dw_dbeta(beta_hat, y, family, X)
    # dw_drho = np.einsum('li,rl->ri',dw_dB,dB)
    return dw

# def d2w_drho2(rho,beta,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False):
#     FLOAT_EPS = np.finfo(float).eps
#     if compute_grad:
#         beta_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
#                                          num_random_init=10)[0]
#     else:
#         beta_hat = beta
#     dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family,phi_est=phi_est)
#     d2B = d2beta_hat(rho,beta_hat,S_all,sm_handler, var_list,y,X,family,phi_est=phi_est)
#
#     ## formula sum_
#     # \sum_{l,j} X_{kj}  X_{kl}  \frac{w'' g' - w' g''}{g'''  }\frac{\partial \beta_j}{\rho_r} \frac{\partial \beta_l}{\partial \rho_h} + \sum_{l} X_{kl} \frac{w'}{g'} \frac{\partial^2 \beta}{\partial \rho_r \partial \rho_r}
#     h_prime = deriv_small_h(mu,y,family)
#     h = small_h_mu(mu,y,family)
#     g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
#     frac = h_prime/g_prime
#     XdB = np.einsum('ik,rk->ri',X,dB)
#     Xd2B = np.einsum('ik,rhk->rhi',X,d2B)
#     add1 = np.zeros(Xd2B.shape)
#     for h in range(rho.shape[0]):
#         for r in range(h,rho.shape[0]):
#             add1[h,r] = XdB[h,:]*XdB[r,:]*frac
#             add1[r, h] = add1[h,r]
#     add2 = Xd2B*h
#     return add1+add2






# def d2w_rho(beta,y,family,X):
#     mu = family.link.inverse(np.dot(X, beta))
#     g_prime = np.clip(family.link.deriv(mu),np.finfo(float).eps,np.inf)
#     h_prime = deriv_small_h(mu,y,family)
#     frac = h_prime/g_prime
#     hes_w = np.einsum('k,kr,kh->rhk',frac,X,X)
#     return hes_w

def grad_H_drho(rho,beta,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False):
    if compute_grad:
        beta_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
                                         num_random_init=10)[0]
    else:
        beta_hat = beta
    w_prime = dw_dbeta(beta_hat,y,family,X)
    dB = dbeta_hat(rho,beta_hat,S_all,sm_handler,var_list,y,X,family)

    # this sum computes \sum_l dH/dB_l * dB_l/d\rho_k
    if np.isfortran(X):
        X = np.array(X,order='C')
    if np.isfortran(dB):
        dB = np.array(dB,order='C')
    if np.isfortran(w_prime):
        w_prime = np.array(w_prime,order='C')
    try:
        grad_H = fast_summations.grad_H_summation(X, dB, w_prime)/phi_est # equivalent np.einsum('ki,lk,kj,rl->rij', X, w_prime, X, dB)/(phi_est)
    except:
        grad_H = np.einsum('ki,lk,kj,rl->rij', X, w_prime, X, dB)/phi_est
    return grad_H


def deriv_compute(rho,y,X,sm_handler,var_list,family,S_all,phi_est,test=True,omega=1,fix_beta=False):
    
    # if all(rho==deriv_compute.last_rho):
    #     # print('fetch cached value')
    #     return deriv_compute.last_res
    
    print('rho',rho)
    
    FLOAT_EPS = np.finfo(float).eps
    if fix_beta == False:
        beta_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
                                         num_random_init=10)[0]
    else:
        beta_hat = fix_beta
    
    mu = family.link.inverse(np.dot(X, beta_hat))
    
    
    ll_penalty = penalty_ll(rho,beta_hat,sm_handler, var_list,phi_est)
    ll_unpen = unpenalized_ll(beta_hat,y,X,family,phi_est,omega=omega)

    # create Slam and transform it to compute determinant
    Slam_trans, S_transf = transform_Slam(S_all, rho)

    # every time rho is changed, S_transf is recomputed, so no need for compute_grad to be true
    log_det_Slam = -0.5*logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all)/phi_est
    
    
    # compute the derivatives for the grad_REML
    alpha_prime = alpha_deriv(y,mu,family)
    g_prime = family.link.deriv(mu)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)
    g_prime_inv = np.array(1/g_prime,order='C')
    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)
    alpha = alpha_mu(y,mu,family)
    alpha_prime = alpha_deriv(y,mu,family)
    alpha_2prime = alpha_deriv2(y,mu,family)
    w_prime = (alpha_prime * g_prime * V - alpha*(2*g_2prime*V + V_prime*g_prime)) / (np.clip(g_prime**3, FLOAT_EPS, np.inf) * V**2)
   
    term1 = (alpha_2prime * g_prime * V - alpha_prime * g_2prime * V - 2 * alpha * g_3prime * V
             -3 * alpha * g_2prime * V_prime - alpha * V_2prime * g_prime) * g_prime * V
    term2 = -(alpha_prime * g_prime * V - alpha*(2*g_2prime*V + V_prime * g_prime))*(4*g_2prime*V + 2*g_prime*V_prime)
    
    # used in hessian computation
    small_h_prime = (term1 + term2)/(g_prime**5 * V**3)
    small_h = w_prime / g_prime
    
    dw_dB = ((w_prime/g_prime)*X.T).T
    
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta ** 2 / family.variance(mu)
    if any(np.abs(w[w<0]) > 10**-15):
        raise ValueError('w takes negative values')
    else:
        w = np.clip(w,0,np.inf)
    WX = (np.sqrt(w) * X.T).T

    Q, R = np.linalg.qr(WX, mode='reduced')
    sm_handler.set_smooth_penalties(np.exp(rho),var_list)
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    _, s, V_T = linalg.svd(np.vstack((R, B)))
    
    
    log_det_sum = -np.sum(np.log(s)) + X.shape[1] * np.log(phi_est)
    
    # remove low val singolar values
    i_rem = np.where(s < 10 ** (-8) * s.max())[0]

    # remove cols
    s = np.delete(s, i_rem, 0)
    V_T = np.delete(V_T, i_rem, 0)

    # compute the diag matrix with the singolar vals
    di = np.diag_indices(s.shape[0])
    Dinv = np.zeros((s.shape[0], s.shape[0]))  # Dinv * Dinv
    
    Dinv[di] = phi_est / (s) ** 2
    
    D = np.zeros((s.shape[0], s.shape[0])) 
    D[di] = (s)**2 / phi_est
    
    Dinv, V_T, D = matrix_transform(Dinv, V_T, D)
    
    # sum_H_Slam = V_T.T * D * V_T
    # log_det_sum = -0.5*np.log(np.linalg.det(sum_H_Slam))
    # eig = np.linalg.eigh(V_T.T * D * V_T)[0]
    

    # null space of Slam
    M = beta_hat.shape[0] - np.linalg.matrix_rank(Slam_trans)
    
    REML = ll_unpen + ll_penalty + log_det_Slam + log_det_sum + M*np.log(np.pi*2)
    if test:
        REML1 =  laplace_appr_REML(rho,beta_hat,S_all,y,X,family,phi_est,sm_handler,var_list,omega=omega,compute_grad=False,
                      fixRand=False,method='Newton-CG',tol=10**-12,num_random_init=1)
        print('REML',np.max(np.abs(REML1-REML)))
    # transform everything needed in matrix
    
    sum_hes_inv = -V_T.T * Dinv * V_T
    
    if test:
        sum_hes_inv1 = Vbeta_rho(rho, beta_hat, y, X, family,sm_handler, var_list,phi_est,inverse=True)
        print('Vb_inv check',np.max(np.abs(sum_hes_inv-sum_hes_inv1)))
    
    # use broadcasting for multiply Sj * lam[j]
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    Slam_tensor = (Slam_tensor.T*np.exp(rho)).T / phi_est
    # use einsum to perform desired combinaiton
    P1 = np.einsum('kij,j->ki', Slam_tensor, beta_hat)
    dB_drho = np.einsum('li,ki->kl',sum_hes_inv,P1)
    
    
    dw_drho_array = np.zeros((X.shape[0],rho.shape[0]))
    dH_drho = np.zeros((rho.shape[0],beta_hat.shape[0],beta_hat.shape[0]))
    t0 = perf_counter()
    for k in range(rho.shape[0]):
        dw_drho_array[:,k] = np.dot(dw_dB,dB_drho[k,:]) #nxk
        dH_drho[k,:,:] = np.dot(X.T*dw_drho_array[:,k], X)/phi_est
    # print('done: %s'%(perf_counter()-t0))
    
    if test:
        dw_drho1 = dw_drho(rho,beta_hat,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False,method='Newton-CG')
        
        t0 = perf_counter()
        grH = grad_H_drho(rho,beta_hat,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False)
        t1 = perf_counter()
        print('old method',t1-t0,np.max(np.abs(grH - dH_drho)))
    
    dVb_drho_arr = Slam_tensor + dH_drho
     
    if test:
        dVb_drho_arr1 = dVb_drho(rho, beta_hat, S_all, y, X, family, sm_handler, var_list, phi_est,compute_grad=False)
        print('dVb diff:',np.max(np.abs(dVb_drho_arr1 - dVb_drho_arr)))
    
    
    add1 = -0.5 * np.einsum('i,rij,j->r', beta_hat, Slam_tensor, beta_hat)


    # every time rho is changed, S_transf is recomputed, so no need for compute_grad to be true
    add2 = -0.5 * grad_logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all) / phi_est

    add3 = np.zeros(add1.shape)
    for j in range(rho.shape[0]):
        # compute the trace of a product with inner1d
        add3[j] = -0.5*np.sum(inner1d(-sum_hes_inv, dVb_drho_arr[j].T))
        
    grad_REML = add1 + add2 + add3
    if test:
        grad_REML1 = grad_laplace_appr_REML(rho,beta_hat,S_all,y,X,family,phi_est,sm_handler,var_list,omega=1,compute_grad=False,
                            fixRand=False,method='Newton-CG',num_random_init=1,tol=10**-12)
        print('grad_reml', np.max(np.abs(grad_REML - grad_REML1)))
    

    # t0 = perf_counter()
    grad_neg_sum = -dVb_drho_arr
    if np.isfortran(sum_hes_inv):
        sum_hes_inv = np.array(sum_hes_inv,order='C')
    if np.isfortran(grad_neg_sum):
        grad_neg_sum = np.array(grad_neg_sum,order='C')
    if np.isfortran(Slam_tensor):
        Slam_tensor = np.array(Slam_tensor,order='C')
    if np.isfortran(beta_hat):
        beta_hat = np.array(beta_hat,order='C')
    if np.isfortran(dB_drho):
        dB_drho = np.array(dB_drho,order='C')

    try:
        add1 = fast_summations.d2beta_hat_summation_1(sum_hes_inv, grad_neg_sum, Slam_tensor, beta_hat) # add1 = np.einsum('ij,hjl,lr,krp,p->hki', neg_sum_inv, grad_neg_sum, -neg_sum_inv, dSlam_drho, b_hat)
        add2 = fast_summations.d2beta_hat_summation_2(sum_hes_inv, Slam_tensor, dB_drho)
    except:
        add1 = np.einsum('ij,hjl,lr,krp,p->hki', sum_hes_inv, grad_neg_sum, -sum_hes_inv, Slam_tensor, b_hat,optimize=True)
        tmp = np.einsum('kjl,hl->hkj', Slam_tensor, dB_drho, optimize=True)
        add2 = np.einsum('hkj,ij->hki', tmp, sum_hes_inv, optimize=True)
    di1, di2 = np.diag_indices(rho.shape[0])
    hes_beta = add1 + add2
    hes_beta[di1, di2] = hes_beta[di1, di2] + np.einsum('ij,hjl,l->hi', sum_hes_inv, Slam_tensor, beta_hat) # equiv np.einsum('ij,kjl,hl->hki', neg_sum_inv, dSlam_drho, grad_beta)
    # t1 = perf_counter()
    # print('done db2',t1-t0)
    if test:
        d2b = d2beta_hat(rho,beta_hat,S_all,sm_handler, var_list,y,X,family,phi_est=1)
        print('d2beta', np.max(np.abs(d2b - hes_beta)))
        
    
    
    NUM = alpha_prime * g_prime * V - alpha*(2*g_2prime*V + V_prime*g_prime)
    NUM_prime = alpha_2prime*g_prime*V + alpha_prime*g_2prime*V + alpha_prime*g_prime*V_prime \
                - alpha_prime * (2*g_2prime*V + g_prime*V_prime)\
                - alpha *(2*g_3prime * V + 2*g_2prime*V_prime + V_2prime*g_prime + V_prime*g_2prime)


    tmp_DEN_prime = 3 * g_2prime*V + 2* g_prime * V_prime
    w_2prime = (NUM_prime*(g_prime*V) - NUM*tmp_DEN_prime)/(g_prime**4*V**3)
    tt0 = perf_counter()
    t0 = perf_counter()
    hess_H = np.zeros((rho.shape[0],rho.shape[0],beta_hat.shape[0], beta_hat.shape[0]))
   
    for i in range(rho.shape[0]):
        tmp_i = np.dot((X.T * small_h_prime * g_prime_inv).T, dB_drho[i,:]) # N x 1
        for j in range(i,rho.shape[0]):
            tmp_j = np.dot(X,dB_drho[j,:]) # N*1
            hess_H_term1 = np.dot(X.T * (tmp_i*tmp_j),X)
            hess_H_term2 = np.dot(X.T * np.dot((X.T*small_h).T, hes_beta[i,j,:]),X)
            hess_H[i, j] = (hess_H_term1 + hess_H_term2)/phi_est
            hess_H[j, i] = hess_H[i, j]
            
    t1 = perf_counter()
    # print('done hess time',t1-t0)
    if test:
        t0 = perf_counter()
        hes_H = hes_H_drho(rho,beta_hat,y,X,S_all,sm_handler,var_list,family,phi_est,return_all=False)
        t1 = perf_counter()
        print('old hess time',t1-t0)
        print(np.max(np.abs(hess_H-hes_H)))
    
    
    d2Vb = hess_H.copy()
    di1,di2 = np.diag_indices(rho.shape[0])
    d2Vb[di1,di2] = hess_H[di1,di2] + Slam_tensor
    
    if test:
        d2Vb2 = d2Vb_drho(rho, beta_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
        print(np.max(np.abs(d2Vb2-d2Vb)))
    
    # Vb1 = -Vbeta_rho(rho, beta_hat, y, X, family, sm_handler, var_list, phi_est, inverse=False, compute_grad=False)
    Vb = V_T.T * D * V_T
    add1 = np.einsum('hj,ji,ki->hk', dB_drho, Vb, dB_drho, optimize='optimal')
    di1, di2 = np.diag_indices(rho.shape[0])
    add1[di1, di2] = add1[di1, di2] - 0.5 * np.einsum('i,hij,j->h', beta_hat, Slam_tensor, beta_hat,optimize='optimal') 
    
    add2 = -0.5 * hes_logDet_Slam(rho, S_transf) / phi_est
    
    Vb_inv = -sum_hes_inv
    # dVb = dVb_drho(rho, beta_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
    # d2Vb1 = d2Vb_drho(rho,beta_hat,S_all,y,X,family,sm_handler,var_list,phi_est)
    try:
        add3 = 0.5 * fast_summations.trace_log_det_H_summation_1(Vb_inv,dVb_drho_arr,d2Vb)
    except:
        tmp = np.einsum('ij,hjk->hik', Vb_inv, dVb_drho_arr, optimize=True)
        add3 = 0.5*(np.einsum('hij,rji->hr', tmp, tmp) - np.einsum('ij,hrji->hr', Vb_inv, d2Vb))
        
    hess_REML = add1 + add2 + add3
    tt1 = perf_counter()
    # print('time hess new:',tt1-tt0)
    if test:
        t0 = perf_counter()
        hess_REML1 = hess_laplace_appr_REML(rho,beta_hat,S_all,y,X,family,phi_est,sm_handler,var_list,compute_grad=False,
                           fixRand=False,method='Newton-CG',num_random_init=1,tol=10**-12)
        t1 = perf_counter()
        print(np.abs(np.max(hess_REML-hess_REML1)),'time hess old:',t1-t0)

    
    return REML, grad_REML, hess_REML

def hes_H_drho(rho,beta_hat,y,X,S_all,sm_handler,var_list,family,phi_est,return_all=False):

    mu = family.link.inverse(np.dot(X, beta_hat))
    h_prime = deriv_small_h(mu,y,family)
    h = small_h_mu(mu,y,family)
    g_prime = family.link.deriv(mu)
    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    d2B = d2beta_hat(rho,beta_hat,S_all,sm_handler,var_list,y,X,family)
    if np.isfortran(X):
        X = np.array(X,order='C')
    if np.isfortran(dB):
        dB = np.array(dB,order='C')
    if np.isfortran(h_prime):
        h_prime = np.array(h_prime,order='C')
    if np.isfortran(d2B):
        d2B = np.array(d2B,order='C')
    if np.isfortran(h):
        h = np.array(h,order='C')

    g_prime_inv = np.array(1/g_prime,order='C')
    #np.einsum_path('ki,kj,kl,ky,k,k,hy,rl->hrij', X, X, X, X, h_prime, 1 / g_prime, dB, dB)
    try:
        part1 = fast_summations.hessian_H_summation_1( X,  dB,  h_prime, g_prime_inv) # equivalent to np.einsum('ki,kj,kl,ky,k,k,hy,rl->hrij',X,X,X,X,h_prime,1/g_prime,dB,dB)
    except:
        part1 = np.einsum('ki,kj,kl,ky,k,k,hy,rl->hrij', X, X, X, X, h_prime, 1 / g_prime, dB, dB)

    try:
        part2 = fast_summations.hessian_H_summation_2(X,d2B,h) # equivalent to np.einsum('ki,kj,kl,k,hrl->hrij',X,X,X,h,d2B)
    except:
        part2 = np.einsum('ki,kj,kl,k,hrl->hrij', X, X, X, h, d2B)
    hes_H = (part1 + part2) / phi_est
    if return_all:
        return hes_H,part1,part2,g_prime_inv,dB,d2B,h,h_prime

    return hes_H

def compute_T_matrices(rho,X,y,family,beta_hat,var_list,phi_est):
    FLOAT_EPS = np.finfo(float).eps
    dw_dB = dw_dbeta(beta_hat, y, family, X)
    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family,phi_est=phi_est)
    dw_drho = np.einsum('li,rl->ri',dw_dB,dB,optimize='optimal')

    mu = family.link.inverse(np.dot(X, beta_hat))
    w = w_mu(mu, y, family)

    Tj = dw_drho / np.abs(w)
    d2B = d2beta_hat(rho,beta_hat,S_all,sm_handler,var_list,y,X,family,phi_est=phi_est)
    w_2prime = w_2deriv(mu,y,family)
    w_prime = w_deriv(mu,y,family)

    g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_2prime = family.link.deriv2(mu)

    den = np.clip(g_prime,FLOAT_EPS,np.inf)
    frac1 = (w_2prime*g_prime - w_prime * g_2prime)/den**3
    frac2 = w_prime/den

    d2w_drho = np.zeros((rho.shape[0],rho.shape[0],X.shape[0]))
    for h in range(rho.shape[0]):
        XdBh = X * dB[h,:]
        for r in range(rho.shape[0]):
            XdBr = X * dB[r, :]
            for k in range(X.shape[0]):

                d2w_drho[h,r,k] = np.dot(XdBh[k,:],XdBr[k,:])*frac1[k]\
                                  + np.dot(X[k,:],d2B[h,r,:])*frac2[k]
    return d2w_drho

def hes_w_wrt_rho(rho,beta,y,X,sm_handler,var_list,family,S_all,phi_est,compute_grad=False):
    FLOAT_EPS = np.finfo(float).eps
    if compute_grad:
        beta_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
                                         num_random_init=10)[0]
    else:
        beta_hat = beta

    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family,phi_est=phi_est)
    d2B = d2beta_hat(rho,beta_hat,S_all,sm_handler,var_list,y,X,family,phi_est=phi_est)

    mu = family.link.inverse(np.dot(X, beta_hat))
    w_prime = w_deriv(mu,y,family)
    w_2prime = w_2deriv(mu, y, family)
    g_prime=family.link.deriv(mu)
    # g_prime = np.clip(family.link.deriv(mu),FLOAT_EPS,np.inf)
    g_2prime = np.clip(family.link.deriv2(mu), FLOAT_EPS, np.inf)
    add2 = np.zeros((rho.shape[0],rho.shape[0],X.shape[0]))
    add1 = np.zeros((rho.shape[0],rho.shape[0],X.shape[0]))
    for r in range(rho.shape[0]):
        for h in range(rho.shape[0]):
            for k in range(X.shape[0]):
                dBrX = np.dot(dB[r,:],X[k,:])
                dBhX = np.dot(dB[h, :], X[k, :])
                add2[r,h,k] = np.dot(X[k,:],d2B[r,h,:])*(w_prime[k]/g_prime[k])
                add1[r,h,k] = dBrX*dBhX * (w_2prime[k]*g_prime[k] - w_prime[k]*g_2prime[k])/(g_prime[k]**3)

    d2w = add1+add2
    # dw = np.zeros((rho.shape[0],X.shape[0]))
    # for h in range(rho.shape[0]):
    #     for k in range(X.shape[0]):
    #         dw[h,k] = np.dot(dB[h, :], X[k, :])*w_prime[k]/g_prime[k]

    return d2w

def det_H_rho(X,y,family,w,beta_hat):
    w_prime = dw_dbeta(beta_hat, y, family, X)
    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    pass

def H_rho(rho,beta,y,X,family,phi_est,comp_gradient=True):
    if comp_gradient:
    # only for test purposes
        beta = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
                                                 num_random_init=10)[0]
    mu = family.link.inverse(np.dot(X, beta))
    w = w_mu(mu, y, family)
    H = np.dot(X.T * w,X) / (phi_est)
    return H

def R_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est):
    Vb = -Vbeta_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=False)
    R = np.linalg.cholesky(Vb)
    return R

def dVb_drho(rho, beta, S_all, y, X, family, sm_handler, var_list, phi_est,compute_grad=False):
    if compute_grad:
        b_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method='Newton-CG',
                                         num_random_init=10)[0]
    else:
        b_hat = beta
    # R = R_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est)
    dH = grad_H_drho(rho,b_hat,y,X,sm_handler,var_list,family,S_all,phi_est)
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    dS = (np.exp(rho)*Slam_tensor.T).T/phi_est
    D = dH + dS
    return D

def d2Vb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est):
    # R = R_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est)
    d2Vb = hes_H_drho(rho,b_hat,y,X,S_all,sm_handler,var_list,family,phi_est)
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    dS = (np.exp(rho)*Slam_tensor.T).T/phi_est
    di1,di2 = np.diag_indices(rho.shape[0])
    d2Vb[di1,di2] = d2Vb[di1,di2] + dS
    return d2Vb

def test_dVb_drho(rho, S_all, y, X, family, sm_handler, var_list, phi_est,inverse=False):
    beta = mle_gradient_bassed_optim(rho, sm_handler,var_list, y, X, family, phi_est=phi_est, method='Newton-CG',

                                     num_random_init=10)[0]
    # print('estim in test',beta[:4])
    # R = R_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est)
    Vb = -Vbeta_rho(rho,beta,y,X,family,sm_handler,var_list,phi_est,inverse=inverse)
    return Vb


def grad_cholesky(grad_D_rho, R):
    """

    :param grad_D_rho: gradient of the matrix before the cholesky decomposition
    :param R: cholesky decomposition result : D = R.T*R
    :return:
    """
    # if np.sum(np.triu(R,1)) == 0:
    #     R = R.T
    grad_chol = np.zeros(grad_D_rho.shape)
    for i in range(R.shape[0]):
        Bii = grad_D_rho[i, i] - np.dot(grad_chol[:i, i].flatten(), R[:i, i].flatten()) - np.dot(R[:i, i].flatten(), grad_chol[:i, i].flatten())
        grad_chol[i, i] = 0.5 * Bii / R[i, i]
        for j in range(i+1,R.shape[0]):
            Bij = grad_D_rho[i, j] - np.dot(grad_chol[:i, i].flatten(), R[:i, j].flatten()) - np.dot(R[:i, i].flatten(), grad_chol[:i, j].flatten())
            grad_chol[i, j] = (Bij - R[i, j] * grad_chol[i, i]) / R[i, i]
    return grad_chol

def grad_chol_Vb_rho(rho,b_hat, S_all, y, X, family, sm_handler, var_list, phi_est):
    # Vb = test_dVb_drho(rho, S_all, y, X, family, sm_handler, var_list, phi_est)
    Vb = -Vbeta_rho(rho,b_hat,y,X,family,sm_handler,var_list,phi_est,inverse=True)
    R = np.array(np.linalg.cholesky(Vb))
    R = R.T
    dVb = dVb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
    dVb = -np.einsum('ij,hjk,kl->hil',Vb,dVb,Vb,optimize='optimal')
    grad_chol_Vb = np.zeros(dVb.shape)
    for j in range(rho.shape[0]):
        grad_chol_Vb[j] = grad_cholesky(dVb[j], R)

    return grad_chol_Vb

def cholesky_Vb_rho(rho, S_all, y, X, family, sm_handler, var_list, phi_est):
    Vb = test_dVb_drho(rho, S_all, y, X, family, sm_handler, var_list, phi_est,inverse=True)

    R = np.array(np.linalg.cholesky(Vb))
    R = R.T
    return R

## CONTROLLED EXAMPLE TO GET THE CORRECT CHOLSESKY DECOMPOSITION DERIVATIVE

def ftest_for_chol_Ax(x):
    A = np.zeros((3, 3))
    A[0, 0] = 1 + x ** 2 + x ** 4
    A[0, 1] = 2 * x + x ** 3
    A[0, 2] = 3 * x ** 2
    A[1, 1] = 1 + 2 * x ** 2
    A[1, 2] = x ** 3 + 2 * x
    A[2, 2] = 1 + x ** 2 + x ** 4
    A = A + np.tril(A.T,1)
    return A

def ftest_for_chol_dA_dx(x):
    A = np.zeros((3,3))
    A[0,0] = 2*x + 4*x**3
    A[0,1] = 2 + 3*x**2
    A[0,2] = 6*x
    A[1,1] = 4*x
    A[1,2] = 3*x**2 + 2
    A[2,2] = 2*x + 4*x**3
    A = A + np.tril(A.T,1)
    return A

def cholA(x):
    A = ftest_for_chol_Ax(x)
    A = A.reshape(3,3)
    B = np.linalg.cholesky(A).T
    return B.reshape((1,)+B.shape)

def gradcholA(x,B):
    dA = ftest_for_chol_dA_dx(x)
    # dA = dA.reshape(3,3)
    dB = grad_cholesky(dA,B[0])
    return dB.reshape((1,)+dB.shape)

### END OF THE EXEMPLE

####### COMPUTATION OF HESSIAN H

def alpha_deriv2(y,mu,family):
    FLOAT_EPS = np.finfo(float).eps
    dy = (y - mu)
    add1 = (family.variance.deriv2(mu) * family.variance(mu) - family.variance.deriv(mu) ** 2) / family.variance(
        mu) ** 2
    add2 = (family.link.deriv3(mu) * family.link.deriv(mu) - family.link.deriv2(mu) ** 2) / np.clip(
        family.link.deriv(mu), FLOAT_EPS, np.inf) ** 2
    term1 = -2*(add1+add2)


    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)
    V_3prime = family.variance.deriv3(mu)

    g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)
    g_4prime = family.link.deriv4(mu)

    add3 = (V_3prime*V**2 -3*V_2prime*V_prime*V + 2*V_prime**3)/(V**3)
    add4 = (g_4prime * g_prime ** 2 - 3 * g_3prime * g_2prime * g_prime + 2 * g_2prime ** 3) / (g_prime ** 3)
    alpha_2prime = term1 + dy * (add3 + add4)
    return alpha_2prime

def small_h_mu(mu,y,family):
    ## see overleaf gradient and hessian of H
    FLOAT_EPS = np.finfo(float).eps
    g_prime = family.link.deriv(mu) #np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    small_h = w_deriv(mu,y,family)/g_prime
    return small_h

def deriv_small_h(mu,y,family):
    ## needs the deriv3 (use the redefined family)
    FLOAT_EPS = np.finfo(float).eps

    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)

    g_prime = family.link.deriv(mu)#np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)

    alpha = alpha_mu(y,mu,family)
    alpha_prime = alpha_deriv(y,mu,family)
    alpha_2prime = alpha_deriv2(y,mu,family)

    term1 = (alpha_2prime * g_prime * V - alpha_prime * g_2prime * V - 2 * alpha * g_3prime * V
             -3 * alpha * g_2prime * V_prime - alpha * V_2prime * g_prime) * g_prime * V
    term2 = -(alpha_prime * g_prime * V - alpha*(2*g_2prime*V + V_prime * g_prime))*(4*g_2prime*V + 2*g_prime*V_prime)
    small_h_prime = (term1 + term2)/(g_prime**5 * V**3)
    return small_h_prime

def laplace_appr_REML(rho,beta,S_all,y,X,family,phi_est,sm_handler,var_list,omega=1,compute_grad=False,
                      fixRand=False,method='Newton-CG',tol=10**-12,num_random_init=1):
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        t0 = perf_counter()
        b_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method=method,
                                     num_random_init=num_random_init,tol=tol)[0]
        t1 = perf_counter()
        print('mle fit: %.3f sec'%(t1-t0))
    else:
        b_hat = beta
    # return b_hat

    ll_penalty = penalty_ll(rho,b_hat,sm_handler, var_list,phi_est)
    ll_unpen = unpenalized_ll(b_hat,y,X,family,phi_est,omega=omega)

    # create Slam and transform it to compute determinant
    Slam_trans, S_transf = transform_Slam(S_all, rho)

    # every time rho is changed, S_transf is recomputed, so no need for compute_grad to be true
    log_det_Slam = -0.5*logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all)/phi_est

    # set compute grad to false because b_hat have already been recomputed
    sum_H_Slam = -Vbeta_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=False, compute_grad=False)
    log_det_sum = -0.5*np.log(np.linalg.det(sum_H_Slam))

    # null space of Slam
    M = b_hat.shape[0] - np.linalg.matrix_rank(Slam_trans)
    reml_approx = ll_unpen + ll_penalty + log_det_Slam + log_det_sum + M*np.log(np.pi*2)
    return reml_approx#,b_hat

def grad_laplace_appr_REML(rho,beta,S_all,y,X,family,phi_est,sm_handler,var_list,omega=1,compute_grad=False,
                           fixRand=False,method='Newton-CG',num_random_init=1,tol=10**-12):
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        b_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method=method,
                                     num_random_init=num_random_init,tol=tol)[0]
    else:
        b_hat = beta

    # create the -0.5 \lambda_j * \beta^T S_j \beta
    lams = np.exp(rho)
    S_tensor = np.zeros((len(S_all),)+S_all[0].shape)
    S_tensor[:,:,:] = S_all
    S_tensor = (S_tensor.T * lams).T

    add1 = -0.5 * np.einsum('i,rij,j->r', b_hat, S_tensor, b_hat) / phi_est

    # create Slam and transform it to compute determinant
    Slam_trans, S_transf = transform_Slam(S_all, rho)

    # every time rho is changed, S_transf is recomputed, so no need for compute_grad to be true
    add2 = -0.5 * grad_logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all) / phi_est

    Vb_inv = -Vbeta_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=True, compute_grad=False)
    dVb = dVb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)

    add3 = np.zeros(add1.shape)
    for j in range(rho.shape[0]):
        # compute the trace of a product with inner1d
        add3[j] = -0.5*np.sum(inner1d(Vb_inv, dVb[j].T))


    return add1 + add2 + add3#, b_hat

def hess_laplace_appr_REML(rho,beta,S_all,y,X,family,phi_est,sm_handler,var_list,compute_grad=False,
                           fixRand=False,method='Newton-CG',num_random_init=1,tol=10**-12):
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        b_hat = mle_gradient_bassed_optim(rho, sm_handler, var_list, y, X, family, phi_est=phi_est, method=method,
                                     num_random_init=num_random_init,tol=tol)[0]
    else:
        b_hat = beta

    Vb = -Vbeta_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=False, compute_grad=False)
    dB = dbeta_hat(rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est)

    # create the -0.5 \lambda_j * \beta^T S_j \beta
    lams = np.exp(rho)
    S_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    S_tensor[:, :, :] = S_all
    S_tensor = (S_tensor.T * lams).T

    add1 = np.einsum('hj,ji,ki->hk', dB, Vb, dB,optimize='optimal')
    di1, di2 = np.diag_indices(rho.shape[0])
    add1[di1, di2] = add1[di1, di2] - 0.5 * np.einsum('i,hij,j->h', b_hat, S_tensor, b_hat,optimize='optimal') / phi_est

    # create Slam and transform it to compute determinant
    Slam_trans, S_transf = transform_Slam(S_all, rho)

    # every time rho is changed, S_transf is recomputed, so no need for compute_grad to be true
    add2 = -0.5 * hes_logDet_Slam(rho, S_transf) / phi_est

    # H + Slam, grad and hessian computation
    Vb_inv = -Vbeta_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=True, compute_grad=False)
    dVb = dVb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
    d2Vb = d2Vb_drho(rho,b_hat,S_all,y,X,family,sm_handler,var_list,phi_est)
    try:
        add3 = 0.5 * fast_summations.trace_log_det_H_summation_1(Vb_inv,dVb,d2Vb)
    except:
        tmp = np.einsum('ij,hjk->hik', Vb_inv, dVb, optimize=True)
        add3 = 0.5*(np.einsum('hij,rji->hr', tmp, tmp) - np.einsum('ij,hrji->hr', Vb_inv, d2Vb))
    return add1 + add2 + add3#, b_hat


def balance_diag_func(rho,s,d):
    lam = np.exp(rho)
    lam_s = (lam * s.T).T
    vec = np.zeros(s.shape[0])
    idx_mat = s > np.finfo(float).eps
    for j in range(s.shape[0]):
        idx = idx_mat[j,:]
        vec[j] = np.mean(d[idx]/(d[idx]+lam_s[j,idx]))-0.4
    return np.sum(vec**2)

def grad_balance_diag_func(rho,s,d):
    lam = np.exp(rho)
    lam_s = (lam * s.T).T
    grad = np.zeros(lam.shape[0])
    idx_mat = s > np.finfo(float).eps
    for j in range(lam.shape[0]):
        idx = idx_mat[j,:]
        n = np.sum(idx)
        grad[j] = 2/n**2 * np.sum(d[idx]/(d[idx]+lam_s[j,idx])-0.4)*(np.sum(-d[idx]*s[j,idx]*lam[j]/(d[idx]+lam_s[j,idx])**2))
    return grad
