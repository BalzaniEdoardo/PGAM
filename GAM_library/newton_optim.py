
import numpy as np
import scipy.linalg as linalg
import statsmodels.api as sm
from numpy.core.umath_tests import inner1d
from gam_data_handlers import *
from time import perf_counter
useCuda = False
try:
    if not useCuda:
        raise ModuleNotFoundError
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as cuda_linalg
    flagUseCuda = True
except ModuleNotFoundError as e:
    print(e)
    flagUseCuda = False

class weights_and_data(object):
    FLOAT_EPS = np.finfo(float).eps
    def __init__(self,y,family,fisher_scoring=True):
        """

        :param y:
        :param family:  statsmodels.genmod.families.family family class
        :param fisher_scoring: if False mehtod is unstable
        """
        self.y = y.copy()
        self.family = family
        self.fisher_scoring = fisher_scoring

    def weight_compute(self,mu,alpha):
        FLOAT_EPS = np.finfo(float).eps
        dmu_deta = np.clip(1 / self.family.link.deriv(mu), FLOAT_EPS, np.inf)
        w = alpha * dmu_deta ** 2 / self.family.variance(mu)

        return w

    def alpha(self,mu):
        if not self.fisher_scoring:
            FLOAT_EPS = np.finfo(float).eps
            dy = self.y - mu
            corr = self.family.variance.deriv(mu) / self.family.variance(mu) + self.family.link.deriv2(mu) / self.family.link.deriv(mu)
            alpha = 1. + dy * np.clip(corr, FLOAT_EPS, np.inf)
        else:
            alpha = 1.
        return alpha

    def z_compute(self,mu,alpha):

        if np.isscalar(mu):
            mu = np.array([mu])
        lin_pred = self.family.predict(mu)
        wlsendog = lin_pred + self.family.link.deriv(mu) * (self.y - mu) / alpha
        # z = self.family.link.deriv(mu) * (self.y-mu)/ self.alpha(mu)+self.family.link(mu)
        return wlsendog

    def get_params(self,mu):
        alpha = self.alpha(mu)
        z = self.z_compute(mu,alpha)
        w = self.weight_compute(mu,alpha)
        return z,w





def gcv_comp(rho, X, Q, R, endog,sm_handler,var_list,return_par='gcv',gamma=1.):
    sm_handler.set_smooth_penalties(np.exp(rho),var_list)
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    n_obs = X.shape[0]
    if np.sum(np.isnan(R)) or np.sum(np.isnan(B)):
        print('NAN')
    U, s, V_T = linalg.svd(np.vstack((R, B[:, :])))

    # remove low val singolar values
    i_rem = np.where(s < 10 ** (-8) * s.max())[0]

    # remove cols
    s = np.delete(s, i_rem, 0)
    U = np.delete(U, i_rem, 1)
    V_T = np.delete(V_T, i_rem, 0)

    # compute the diag matrix with the singular vals
    D = np.zeros((s.shape[0], s.shape[0]))
    di = np.diag_indices(s.shape[0])
    D[di] = s

    ## check several steps
    U1 = U[:R.shape[0], :s.shape[0]]
    trA = np.sum(inner1d(np.array(U1),np.array(U1)))

    delta = n_obs - gamma*trA
    y = endog[:n_obs]
    Dinv = np.zeros(D.shape)
    Dinv[di] = 1 / s


    # transform everything needed in matrix
    y, Q, R, U1, Dinv, V_T = matrix_transform(y, Q, R, U1, Dinv, V_T)
    D2inv = np.zeros(D.shape)#Dinv * Dinv
    D2inv[di] = 1/s**2
    D2inv = np.matrix(D2inv)

    # X = Q * R


    y = np.matrix(endog[:n_obs].reshape(n_obs,1))

    Ay = X * (V_T.T * (D2inv * (V_T * (X.T * y))))

    alpha = np.sum(np.power(Ay - y, 2))

    gcv = n_obs*alpha/(delta)**2




    if return_par == 'gcv':
        return gcv
    elif return_par == 'alpha':
        return alpha
    elif return_par == 'delta':
        return delta
    elif return_par=='A':
        A = X * V_T.T * D2inv * V_T * X.T
        return A
    elif return_par == 'all':

        return gcv,alpha,delta, V_T.T * D2inv * V_T
    else:
        raise ValueError('unknow output specification')


def matrix_transform(*M):
    mat_list = []
    for R in M:
        mat_list +=[np.matrix(R)]
    return mat_list

def get_var_beta_inv(rho, R,sm_handler,var_list,remove_zeros=True):
    sm_handler.set_smooth_penalties(np.exp(rho),var_list)
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    U, s, V_T = linalg.svd(np.vstack((R, B[:, :])))

    # remove low val singolar values
    i_rem = np.where(s < 10 ** (-8) * s.max())[0]

    if remove_zeros:
        # remove cols
        s = np.delete(s, i_rem, 0)
        V_T = np.delete(V_T, i_rem, 0)
        D = np.zeros((s.shape[0], s.shape[0]))
        di1, di2 = np.diag_indices(s.shape[0])
        D[di1, di2] = s
        # compute alpha and delta
        D2inv = np.zeros(D.shape)
        D2inv[di1, di2] = 1 / s ** 2
        D2inv = np.matrix(D2inv)
    else:
        i_keep = np.where(s >= 10 ** (-8) * s.max())[0]
        # compute the diag matrix with the singolar vals
        D = np.zeros((s.shape[0], s.shape[0]))
        di1,di2 = np.diag_indices(s.shape[0])
        D[di1[i_keep],di2[i_keep]] = s[i_keep]
        # compute alpha and delta
        D2inv = np.zeros(D.shape)
        D2inv[di1[i_keep], di2[i_keep]] = 1 / s[i_keep] ** 2
        D2inv = np.matrix(D2inv)

    # transform everything needed in matrix
    V_T, = matrix_transform(V_T)


    return V_T.T*D2inv*V_T


def gcv_grad_comp(rho, X, Q, R, endog,sm_handler,var_list,return_par='gcv',gamma=1.):
    """
    A lot of shared computation for hess, gradient and evaluation... can be further improved
    """
    sm_handler.set_smooth_penalties(np.exp(rho),var_list)
    n_obs = Q.shape[0]
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    U, s, V_T = linalg.svd(np.vstack((R, B[:, :])))

    # remove low val singolar values
    i_rem = np.where(s < 10 ** (-8) * s.max())[0]

    # remove cols
    s = np.delete(s, i_rem, 0)
    U = np.delete(U, i_rem, 1)
    V_T = np.delete(V_T, i_rem, 0)

    # compute the diag matrix with the singolar vals
    D = np.zeros((s.shape[0], s.shape[0]))
    di = np.diag_indices(s.shape[0])
    D[di] = s

    ## check several steps
    U1 = U[:R.shape[0], :s.shape[0]]


    y = endog[:n_obs]
    Dinv = np.zeros(D.shape)
    Dinv[di] = 1 / s

    delta = n_obs - gamma*np.sum(inner1d(np.array(U1),np.array(U1)))
    # transform everything needed in matrix
    y,Q,R,U1,Dinv,V_T = matrix_transform(y,Q,R,U1,Dinv,V_T)
    y = np.matrix(endog[:n_obs].reshape(n_obs, 1))
    S_all = compute_Sjs(sm_handler,var_list)
    S_all = matrix_transform(*S_all)

    # compute alpha and delta
    D2inv = np.zeros(D.shape)
    D2inv[di] = 1 / s ** 2
    D2inv = np.matrix(D2inv)

    # parentheses necessary for fast computation (always matrix dot vector)
    Ay = X * (V_T.T * (D2inv * (V_T * (X.T * y))))
    alpha = np.sum(np.power(Ay - y, 2))
    # compute useful vector
    y1 = U1.T * (Q.T * y)
    UTU = U1.T * U1

    alpha_grad = np.zeros(rho.shape[0])
    delta_grad = np.zeros(rho.shape[0])

    if return_par == 'A':
        A_grad = np.zeros((rho.shape[0],n_obs,n_obs),dtype=np.float32)

    #compute derivs
    for j in range(rho.shape[0]):
        Sj = S_all[j]

        lamj = np.exp(rho[j])

        Mj = Dinv*V_T*Sj*V_T.T*Dinv
        Fj = Mj * UTU

        alpha_grad[j] = lamj*(2*y1.T*Mj*y1 - y1.T*Fj*y1 - y1.T*Fj.T*y1)
        delta_grad[j] = gamma*lamj*np.trace(Fj)

        if return_par == 'A':
            A_grad[j] = -lamj* Q*U1*Dinv*V_T*Sj*V_T.T*Dinv*U1.T*Q.T



    gcv_grad = (n_obs/delta**2) * alpha_grad - (2*n_obs*alpha/delta**3) * delta_grad

    if return_par == 'gcv':
        return gcv_grad
    elif return_par == 'alpha':
        return alpha_grad
    elif return_par == 'delta':
        return delta_grad
    elif return_par is 'A':
        return A_grad
    else:
        raise ValueError('unknow output specification')

def gcv_hess_comp(rho, X, Q, R, endog, sm_handler, var_list, return_par='gcv',gamma=1.):
    sm_handler.set_smooth_penalties(np.exp(rho),var_list)
    n_obs = Q.shape[0]
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B,dtype=np.float64)

    U, s, V_T = linalg.svd(np.vstack((R, B[:, :])))

    # remove low val singolar values
    i_rem = np.where(s < 10 ** (-8) * s.max())[0]

    # remove cols
    s = np.delete(s, i_rem, 0)
    U = np.delete(U, i_rem, 1)
    V_T = np.delete(V_T, i_rem, 0)

    # compute the diag matrix with the singular vals
    D = np.zeros((s.shape[0], s.shape[0]))
    di = np.diag_indices(s.shape[0])
    D[di] = s

    ## check several steps
    U1 = U[:R.shape[0], :s.shape[0]]

    y = endog[:n_obs]
    Dinv = np.zeros(D.shape)
    Dinv[di] = 1 / s

    # transform everything needed in matrix
    y, Q, R, U1, Dinv, V_T = matrix_transform(y, Q, R, U1, Dinv, V_T)
    y = np.matrix(endog[:n_obs].reshape(n_obs, 1))
    S_all = compute_Sjs(sm_handler, var_list)
    S_all = matrix_transform(*S_all)

    # compute alpha and delta
    delta = n_obs - gamma*np.sum(inner1d(np.array(U1),np.array(U1)))
    D2inv = np.zeros(D.shape)
    D2inv[di] = 1 / s ** 2
    D2inv = np.matrix(D2inv)

    # parentheses necessary for fast computation (always matrix dot vector)
    Ay = X * (V_T.T * (D2inv * (V_T * (X.T * y))))
    alpha = np.sum(np.power(Ay - y, 2))

    # compute useful vector
    y1 = U1.T * (Q.T * y)
    UTU = U1.T * U1

    alpha_hes = np.zeros(shape=(rho.shape[0],rho.shape[0]))
    delta_hes = np.zeros(shape=(rho.shape[0],rho.shape[0]))
    gcv_hes = np.zeros(shape=(rho.shape[0],rho.shape[0]))
    alpha_grad = np.zeros(shape=(rho.shape[0],))
    delta_grad = np.zeros(shape=(rho.shape[0],))

    if return_par == 'A':
        A_grad = np.zeros((rho.shape[0], n_obs, n_obs), dtype=np.float32)
        A_hes = np.zeros((rho.shape[0],rho.shape[0], n_obs, n_obs), dtype=np.float32)

    # compute derivs
    if return_par == 'A':
        for j in range(rho.shape[0]):
            Sj = S_all[j]
            lamj = np.exp(rho[j])
            A_grad[j] = -lamj * Q * U1 * Dinv * V_T * Sj * V_T.T * Dinv * U1.T * Q.T

    for j in range(rho.shape[0]):
        Sj = S_all[j]
        lamj = np.exp(rho[j])
        Mj = Dinv * V_T * Sj * V_T.T * Dinv
        Fj = Mj * UTU

        for k in range(rho.shape[0]):
            Sk = S_all[k]
            lamk = np.exp(rho[k])
            Mk = Dinv * V_T * Sk * V_T.T * Dinv
            Fk = Mk * UTU

            yT_AT_hes_A_y = lamj * lamk * y1.T * (Fk.T * Mj + Fj.T * Mk) * y1
            yT_hes_AT_A_y = lamj * lamk * y1.T * (Mj * Fk + Mk * Fj) * y1
            yT_hes_A_y = lamj*lamk*y1.T*(Mk*Mj + Mj*Mk)*y1
            yT_grad_Aj_grad_Ak_y = lamj*lamk*y1.T*Fk*Mj*y1

            delta_hes[j,k] = -2*lamj*lamk*np.trace(Mk*Fj)*gamma

            if j == k:
                yT_AT_hes_A_y = yT_AT_hes_A_y - lamj * y1.T * Fj.T * y1
                yT_hes_AT_A_y = yT_hes_AT_A_y - lamj * y1.T * Fj * y1
                yT_hes_A_y = yT_hes_A_y - lamj * y1.T * Mj * y1

                delta_hes[j, k] = delta_hes[j,k] + (lamj * np.trace(Fj))*gamma

            alpha_hes[j,k] = -2*yT_hes_A_y + yT_hes_AT_A_y  + yT_AT_hes_A_y + 2*yT_grad_Aj_grad_Ak_y

            if return_par == 'A':
                A_hes[j,k] = lamj*lamk* Q*U1*Dinv*V_T*(Sk*V_T.T*D2inv*V_T*Sj + Sj*V_T.T*D2inv*V_T*Sk) * V_T.T * Dinv * U1.T * Q.T

                if j == k:
                    A_hes[j,k] = A_hes[j,k] + A_grad[j]

            gcv_hes[j,k] = - (2*n_obs/(delta**3)) * delta_grad[k] * alpha_grad[j] \
                           + (n_obs/(delta)**2) * alpha_hes[j,k] \
                           - (2*n_obs/(delta)**3) * alpha_grad[k] * delta_grad[j]\
                           + (6*n_obs*alpha/(delta)**4) * delta_grad[j] * delta_grad[k]\
                           - (2*n_obs*alpha/(delta)**3) * delta_hes[j,k]


    if return_par == 'gcv':
        return gcv_hes
    elif return_par == 'alpha':
        return alpha_hes
    elif return_par == 'delta':
        return delta_hes
    elif return_par == 'A':
        return A_hes
    else:
        raise ValueError('unknow output specification')



def norm_compute(*M):
    norms = []
    for mm in M:
        norms += [np.linalg.norm(mm,ord='fro')]
    return np.argmax(norms)

def single_similarity_transform(lamSall):

    k = norm_compute(*lamSall)
    Sk = lamSall[k]
    e,U = np.linalg.eig(Sk)
    keep = e != 0
    Ur = U[:,keep]
    Un = U[:,~keep]
    S_not_k = 0
    lamSall_red = []
    for i in range(len(lamSall)):
        if i == k:
            continue
        S_not_k = S_not_k + lamSall[i]
        lamSall_red += [np.dot(np.dot(Un.T,lamSall[i]),Un)]
    D = np.diag(e[keep])
    SS11 = D + np.dot(np.dot(Ur.T,S_not_k),Ur)
    SS21 = np.dot(np.dot(Un.T, S_not_k), Ur)
    SS12 = np.dot(np.dot(Ur.T, S_not_k), Un)
    SS22 = np.dot(np.dot(Un.T, S_not_k), Un)
    SS = np.block([[SS11,SS12],[SS21,SS22]])

    return SS,SS11,lamSall_red



if __name__ == '__main__':

    np.random.seed(4)
    link = sm.genmod.families.links.log()
    family = sm.genmod.families.family.Gamma(link=link)
    tp = 5 * 10 ** 4
    x1, x2, x3 = np.random.uniform(0.05, 1, size=tp), np.random.uniform(0, 1, size=tp), np.random.uniform(-2, 2,
                                                                                                          size=tp)
    smooth_pen = np.array([1.5, .51,2])
    rho0 = np.log(smooth_pen)
    xs = [x1, x2, x3]

    # define smooth handler
    sm_handler = smooths_handler()
    sm_handler.add_smooth('1d_var', [x1], ord=4, knots=None, knots_num=10, perc_out_range=0.25,
                          is_cyclic=[False], lam=[smooth_pen[0]])
    sm_handler.add_smooth('1d_var2', [x2,x3], ord=4, knots=None, knots_num=10, perc_out_range=0.25,
                          is_cyclic=[False,True], lam=[smooth_pen[1],smooth_pen[2]])
    var_list = ['1d_var', '1d_var2']

    num_samp = tp
    gamma = 1.5

    y = 2+np.sin(np.linspace(0,100,tp))

    mu = y + 10**-5
    eta = family.link(mu)
    n_obs = y.shape[0]
    mu = family.link.inverse(eta)
    data_model = weights_and_data(y, family, fisher_scoring=True)
    z, w = data_model.get_params(mu)
    # exog = sqrt(diag(w)).dot(X) the X' pg. 272 in GAM texbook
    t0 = perf_counter()
    exog, endog, index_var = sm_handler.get_general_additive_model_endog_and_exog(var_list, z, weights=w)
    t1 = perf_counter()
    print('prepare variables time: %s sec'%(t1-t0))
    model = sm.OLS(endog, exog)
    fit_OLS = model.fit()

    eta = np.dot(exog[:n_obs, :], fit_OLS.params)
    X = exog[:n_obs, :]

    if flagUseCuda:
        cuda_linalg.init()
        X_gpu = gpuarray.to_gpu(np.array(X, order='F'))
        # tmp = X_gpu[2:10,3:6]
        # tmp1 = X[2:10,3:6]
        # print(tmp1-tmp.get())
        print('\n\n')
        # help(linalg.qr)
        print('\n\n')
        t0 = perf_counter()
        Q_gpu, R_gpu = cuda_linalg.qr(X_gpu, mode='reduced')
        t1 = perf_counter()
        print('QR-cuda time',t1-t0)
        Q = Q_gpu.get()
        R = R_gpu.get()
        # check decomp
        t0 = perf_counter()
        Q1, R1 = np.linalg.qr(X, 'reduced')
        t1 = perf_counter()
        print('QR-normal',t1-t0)
        DQ = (Q-Q1)
        DR = (R-R1)
        print('range dQ: (%f,%f)'%(DQ.min(),DQ.max()))
        print('range dR: (%f,%f)' % (DR.min(), DR.max()))
    else:
        Q, R = np.linalg.qr(X,'reduced')



    ## gcv from ols
    hat_diag = fit_OLS.get_influence().hat_matrix_diag[:n_obs]
    trA = hat_diag.sum()
    # print('TRA',trA)
    # print('trA form old',trA)
    rsd = endog[:n_obs] - fit_OLS.fittedvalues[:n_obs]
    rss = np.sum(np.power(rsd, 2))
    # print('alpha', rss)
    sig_hat = rss / (n_obs - trA)
    gcv = sig_hat * n_obs / (n_obs - trA)

    #debug only

    # decomp based
    t0 = perf_counter()
    gcv1 = gcv_comp(rho0, X,Q,R,endog,sm_handler,var_list)
    t1= perf_counter()
    print('gcv diff',gcv - gcv1)


    rho0 = np.array(rho0,dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    R = np.array(R, dtype=np.float64)
    endog = np.array(endog,dtype=np.float64)
    grad = gcv_grad_comp(rho0,X,Q,R,endog,sm_handler,var_list,return_par='alpha',gamma=gamma)
    func = lambda rho : gcv_comp(rho,X,Q,R,endog,sm_handler,var_list,return_par='alpha',gamma=gamma)
    grad_app = approx_grad(rho0,grad.shape,func,5*10 ** -3)
    if checkGrad(grad,grad_app,tol=10**-3):
        print('alpha gradient possibly wrong')

    grad = gcv_grad_comp(rho0,X, Q, R, endog, sm_handler, var_list, return_par='delta',gamma=gamma)
    func = lambda rho: gcv_comp(rho,X, Q, R, endog, sm_handler, var_list, return_par='delta',gamma=gamma)
    grad_app = approx_grad(rho0, grad.shape, func, 5*10 ** -3)
    if checkGrad(grad,grad_app,tol=10**-3):
        print('delta gradient possibly wrong')

    grad = gcv_grad_comp(rho0,X, Q, R, endog, sm_handler, var_list, return_par='gcv',gamma=gamma)
    func = lambda rho: gcv_comp(rho, X, Q, R, endog, sm_handler, var_list, return_par='gcv',gamma=gamma)
    grad_app = approx_grad(rho0, grad.shape, func, 5*10 ** -3)
    if checkGrad(grad,grad_app,tol=10**-3):
        print('gcv gradient possibly wrong')


    grad = gcv_hess_comp(rho0,X, Q, R, endog, sm_handler, var_list, return_par='alpha',gamma=gamma)
    func = lambda rho: gcv_grad_comp(rho,X, Q, R, endog, sm_handler, var_list, return_par='alpha',gamma=gamma)
    grad_app = approx_grad(rho0, grad.shape, func, 5*10 ** -3)
    if checkGrad(grad,grad_app,tol=10**-3):
        print('alpha hessian possibly wrong')

    grad = gcv_hess_comp(rho0,X, Q, R, endog, sm_handler, var_list, return_par='delta',gamma=gamma)
    func = lambda rho: gcv_grad_comp(rho,X, Q, R, endog, sm_handler, var_list, return_par='delta',gamma=gamma)
    grad_app = approx_grad(rho0, grad.shape, func, 5* 10 ** -4)
    if checkGrad(grad,grad_app,tol=10**-3):
        print('\ndelta hessian possibly wrong\n')
    else:
        print('\ndelta grad ok!\n')

    grad = gcv_hess_comp(rho0,X, Q, R, endog, sm_handler, var_list, return_par='gcv',gamma=gamma)
    func = lambda rho: gcv_grad_comp(rho,X, Q, R, endog, sm_handler, var_list, return_par='gcv',gamma=gamma)
    grad_app = approx_grad(rho0, grad.shape, func, 5 * 10 ** -3)
    if checkGrad(grad, grad_app, tol=10 ** -3):
        print('\ngcv hessian possibly wrong\n')
    else:
        print('\ngcv hessian ok!\n')
