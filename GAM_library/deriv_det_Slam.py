import numpy as np
from der_wrt_smoothing import *
from gam_data_handlers import *
import scipy.stats as sts
from numpy.core.umath_tests import inner1d

def symmetrize_tensor(S_tens):
    new_tens = np.zeros(S_tens.shape)
    cc = 0
    for S in S_tens:
        Ssym = np.triu(S) + np.triu(S,1).T
        new_tens[cc] = Ssym
        cc+=1

    return new_tens

def transform_Slam(S_all,rho):
    FLOAT_EPS = np.finfo(float).eps
    # preprocess to remove 0 eig
    lam = np.exp(rho)
    S_tensor = np.zeros((rho.shape[0],)+S_all[0].shape)
    S_tensor[:,:,:] = S_all
    S_tensor = symmetrize_tensor(S_tensor)
    sum_S_norm = 0
    for Si in S_tensor:
        sum_S_norm = sum_S_norm + Si / np.linalg.norm(Si,ord='fro')

    Slam = np.einsum('ijk,i->jk',S_tensor,lam)
    Slam = np.triu(Slam) + np.triu(Slam, 1).T

    d_tild,U_tild = np.linalg.eigh(Slam)
    idx = d_tild > FLOAT_EPS
    U_plus = U_tild[:,idx]
    # print('eigRange Slam: ',np.min(d_tild[idx]),np.max(d_tild[idx]),'\nlam: ',lam)

    S_transformed = np.einsum('ji,rjk,kh->rih',U_plus,S_tensor,U_plus)
    # should be symmetric... resymmetrize here
    S_transformed = symmetrize_tensor(S_transformed)


    # initialize algorithm
    Slam = np.einsum('kih,k->ih', S_transformed, lam)
    K = 0
    Q = Slam.shape[0]
    S_tild = S_transformed.copy()
    M = S_tild.shape[0]
    index_gamma = np.arange(M)
    totDim = Slam.shape[0]

    while True:
        # step 1

        Omega = np.zeros(index_gamma.shape[0])
        cc = 0
        for j in index_gamma:
            Omega[cc] = np.linalg.norm(S_tild[j]*lam[j],ord='fro')
            cc += 1

        # step 2
        try:
            select_alpha = Omega >= FLOAT_EPS**(1/3.) * Omega.max()
        except:
            print('Except raised')
            pass
        index_alpha = index_gamma[select_alpha]
        Omega_alpha = Omega[select_alpha]
        index_gamma_prime = index_gamma[~select_alpha]

        # step 3
        normSum = np.einsum('ijk,i->jk',S_tild[index_alpha],1/Omega_alpha)
        e,_ = np.linalg.eigh(normSum)
        r = np.sum(e > e.max()*(FLOAT_EPS**0.7))

        # step 4
        if r == Q:
            break

        # step 5
        S_alpha =  np.einsum('kih,k->ih', S_tild[index_alpha], lam[index_alpha])
        S_gamma_prime = np.einsum('kih,k->ih', S_tild[index_gamma_prime], lam[index_gamma_prime])

        d,U = np.linalg.eigh(S_alpha)

        # check that the eig works
        # REC = np.dot(np.dot(U,np.diag(d)),U.T)
        # diff2 = REC - S_alpha
        # print(np.linalg.norm(diff2))
        ## end check

        descending_idx = np.argsort(d)[::-1]
        d = d[descending_idx]
        U = U[:,descending_idx]
        Ur = U[:,:r]
        Un = U[:,r:]
        Dr = np.diag(d[:r])

        # step 6
        A = Slam[:K,:K]
        B = Slam[:K,K:]
        B_prime = np.dot(B,U)

        M11 = np.einsum('ji,jk,kl->il',Ur,S_gamma_prime,Ur)
        M12 = np.einsum('ji,jk,kl->il', Ur, S_gamma_prime, Un)
        M21 = np.einsum('ji,jk,kl->il', Un, S_gamma_prime, Ur)
        M22 = np.einsum('ji,jk,kl->il', Un, S_gamma_prime, Un)
        C_prime = np.block([[Dr+M11, M12],[M21,M22]])

        S_prime = np.block([[A,B_prime],[B_prime.T,C_prime]])

        T_alpha = np.block([[np.eye(K), np.zeros((K,r)), np.zeros((K,totDim-K-r))],[np.zeros((Q,K)), Ur, np.zeros((Q,totDim-K-r))]])
        T_gamma = np.block([[np.eye(K), np.zeros((K,Q))],[np.zeros((Q,K)),U]])

        S_transformed[index_alpha] = np.einsum('ji,rjk,kh->rih',T_alpha,S_transformed[index_alpha],T_alpha)
        S_transformed[index_gamma_prime] = np.einsum('ji,rjk,kh->rih',T_gamma,S_transformed[index_gamma_prime],T_gamma)

        # step 8

        tmp = np.einsum('ji,rjk,kh->rih',Un,S_tild[index_gamma_prime],Un)
        S_tild = np.zeros((M, Un.shape[1], Un.shape[1]))
        S_tild[index_gamma_prime] = tmp
        # step 9
        K = K + r
        Q = Q - r
        Slam = S_prime
        index_gamma = index_gamma_prime
        if len(index_gamma_prime) == 0:
            break
    return Slam,S_transformed


def logDet_Slam(rho,S_transf,compute_grad=False,S_all=None):
    lam = np.exp(rho)
    if compute_grad:
        _, S_transf = transform_Slam(S_all, rho)
    Slam = np.einsum('ijk,i',S_transf,lam)
    ## check eig lam

    try:
    # stable det compute
        Pinv = np.diag(1 / np.sqrt(np.abs(np.diag(Slam))))
        P = np.diag(np.sqrt(np.abs(np.diag(Slam))))
        L = np.linalg.cholesky(np.einsum('ij,jh,hk->ik', Pinv, Slam, Pinv))
        log_det = 2 * np.sum(np.log(np.diag(L))) + 2 * np.sum(np.log(np.diag(P)))
    except np.linalg.LinAlgError:
        # try simple eig decomp
        Slam = np.triu(Slam) + np.triu(Slam, 1).T
        d_tild, U_tild = np.linalg.eigh(Slam)
        idx = d_tild > np.finfo(float).eps
        log_det = np.sum(np.log(d_tild[idx]))

    return log_det

def grad_logDet_Slam(rho,S_transf,compute_grad=False,S_all=None):
    lam = np.exp(rho)
    if compute_grad:
        _, S_transf = transform_Slam(S_all, rho)
    Slam = np.einsum('ijk,i', S_transf, lam)
    try:
        ## stable square root after transforming S
        Pinv = np.diag(1/np.sqrt(np.abs(np.diag(Slam))))
        L = np.linalg.cholesky(np.einsum('ij,jh,hk->ik',Pinv,Slam,Pinv))
        Linv = np.linalg.pinv(L)
        Sinv = np.einsum('ij,kj,kh,hl->il',Pinv,Linv,Linv,Pinv)
    except np.linalg.LinAlgError:
        Slam = np.triu(Slam) + np.triu(Slam, 1).T
        d_tild, U_tild = np.linalg.eigh(Slam)
        idx = d_tild > np.finfo(float).eps
        U_tild = U_tild[:,idx]
        Utmp = U_tild * (1/np.sqrt(d_tild[idx]))
        Sinv = np.dot(Utmp,Utmp.T)

    grad_det = np.zeros((rho.shape[0],))
    for j in range(rho.shape[0]):
        grad_det[j] = lam[j] * np.sum(inner1d(Sinv, S_transf[j].T))
    return grad_det

def hes_logDet_Slam(rho,S_transf):
    lam = np.exp(rho)

    Slam = np.einsum('ijk,i', S_transf, lam)
    try:
        ## stable square root after transforming S
        Pinv = np.diag(1 / np.sqrt(np.abs(np.diag(Slam))))
        L = np.linalg.cholesky(np.einsum('ij,jh,hk->ik', Pinv, Slam, Pinv,optimize='optimal'))
        Linv = np.linalg.pinv(L)
        Sinv = np.einsum('ij,kj,kh,hl->il', Pinv, Linv, Linv, Pinv,optimize='optimal')

    except np.linalg.LinAlgError:
        Slam = np.triu(Slam) + np.triu(Slam, 1).T
        d_tild, U_tild = np.linalg.eigh(Slam)
        idx = d_tild > np.finfo(float).eps
        U_tild = U_tild[:,idx]
        Utmp = U_tild * (1/np.sqrt(d_tild[idx]))
        Sinv = np.dot(Utmp,Utmp.T)

    hes_det = np.zeros((rho.shape[0],rho.shape[0]))
    tmp_dict = {}
    for i in range(rho.shape[0]):
        # Sinv_Si = np.einsum('ij,jk->ik', Sinv, S_transf[i])
        for j in range(rho.shape[0]):
            if i == 0:
                Sinv_Sj = np.einsum('ij,jk->ik', Sinv, S_transf[j]) # use symmetry to half the time
                tmp_dict[j] = Sinv_Sj
            else:
                Sinv_Sj = tmp_dict[j]
            Sinv_Si = tmp_dict[i]
            hes_det[i,j] = -lam[j]*lam[i] * np.sum(inner1d(Sinv_Si,Sinv_Sj.T))
            if i == j:
                hes_det[i, j] = hes_det[i,j] + lam[i] * np.sum(inner1d(Sinv, S_transf[i].T))
    return hes_det



if __name__ == '__main__':
    from gam_data_handlers import *
    np.random.seed(4)

    tp = 1 * 10 ** 3
    x1, x2, x3 = np.random.uniform(0.05, 1, size=tp), np.random.uniform(0, 1, size=tp), np.random.uniform(-2, 2,
                                                                                                          size=tp)
    xs = [x1, x2, x3]

    # define smooth handler
    sm_handler = smooths_handler()
    sm_handler.add_smooth('1d_var', [x1], ord=4, knots=None, knots_num=10, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)
    sm_handler.add_smooth('1d_var2', [x2], ord=4, knots=None, knots_num=10, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)
    sm_handler.add_smooth('1d_var3', [x3], ord=4, knots=None, knots_num=15, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)

    sm_handler.add_smooth('1d_var4', [x3], ord=4, knots=None, knots_num=15, perc_out_range=0.0,
                          is_cyclic=[False], lam=None,penalty_type='der',der=2)
    var_list = ['1d_var', '1d_var2','1d_var3','1d_var4']
    # Define a gamma variable


    rho = np.array([14,14.1,-13,1]*2)
    S_all = compute_Sjs(sm_handler,var_list)
    # S_all[1] = S_all[1]#*10**8
    # S_all[0] = S_all[0]# * 10 ** 8
    # S_all[2] = S_all[2]# *10**-2

    Slam = create_Slam(rho,sm_handler,var_list)
    Slam_trans,S_transf = transform_Slam(S_all, rho)

    func = lambda rho : logDet_Slam(rho,S_transf,compute_grad=True,S_all=S_all)
    grad_log_det = grad_logDet_Slam(rho, S_transf)
    app_grad = approx_grad(rho, grad_log_det.shape, func, 10 ** -4)

    func2 = lambda rho :grad_logDet_Slam(rho, S_transf,compute_grad=True,S_all=S_all)
    hes_log_det = hes_logDet_Slam(rho, S_transf)
    app_grad2 = approx_grad(rho, hes_log_det.shape, func2, 10 ** -4)

