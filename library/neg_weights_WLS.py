import numpy as np
import scipy.linalg as linalg
import sys
from copy import deepcopy
from newton_optim import weights_and_data
import scipy.stats as sts
from time import perf_counter

def pivoted_QR(A,return_Q=True):
    # fortran function for fast pivoted QR decomp
    geqp3 = linalg.get_lapack_funcs('geqp3')
    # function for reconstructing the matrix Q of the QR
    orgqr = linalg.get_lapack_funcs('orgqr')


    QR,piv,tau,_,_ = geqp3(np.array(A,order='F'))
    # triu is the R matrix
    R = np.triu(QR)
    if return_Q:
        # Q can be constructed by the tirl of QR and the weights tau (compact representation)
        Q,_,_ = orgqr(QR,tau)

    #  index are 1 based in fortran
    piv = piv - 1
    # reverse permutation
    piv_revert = np.argsort(piv)
    R = np.array(R,order='C')
    if return_Q:
        return R, piv, piv_revert, Q
    return R, piv, piv_revert


def example_robust_WLS(w,z,X,S_list=[],lambda_list=[]):
    """
    FUNCTION FOR PATHOLOGICAL EXAMPLE TESTING
    """
    w_abs = np.abs(w)
    # w_minus = w.copy()
    # w_minus[w>=0] = 0
    # w_minus = -w_minus
    sqrtWX = (X.T * np.sqrt(w_abs)).T
    QQ, RR = np.linalg.qr(sqrtWX)
    z_bar = z.copy()
    z_bar[w < 0] = -z[w < 0]
    if S_list == []:
        S_list = [np.zeros((X.shape[1], X.shape[1]))]
        lambda_list = [0]
        R, piv, piv_revert, Q = pivoted_QR(np.vstack((RR / np.linalg.norm(RR, 'fro'))))


    else:
        M = np.zeros(S_list[0].shape)
        for S in S_list:
            M = M + S / np.linalg.norm(S, 'fro')
        try:
            E = np.linalg.cholesky(M)
        except:
            eig, U = np.linalg.eigh(M)
            sort_col = np.argsort(eig)[::-1]
            # Sx2 = np.dot(np.dot(U[:,sort_col],np.diag(eig[sort_col])),U[:,sort_col].T)
            eig = eig[sort_col]
            U = U[:, sort_col]
            # matrix is sym should be positive
            eig = np.abs(eig)
            i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
            eig = np.delete(eig, i_rem, 0)
            E = np.zeros(U.shape)
            mask = np.arange(U.shape[1])
            mask = mask[np.delete(mask, i_rem, 0)]
            E[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
            E = E.T
        R, piv, piv_revert, Q = pivoted_QR(np.vstack((RR / np.linalg.norm(RR, 'fro'), E / np.linalg.norm(E, 'fro'))))

    # # get the rank of the R matrix
    rank = np.linalg.matrix_rank(R)
    keep = piv[:rank]
    keep.sort()
    # keep only the identifiable rows
    RR_sub = RR[:, keep]
    S_list_sub = deepcopy(S_list)

    for cc in range(len(S_list_sub)):
        S_list_sub[cc] = S_list_sub[cc][:, keep]
        S_list_sub[cc] = S_list_sub[cc][keep, :]

    # create the total penalty matrix
    S_tensor = np.zeros((len(S_list_sub),) + S_list_sub[0].shape)
    S_tensor[:, :, :] = S_list_sub
    S_tensor = (S_tensor.T * lambda_list).T
    S_lam = S_tensor.sum(axis=0)

    if all(S_lam.flatten() < 10 ** -12):
        R, piv_2, piv_revert_2, Q = pivoted_QR(np.vstack((RR_sub)))
    else:
        try:
            E = np.linalg.cholesky(S_lam).T
        except:
            eig, U = np.linalg.eigh(S_lam)
            sort_col = np.argsort(eig)[::-1]
            # Sx2 = np.dot(np.dot(U[:,sort_col],np.diag(eig[sort_col])),U[:,sort_col].T)
            eig = eig[sort_col]
            U = U[:, sort_col]
            # matrix is sym should be positive, so maybe small negatives due to comp noise are zeros
            eig = np.abs(eig)
            i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
            eig = np.delete(eig, i_rem, 0)
            E = np.zeros(U.shape)
            mask = np.arange(U.shape[1])
            mask = mask[np.delete(mask, i_rem, 0)]
            E[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
            E = E.T

        R, piv_2, piv_revert_2, Q = pivoted_QR(np.vstack((RR_sub, E)))
    R = R[:R.shape[1], :]
    Q1 = np.dot(QQ, Q[:X.shape[1], :])
    i_diag = np.ones(w.shape[0])
    i_diag[w > 0] = 0

    sele_row = i_diag != 0
    if sele_row.sum() == 0:
        P = np.linalg.pinv(R[:, piv_revert_2])
        K = Q1
        beta = np.dot(np.dot(P, K.T), np.sqrt(w_abs) * z_bar)
    else:
        # check (temporary
        # print(np.max(np.abs(sqrtWX-np.dot(Q1,R[:,piv_revert_2]))))
        # print(np.max(np.abs(np.dot(R[:,piv_revert_2].T,R[:,piv_revert_2]) - 2 * np.dot(X.T*w_minus,X) - np.dot(X.T*w,X) - S_lam)))

        # print(np.max(np.abs(np.dot(np.dot(R[:,piv_revert_2].T,np.eye(R.shape[0]) - 2*np.dot(Q1.T*i_diag,Q1)),R[:,piv_revert_2])- np.dot(X.T*w,X) - S_lam)))
        U, d, VT = linalg.svd((Q1[sele_row, :].T * i_diag[sele_row]).T, full_matrices=False, overwrite_a=True)
        # print(np.max(np.abs(np.dot(np.dot(R[:,piv_revert_2].T,np.dot(VT.T*(1-2*d**2),VT)),R[:,piv_revert_2]) - np.dot(X.T*w,X) - S_lam)))

        if any(d > 1 / np.sqrt(2)):
            # recompute weights
            print('Use Fisher scoring for the step')
            return robust_WLS(y, mu, family, X, S_list, lambda_list, fisher_scoring=True)
        else:
            # use full newton step
            P = np.dot(np.linalg.pinv(R[:, piv_revert_2]), VT.T / np.sqrt(1 - 2 * d ** 2))
            K = np.dot(Q1, VT.T / np.sqrt(1 - 2 * d ** 2))

        beta = np.dot(np.dot(P, K.T), np.sqrt(w_abs) * z_bar)

    beta_all = np.zeros(X.shape[1])
    beta_all[keep] = beta

    return beta_all, P, K, rank, z, w, keep


def robust_WLS(y,mu,family,X,f_weights_and_data,S_list=[],lambda_list=[], fisher_scoring=False,outputML=False):
    """
    FUNCTION FOR PATHOLOGICAL EXAMPLE TESTING
    """
    FLOAT_EPS = np.finfo(float).eps
    f_weights_and_data.fisher_scoring = fisher_scoring
    z, w = f_weights_and_data.get_params(mu)



    if any(np.isinf(z)):
        print('USE Fisher scoring')
        f_weights_and_data = weights_and_data(y, family, fisher_scoring=True)
        z, w = f_weights_and_data.get_params(mu)

    zero_rows = np.abs(w) < (FLOAT_EPS) ** 0.5
    w[zero_rows] = 0
    # if any(zero_rows):
    #     y = y.copy()[~zero_rows]
    #     X = X.copy()[~zero_rows,:]
    #     if not mu is None:
    #         mu = mu.copy()[~zero_rows]
    #     z = z[~zero_rows]
    #     w = w[~zero_rows]

    w_abs = np.abs(w)
    # w_minus = w.copy()
    # w_minus[w>=0] = 0
    # w_minus = -w_minus
    sqrtWX = (X.T*np.sqrt(w_abs)).T
    QQ,RR = np.linalg.qr(sqrtWX)
    z_bar = z.copy()
    z_bar[w<0] = -z[w<0]
    if S_list == []:
        S_list = [np.zeros((X.shape[1],X.shape[1]))]
        lambda_list = [0]
        R, piv, piv_revert, Q = pivoted_QR(np.vstack((RR / np.linalg.norm(RR, 'fro'))))


    else:
        M = np.zeros(S_list[0].shape)
        for S in S_list:

            M = M + S/np.linalg.norm(S,'fro')
        try:
            E = np.linalg.cholesky(M)
        except:
            eig, U = np.linalg.eigh(M)
            sort_col = np.argsort(eig)[::-1]
            # Sx2 = np.dot(np.dot(U[:,sort_col],np.diag(eig[sort_col])),U[:,sort_col].T)
            eig = eig[sort_col]
            U = U[:, sort_col]
            # matrix is sym should be positive
            eig = np.abs(eig)
            i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
            eig = np.delete(eig, i_rem, 0)
            E = np.zeros(U.shape)
            mask = np.arange(U.shape[1])
            mask = mask[np.delete(mask, i_rem, 0)]
            E[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
            E = E.T
        R,piv,piv_revert,Q = pivoted_QR(np.vstack((RR/np.linalg.norm(RR,'fro'),E/np.linalg.norm(E,'fro'))))


    # # get the rank of the R matrix
    rank = np.linalg.matrix_rank(R)
    keep = piv[:rank]
    keep.sort()
    # keep only the identifiable rows
    RR_sub = RR[:,keep]
    S_list_sub= deepcopy(S_list)

    for cc in range(len(S_list_sub)):
        S_list_sub[cc] = S_list_sub[cc][:,keep]
        S_list_sub[cc] = S_list_sub[cc][keep,:]

    # create the total penalty matrix
    S_tensor = np.zeros((len(S_list_sub),) + S_list_sub[0].shape)
    S_tensor[:, :, :] = S_list_sub
    S_tensor = (S_tensor.T * lambda_list).T
    S_lam = S_tensor.sum(axis=0)

    if all(S_lam.flatten()<10**-12):
        R, piv_2, piv_revert_2, Q = pivoted_QR(np.vstack((RR_sub)))
    else:
        try:
            E = np.linalg.cholesky(S_lam).T
        except:
            eig, U = np.linalg.eigh(S_lam)
            sort_col = np.argsort(eig)[::-1]
            # Sx2 = np.dot(np.dot(U[:,sort_col],np.diag(eig[sort_col])),U[:,sort_col].T)
            eig = eig[sort_col]
            U = U[:, sort_col]
            # matrix is sym should be positive, so maybe small negatives due to comp noise are zeros
            eig = np.abs(eig)
            i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
            eig = np.delete(eig, i_rem, 0)
            E = np.zeros(U.shape)
            mask = np.arange(U.shape[1])
            mask = mask[np.delete(mask, i_rem, 0)]
            E[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
            E = E.T


        R,piv_2,piv_revert_2,Q = pivoted_QR(np.vstack((RR_sub,E)))
    R = R[:R.shape[1],:]
    Q1 = np.dot(QQ,Q[:X.shape[1],:])
    i_diag = np.ones(w.shape[0])
    i_diag[w >= 0] = 0

    sele_row = i_diag != 0
    if sele_row.sum() == 0:
        P = np.linalg.pinv(R[:,piv_revert_2])
        K = Q1
        beta = np.dot(np.dot(P, K.T), np.sqrt(w_abs) * z_bar)
        d = None
    else:
        # check (temporary
        # print(np.max(np.abs(sqrtWX-np.dot(Q1,R[:,piv_revert_2]))))
        # print(np.max(np.abs(np.dot(R[:,piv_revert_2].T,R[:,piv_revert_2]) - 2 * np.dot(X.T*w_minus,X) - np.dot(X.T*w,X) - S_lam)))

        # print(np.max(np.abs(np.dot(np.dot(R[:,piv_revert_2].T,np.eye(R.shape[0]) - 2*np.dot(Q1.T*i_diag,Q1)),R[:,piv_revert_2])- np.dot(X.T*w,X) - S_lam)))
        U, d, VT = linalg.svd((Q1[sele_row,:].T * i_diag[sele_row]).T, full_matrices=False,overwrite_a=True)
        # print(np.max(np.abs(np.dot(np.dot(R[:,piv_revert_2].T,np.dot(VT.T*(1-2*d**2),VT)),R[:,piv_revert_2]) - np.dot(X.T*w,X) - S_lam)))

        if any(d > 1/np.sqrt(2)):
            # recompute weights
            print('Use Fisher scoring for the step')
            return robust_WLS(y,mu,family,X,weights_and_data,S_list,lambda_list,fisher_scoring=True)
        else:
            # use full newton step
            P = np.dot(np.linalg.pinv(R[:,piv_revert_2]),VT.T/np.sqrt(1-2*d**2))
            K = np.dot(Q1,VT.T/np.sqrt(1-2*d**2))


        beta = np.dot(np.dot(P,K.T),np.sqrt(w_abs)*z_bar)

    beta_all = np.zeros(X.shape[1])
    beta_all[keep] = beta
    if outputML:
        return beta_all,P,d,R,w,z,S_lam,zero_rows,piv_2,piv_revert_2
    return beta_all,P,K,rank,z,w,keep

def statsmodels_WLS(y,mu,family,X,S_list=[],lambda_list=[], fisher_scoring=False):
    """
    FUNCTION FOR PATHOLOGICAL EXAMPLE TESTING
    """
    f_weights_and_data = weights_and_data(y, family, fisher_scoring=fisher_scoring)
    z, w = f_weights_and_data.get_params(mu)
    if len(S_list)>0:
        S_tensor = np.zeros((len(S_list),) + S_list[0].shape)
        S_tensor[:, :, :] = S_list
        S_tensor = (S_tensor.T * lambda_list).T
        S_lam = S_tensor.sum(axis=0)
        # check solution
    else:
        S_lam = np.zeros((X.shape[1],)*2)

    try:
        E = np.linalg.cholesky(S_lam).T
    except:
        eig, U = np.linalg.eigh(S_lam)
        sort_col = np.argsort(eig)[::-1]
        # Sx2 = np.dot(np.dot(U[:,sort_col],np.diag(eig[sort_col])),U[:,sort_col].T)
        eig = eig[sort_col]
        U = U[:, sort_col]
        # matrix is sym should be positive, so maybe small negatives due to comp noise are zeros
        eig = np.abs(eig)
        i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
        eig = np.delete(eig, i_rem, 0)
        E = np.zeros(U.shape)
        mask = np.arange(U.shape[1])
        mask = mask[np.delete(mask, i_rem, 0)]
        E[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
        E = E.T
    M = np.vstack((X,E))
    z_hat = np.hstack((z,np.zeros(S_lam.shape[0])))
    w_hat = np.hstack((w,np.zeros(S_lam.shape[0])))
    model = sm.WLS(z_hat,M,weights=w_hat)
    res_fit = model.fit()
    return res_fit.params,z,w


if __name__ == "__main__":
    import matplotlib.pylab as plt
    import statsmodels.api as sm
    tp = 10**5
    np.random.seed(1000)
    full_newton = True
    low_rank = True
    lams = []# np.random.uniform(size=3)
    S_list = []
    for k in range(len(lams)):
        A = np.random.normal(size=(1,10))
        S_list += [np.dot(A.T,A)]

    X = np.random.normal(loc=1,size=(tp,10))
    X = np.hstack((X, np.ones((tp, 0))))

    # positive use fulll newton
    if full_newton:
        w = np.random.uniform(size=tp)
        ii = np.random.choice(np.arange(tp),150,replace=False)
        w[ii] = -w[ii]
    else:
        w = np.random.normal(size=tp)


    beta_0 = np.random.normal(size=X.shape[1])
    z = np.random.normal(loc=np.dot(X,beta_0), size=tp)
    beta,P,K ,rank,a,b,c = example_robust_WLS(w,z,X,S_list=S_list,lambda_list=lams)

    if len(S_list)>0:
        S_tensor = np.zeros((len(S_list),) + S_list[0].shape)
        S_tensor[:, :, :] = S_list
        S_tensor = (S_tensor.T * lams).T
        S_lam = S_tensor.sum(axis=0)

        # check solution
        delta = np.dot(np.dot(X.T*w,X) + S_lam,beta) - np.dot(X.T*w,z)
    else:
        delta = np.dot(np.dot(X.T * w, X) , beta) - np.dot(X.T * w, z)
    print(np.max(np.abs(delta)))

    link = sm.families.links.sqrt()
    family = sm.families.family.Poisson(link=link)
    y = np.random.poisson(family.link.inverse(np.dot(X,-beta_0)))

    mu_true = family.link.inverse(np.dot(X,-np.random.normal(size=10)))
    # mu = np.clip(np.dot(X,-beta_0 + 0.001),0.001,np.inf)
    mu = mu_true.copy()

    if len(S_list)>0:
        S_tensor = np.zeros((len(S_list),) + S_list[0].shape)
        S_tensor[:, :, :] = S_list
        S_tensor = (S_tensor.T * lams).T
        S_lam = S_tensor.sum(axis=0)
        # check solution
    else:
        S_lam = np.zeros((X.shape[1],)*2)
    f_weights_and_data = weights_and_data(y,family,fisher_scoring=False)

    for k in range(100):
        t0 = perf_counter()
        beta, P, K, rank,zz, ww, keep = robust_WLS(y, mu, family, X,f_weights_and_data, S_list=S_list, lambda_list=lams, fisher_scoring=False)
        t1 = perf_counter()
        print('neg weight',t1-t0)
        t0 = perf_counter()
        beta1,z1,w1 = statsmodels_WLS(y, mu, family, X, S_list=S_list, lambda_list=lams, fisher_scoring=False)
        t1 = perf_counter()
        print('pos weight', t1 - t0)
        mu = family.link.inverse(np.dot(X, beta))
        assert(np.max(np.abs(zz-z1))==0)
        assert (np.max(np.abs(w1 - w1)) == 0)
        delta = np.dot(np.dot(X.T*ww,X) + S_lam,beta) - np.dot(X.T*ww,zz)
        delta1 = np.dot(np.dot(X.T*w1,X) + S_lam,beta1) - np.dot(X.T*w1,z1)
        print(k,np.max(np.abs(delta)),np.max(np.abs(delta1)))


    # plt.scatter(np.dot(X,beta),z)
    # plt.scatter(np.dot(X, np.random.normal(size=X.shape[1])), z)