# cython: infer_types=True
import numpy as np
cimport cython
from cython.parallel import prange

ctypedef fused my_type:
    int
    double
    long long


cdef my_type clip(my_type a, my_type min_value, my_type max_value):
    return min(max(a, min_value), max_value)


@cython.boundscheck(False)
@cython.wraparound(False)
def hessian_H_summation_1(my_type[:, ::1] X, my_type[:, ::1] dB, my_type[::1] h_prime, my_type[::1] g_prime_inv):
    #np.einsum('ki,kj,kl,ky,k,k,hy,rl->hrij',X,X,X,X,h_prime,1/g_prime,dB,dB)
    n_obs = X.shape[0]
    n_param = X.shape[1]
    n_smooth = dB.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((n_smooth, n_smooth, n_param, n_param), dtype=dtype)
    XdBdr = np.zeros((n_obs,n_smooth),dtype=dtype)


    cdef my_type[: , :, :, ::1] result_view = result
    cdef my_type[: , ::1] XdBdr_view = XdBdr

    cdef Py_ssize_t k1,k, y, l, i, j, h

    for k in range(n_obs):
        for r in range(n_smooth):
            for l in range(n_param):
                XdBdr_view[k,r] = XdBdr_view[k,r] + X[k,l] * dB[r,l]

    for k in range(n_obs):
        for r in range(n_smooth):
            for i in prange(n_param,nogil=True):
                for j in range(n_param):
                    for h in range(n_smooth):
                        result_view[h,r,i,j] = result_view[h,r,i,j] + X[k,i] * X[k,j] * XdBdr_view[k,r] * XdBdr_view[k,h] * h_prime[k] * g_prime_inv[k]


    return result



#np.einsum('ki,kj,kl,k,hrl->hrij',X,X,X,h,d2B)

@cython.boundscheck(False)
@cython.wraparound(False)
def hessian_H_summation_2(my_type[:, ::1] X, my_type[:, :, ::1] d2B, my_type[::1] h_func):
    n_obs = X.shape[0]
    n_param = X.shape[1]
    n_smooth = d2B.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((n_smooth, n_smooth, n_param, n_param), dtype=dtype)
    cdef my_type[: , :, :, ::1] result_view = result

    Xd2Bdhr = np.zeros((n_obs,n_smooth,n_smooth),dtype=dtype)
    cdef my_type[:, :, ::1] Xd2Bdhr_view = Xd2Bdhr


    cdef Py_ssize_t k, y, l, i, j, h
    for k in range(n_obs):
        for r in range(n_smooth):
            for h in range(n_smooth):
                for l in range(n_param):
                    Xd2Bdhr_view[k, h,r] = Xd2Bdhr_view[k,h,r] + X[k,l] * d2B[h, r, l]

    for h in range(n_smooth):
        for r in range(n_smooth):
            for i in prange(n_param,nogil=True):
                for j in range(n_param):
                    for k in range(n_obs):
                        result_view[h,r,i,j] = result_view[h,r,i,j] + X[k,i] * X[k,j] * Xd2Bdhr_view[k,h,r] * h_func[k]

    return result

#np.einsum('ki,lk,kj,rl->rij', X, w_prime, X, dB)
@cython.boundscheck(False)
@cython.wraparound(False)
def grad_H_summation(my_type[:, ::1] X, my_type[:, ::1] dB, my_type[:, ::1] w_prime):
    n_obs = X.shape[0]
    n_param = X.shape[1]
    n_smooth = dB.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((n_smooth, n_param, n_param), dtype=dtype)
    cdef my_type[: , :, ::1] result_view = result

    wprime_dB = np.zeros((n_obs,n_smooth))
    cdef my_type[: , ::1] wprime_dB_view = wprime_dB


    cdef Py_ssize_t k, l, i, j, r
    for r in range(n_smooth):
        for k in range(n_obs):
            for l in range(n_param):
                wprime_dB_view[k,r] = wprime_dB_view[k,r] + w_prime[l,k] * dB[r, l]

    for r in range(n_smooth):
        for i in prange(n_param,nogil=True):
            for j in range(n_param):
                for k in range(n_obs):
                    result_view[r,i,j] = result_view[r,i,j] + X[k,i] * X[k,j] * wprime_dB_view[k,r]

    return result

# np.einsum('ij,hjl,lr,krp,p->hki', neg_sum_inv, grad_neg_sum, -neg_sum_inv, dSlam_drho, b_hat)
@cython.boundscheck(False)
@cython.wraparound(False)
def d2beta_hat_summation_1(my_type[:, ::1] neg_sum_inv, my_type[:, :, ::1] grad_neg_sum, my_type[:, :, ::1] dSlam_drho,
                         my_type[::1] b_hat):

    n_param = dSlam_drho.shape[1]
    n_smooth = dSlam_drho.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result_tmp = np.zeros((n_smooth, n_smooth, n_param), dtype=dtype)
    prod_tmp = np.zeros((n_smooth,n_param))
    cdef my_type[: , :, ::1] result_tmp_view = result_tmp

    cdef my_type[:, ::1] prod = prod_tmp

    cdef Py_ssize_t k, l, i, j, r, h, p
    for l in range(n_param):
        for k in range(n_smooth):
            for p in range(n_param):
                for r in range(n_param):
                    prod[k,l] =  prod[k,l] + (- neg_sum_inv[l,r]) * dSlam_drho[k,r,p] * b_hat[p]

    for h in range(n_smooth):
        for k in range(n_smooth):
            for i in prange(n_param,nogil=True):
                for j in range(n_param):
                    for l in range(n_param):
                        result_tmp_view[h,k,i] = result_tmp_view[h,k,i] + neg_sum_inv[i,j] * grad_neg_sum[h,j,l] * prod[k,l]

    return result_tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def d2beta_hat_summation_1_old(my_type[:, ::1] neg_sum_inv, my_type[:, :, ::1] grad_neg_sum, my_type[:, :, ::1] dSlam_drho,
                         my_type[::1] b_hat):

    n_param = dSlam_drho.shape[1]
    n_smooth = dSlam_drho.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((n_smooth, n_smooth, n_param), dtype=dtype)
    prod_tmp = np.zeros((n_smooth,n_param))
    cdef my_type[: , :, ::1] result_view = result

    cdef Py_ssize_t k, l, i, j, r, h, p


    for h in range(n_smooth):
        for k in range(n_smooth):
            for i in prange(n_param,nogil=True):
                for j in range(n_param):
                    for l in range(n_param):
                        for p in range(n_param):
                            for r in range(n_param):
                                result_view[h,k,i] = result_view[h,k,i] + neg_sum_inv[i,j] * grad_neg_sum[h,j,l] * (- neg_sum_inv[l,r]) * dSlam_drho[k,r,p] * b_hat[p]




    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def d2beta_hat_summation_2(my_type[:, ::1] neg_sum_inv, my_type[:, :, ::1] dSlam_drho, my_type[:,::1] grad_beta):

    n_param = dSlam_drho.shape[1]
    n_smooth = dSlam_drho.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    prod_tmp = np.zeros((n_smooth, n_smooth, n_param), dtype=dtype)
    result_tmp = np.zeros((n_smooth, n_smooth, n_param), dtype=dtype)

    cdef my_type[: , :, ::1] result_tmp_view = result_tmp
    cdef my_type[:,:,::1] prod = prod_tmp

    cdef Py_ssize_t k, l, i, j, h
    for j in prange(n_param,nogil=True):
        for l in range(n_param):
            for k in range(n_smooth):
                for h in range(n_smooth):
                    prod[h,k,j] = prod[h,k,j] + dSlam_drho[k,j,l] * grad_beta[h,l]

    for h in prange(n_smooth,nogil=True):
        for k in range(n_smooth):
            for j in range(n_param):
                for i in range(n_param):
                    result_tmp_view[h,k,i] = result_tmp_view[h,k,i] + neg_sum_inv[i,j] * prod[h,k,j]

    return result_tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def trace_log_det_H_summation_1(my_type[:, ::1] Vb_inv, my_type[:, :, ::1] dVb, my_type[:, :, :, ::1] d2Vb):
    # A = np.einsum('ij,hjk->hik',Vb_inv,dVb)
    # 0.5*np.einsum('hij,rji->hr',A,A) that's what should replicate
    n_param = dVb.shape[1]
    n_smooth = dVb.shape[0]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((n_smooth, n_smooth), dtype=dtype)
    cdef my_type[: , ::1] result_view = result

    VinvDVB = np.zeros((n_smooth, n_param, n_param), dtype=dtype)
    cdef my_type[: , :, ::1] VinvDVB_view = VinvDVB

    cdef Py_ssize_t k, l, i, j, h
    for h in range(n_smooth):
        for i in range(n_param):
            for k in range(n_param):
                for j in range(n_param):
                    VinvDVB_view[h,i,k] = VinvDVB_view[h,i,k]  +  Vb_inv[i,j] * dVb[h,j,k]

    cdef Py_ssize_t k1, l1, i1, j1, h1
    for h1 in range(n_smooth):
        for r1 in range(n_smooth):
            for i1 in range(n_param):
                for j1 in range(n_param):
                    result_view[h1,r1] = result_view[h1,r1] + VinvDVB_view[h1,i1,j1] * VinvDVB_view[r1,j1,i1] - Vb_inv[i1,j1]*d2Vb[h1,r1,j1,i1]


    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def d3beta_unpenalized_ll_summation(my_type[:, ::1] X, my_type[::1] dalpha_dmu, my_type[::1] dmu_deta):
    # np.einsum('i,i,im,ir,il->mrl',dalpha_dmu,dmu_deta,X,X,X)
    n_obs = X.shape[0]
    n_param = X.shape[1]


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    result = np.zeros((n_param,n_param,n_param),dtype=dtype)
    cdef my_type[:,:,::1] result_view = result

    cdef Py_ssize_t i,r,l,m
    for i in range(n_obs):
        for m in prange(n_param,nogil=True):
            for l in range(n_param):
                for r in range(n_param):
                    result_view[m,r,l] = result_view[m,r,l] + X[i,m] * X[i,r] * X[i,l] * dmu_deta[i] * dalpha_dmu[i]
    return result
