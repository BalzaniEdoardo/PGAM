import cython
cimport scipy.linalg.cython_blas as blas
import numpy as np
from cython.parallel import prange



@cython.boundscheck(False)
@cython.wraparound(False)
def kron_cython(double[:, ::1] a, double[:, ::1] b):
    cdef int i = a.shape[0]
    cdef int j = a.shape[1]
    cdef int k = b.shape[0]
    cdef int l = b.shape[1]
    cdef int onei = 1
    cdef double oned = 1
    cdef int m, n



    result = np.zeros((i*k, j*l), dtype=np.double)
    cdef double[:, ::1] result_v = result
    for n in prange(i,nogil=True):
        for m in range(k):
            blas.dger(&l, &j, &oned, &b[m, 0], &onei, &a[n, 0], &onei, &result_v[m+k*n, 0], &l)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def rowwise_kron_cython(double[:, ::1] a, double[:, ::1] b):
    cdef int i = a.shape[0]
    cdef int j = a.shape[1]
    cdef int l = b.shape[1]
    cdef int onei = 1
    cdef double oned = 1
    cdef int m, n,p,q



    result = np.zeros((i, j*l), dtype=float)
    temp = np.zeros((1, j*l), dtype=np.double)
    a_reshape = np.zeros((1, j), dtype=np.double)
    b_reshape = np.zeros((1, l), dtype=np.double)
    cdef double[:, ::1] result_v = result
    cdef double[:, ::1] temp_v = temp

    cdef double xx = 0


    for m in range(i):
        for p in range(j):
            a_reshape[0,p] = a[m,p]
        for q in range(l):
            b_reshape[0,q] = b[m,q]
        temp_v = kron_cython(a_reshape,b_reshape)
        for n in range(j*l):
            result_v[m,n] = temp_v[0,n]


    return result

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def kron_cython(my_type[:, ::1] a, my_type[:, ::1] b):
#     cdef int i = a.shape[0]
#     cdef int j = a.shape[1]
#     cdef int k = b.shape[0]
#     cdef int l = b.shape[1]
#     cdef my_type onei = 1
#     cdef my_type oned = 1
#
#     result = np.zeros((i*k, j*l), float)
#     cdef my_type[:, ::1] result_v = result
#
#     cdef int n,m
#     for n in prange(i,nogil=True):
#         for m in range(k):
#             blas.dger(&l, &j, &oned, &b[m, 0], &onei, &a[n, 0], &onei, &result_v[m+k*n, 0], &l)
#     return result