from copy import deepcopy

import numpy as np
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import statsmodels.api as sm

try:
    import kron_cython

    use_fast_kron = True
except:
    use_fast_kron = False
import warnings
from time import perf_counter

import scipy.signal as signal
from numba import jit
from scipy.integrate import dblquad, simps
from scipy.spatial import Delaunay


def splineDesign(knots, x, ord=4, der=0, outer_ok=False):
    """Reproduces behavior of R function splineDesign() for use by ns(). See R documentation for more information.

    Python code uses scipy.interpolate.splev to get B-spline basis functions, while R code calls C.
    Note that der is the same across x."""
    knots = np.array(knots, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    xorig = x.copy()
    not_nan = ~np.isnan(xorig)
    nx = x.shape[0]
    knots.sort()
    nk = knots.shape[0]
    need_outer = any(x[not_nan] < knots[ord - 1]) or any(x[not_nan] > knots[nk - ord])
    in_x = (x >= knots[0]) & (x <= knots[-1]) & not_nan

    if need_outer:
        if outer_ok:
            # print('knots do not contain the data range')

            out_x = ~all(in_x)
            if out_x:
                x = x[in_x]
                nnx = x.shape[0]
            dkn = np.diff(knots)[::-1]
            reps_start = ord - 1
            if any(dkn > 0):
                reps_end = max(0, ord - np.where(dkn > 0)[0][0] - 1)
            else:
                reps_end = np.nan  # this should give an error, since all knots are the same
            idx = [0] * (ord - 1) + list(range(nk)) + [nk - 1] * reps_end
            knots = knots[idx]
        else:
            raise ValueError("the 'x' data must be in the range %f to %f unless you set outer_ok==True'" % (
                knots[ord - 1], knots[nk - ord]))
    else:
        reps_start = 0
        reps_end = 0
    if (not need_outer) and any(~not_nan):
        x = x[in_x]
    idx0 = reps_start
    idx1 = len(knots) - ord - reps_end
    cycleOver = np.arange(idx0, idx1)
    m = len(knots) - ord
    v = np.zeros((cycleOver.shape[0], len(x)), dtype=np.float64)
    # v = np.zeros((m, len(x)))

    d = np.eye(m, len(knots))
    for i in range(cycleOver.shape[0]):
        v[i] = interpolate.splev(x, (knots, d[cycleOver[i]], ord - 1), der=der)
        # v[i] = interpolate.splev(x, (knots, d[i], ord - 1), der=der)

    # before = np.sum(xorig[not_nan] < knots[0])
    # after = np.sum(xorig[not_nan] > knots[-1])
    design = np.zeros((v.shape[0], xorig.shape[0]), dtype=np.float64)
    for i in range(v.shape[0]):
        #     design[i, before:xorig.shape[0] - after] = v[i]
        design[i, in_x] = v[i]

    return design.transpose()


def cSplineDes(knots, x, ord=4, der=0):
    """
    Description:
    ===========
    This function is equivalent to cSplineDes of R, it generates the matrix of a cyclic B-spline for given knots
    evaluated in the points x.
    """
    knots = np.array(knots)
    x = np.array(x)
    nk = knots.shape[0]
    if ord < 2:
        raise ValueError('order too low')
    if (nk < ord):
        raise ValueError('too few knots')
    knots.sort()
    k1 = knots[0]
    if x.min() < k1 or x.max() > knots[-1]:
        raise ValueError('x out of range')
    xc = knots[nk - ord]
    knots = np.hstack((k1 - knots[-1] + knots[nk - ord:nk - 1], knots))
    ind = x > xc
    X1 = splineDesign(knots, x, ord=ord, der=der, outer_ok=True)
    x[ind] = x[ind] - knots.max() + k1
    if np.sum(ind):
        X2 = splineDesign(knots, x[ind], ord=ord, outer_ok=True, der=der)
        X1[ind,] = X1[ind,] + X2
    return X1


def smPenalty_1D_BSpline(k):
    P = - np.eye(k) + np.diag(np.ones(k - 1), 1)
    P = P[:k - 1, :]
    M = np.dot(P.T, P)
    return sparse.csr_matrix(M, dtype=np.float64)


def smPenalty_1D_cyclicBSpline(k):
    P = - np.eye(k) + np.diag(np.ones(k - 1), 1)
    P = P[:k - 1, :]
    M = np.dot(P.T, P)
    M[0, :] = np.roll(M[1, :], -1)
    M[-1, :] = np.roll(M[-2, :], 1)
    return sparse.csr_matrix(M)


def smoothPen_sqrt(M):
    # find a square root matrix
    eig, U = np.linalg.eigh(M)
    sort_col = np.argsort(eig)[::-1]
    # Sx2 = np.dot(np.dot(U[:,sort_col],np.diag(eig[sort_col])),U[:,sort_col].T)
    eig = eig[sort_col]
    U = U[:, sort_col]
    # matrix is sym should be positive
    eig = np.abs(eig)
    i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
    eig = np.delete(eig, i_rem, 0)
    Bx = np.zeros(U.shape)
    mask = np.arange(U.shape[1])
    mask = mask[np.delete(mask, i_rem, 0)]
    Bx[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
    Bx = Bx.T
    return Bx


def non_eqSpaced_diff_pen(knots, order, outer_ok=False, cyclic=False):
    assert (all(knots[:order] == knots[0]))
    assert (all(knots[-order:] == knots[-1]))
    if order == 1:
        int_knots = knots
    else:
        int_knots = knots[order - 1: -(order - 1)]

    if not cyclic:
        Ak = splineDesign(knots, int_knots, ord=order, der=0, outer_ok=False)
        Amid = splineDesign(knots, int_knots[:-2] + int_knots[2:] - int_knots[1:-1], ord=order, der=0, outer_ok=False)
    else:
        Ak = cSplineDes(knots, int_knots, ord=order, der=0)
        Amid = cSplineDes(knots, int_knots[:-2] + int_knots[2:] - int_knots[1:-1], ord=order, der=0)

    B = np.zeros((Ak.shape[0] - 2, Ak.shape[1]))

    for i in range(Ak.shape[1]):
        B[:, i] = (Ak[:-2, i] - Ak[1:-1, i] - Amid[:, i] + Ak[2:, i])

    M = sparse.csr_matrix(np.dot(B.T, B))
    return M, B


def smPenalty_1D_derBased(knots, xmin, xmax, n_points, ord=4, der=1, outer_ok=False, cyclic=False,
                          measure=None):
    """
    Derivative based penalty
    :param knots:
    :param xmin:
    :param xmax:
    :param n_points:
    :param ord:
    :param der:
    :param outer_ok:
    :param cyclic:
    :return:
    """
    if measure is None:
        measure = lambda x: 1

    x = np.linspace(xmin, xmax, n_points)

    mux = measure((x - xmin) / (xmax - xmin))
    dx = x[1] - x[0]
    if not cyclic:
        D = splineDesign(knots, x, ord=ord, der=der, outer_ok=outer_ok)
    else:
        x = x[1:-1]
        D = cSplineDes(knots, x, ord=ord, der=der)
    M = np.zeros((D.shape[1], D.shape[1]))
    for i in range(D.shape[1]):
        for j in range(i, D.shape[1]):
            M[i, j] = simps(D[:, i] * D[:, j] * mux, dx=dx)
    M = M + np.triu(M, 1).T
    Bx = smoothPen_sqrt(M)
    M = sparse.csr_matrix(M)

    return M, Bx


def adaptiveSmoother_1D_derBased(knots, xmin, xmax, n_points, ord_AD=3, ad_smooth_basis_size=4, ord=4, der=2,
                                 outer_ok=False, cyclic=False):
    knots_ADSM = np.hstack(
        ([xmin] * (ord_AD - 1), np.linspace(xmin, xmax, ad_smooth_basis_size), [xmax] * (ord_AD - 1)))
    x = np.linspace(xmin, xmax, n_points)

    if cyclic:
        kX = cSplineDes(knots_ADSM, x, ord=ord_AD, der=der)
    else:
        kX = splineDesign(knots_ADSM, x, ord=ord_AD, outer_ok=outer_ok)

    dx = x[1] - x[0]
    if not cyclic:
        D = splineDesign(knots, x, ord=ord, der=der, outer_ok=outer_ok)
    else:
        D = cSplineDes(knots, x, ord=ord, der=der)

    k_dim_basis = kX.shape[1]
    M_list = []  # np.zeros((k_dim_basis, D.shape[1], D.shape[1]))
    Bx = []
    for k in range(k_dim_basis):
        M = np.zeros((D.shape[1], D.shape[1]))
        for i in range(D.shape[1]):
            for j in range(i, D.shape[1]):
                M[i, j] = simps(D[:, i] * D[:, j] * kX[:, k], dx=dx)
        M[:, :] = M[:, :] + np.triu(M[:, :], 1).T
        Bx += [smoothPen_sqrt(M[:, :])]
        M_list += [M]
    return M_list, Bx


@jit(nopython=True,parallel=False)
def sumtriangles( xy, z, triangles ):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
        # z concave or convex => under or overestimates
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    # assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    # assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = 0#np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = np.abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = np.abs( np.linalg.det( t )) / dimfac  # v slow
        zsum += area * np.mean(z[tri])
        areasum += area
    return (zsum, areasum)

def smPenalty_Disk(knot_list, n_points,
                   ord=4, der=1, outer_ok=False, cyclic=[False, False],
                   measure=None, check_tri=False, domain_fun=lambda x: np.ones(x.shape, dtype=bool)):
    if len(knot_list) != 2:
        raise ValueError('not implemented for D != 2')


    # get basis dim
    if not cyclic[0]:
        spdes_x = lambda x: splineDesign(knot_list[0], x, ord=ord, der=0, outer_ok=outer_ok)
        spdes_der_x = lambda x: splineDesign(knot_list[0], x, ord=ord, der=der, outer_ok=outer_ok)
    else:
        spdes_x = lambda x: cSplineDes(knot_list[0], x, ord=ord, der=0)
        spdes_der_x = lambda x: cSplineDes(knot_list[0], x, ord=ord, der=der)

    if not cyclic[1]:
        spdes_y = lambda y: splineDesign(knot_list[1], y, ord=ord, der=0, outer_ok=outer_ok)
        spdes_der_y = lambda y: splineDesign(knot_list[1], y, ord=ord, der=der, outer_ok=outer_ok)
    else:
        spdes_y = lambda y: cSplineDes(knot_list[1], y, ord=ord, der=0)
        spdes_der_y = lambda y: cSplineDes(knot_list[1], y, ord=ord, der=der)

    # create a grid of points including the knots contained in the circle
    knots_x = np.unique(knot_list[0])
    knots_y = np.unique(knot_list[1])
    x_grid = []
    y_grid = []
    for ix in range(knots_x.shape[0] - 1):
        xmin, xmax = knots_x[ix], knots_x[ix + 1]
        x_grid = np.hstack((x_grid, np.linspace(xmin, xmax, n_points // (knots_x.shape[0] - 1))[:-1]))
    x_grid = np.hstack((x_grid, knots_x[-1:]))

    for iy in range(knots_y.shape[0] - 1):
        ymin, ymax = knots_y[iy], knots_y[iy + 1]
        y_grid = np.hstack((y_grid, np.linspace(ymin, ymax, n_points // (knots_y.shape[0] - 1))[:-1]))
    y_grid = np.hstack((y_grid, knots_y[-1:]))

    # create mash grid

    in_disk = domain_fun  #
    X, Y = np.meshgrid(x_grid, y_grid)
    X = X.flatten()
    Y = Y.flatten()
    keep = in_disk([X, Y])
    X = X[keep]
    Y = Y[keep]
    XY = np.zeros((X.shape[0], 2))
    XY[:, 0] = X
    XY[:, 1] = Y
    tri = Delaunay(XY)

    if check_tri:
        import matplotlib.pylab as plt
        xx, yy = np.meshgrid(knots_x, knots_y)
        plt.scatter(X[::1], Y[::1])
        plt.scatter(xx, yy, color='r')
        plt.triplot(XY[:, 0], XY[:, 1], tri.simplices)

    # contaiiner for derivatives
    Sx = np.array(spdes_der_x(X), dtype=np.double, order='C')
    Sy = np.array(spdes_y(Y), dtype=np.double, order='C')
    bxxa = kron_cython.rowwise_kron_cython(Sx, Sy)

    Sx = np.array(spdes_x(X), dtype=np.double, order='C')
    Sy = np.array(spdes_der_y(Y), dtype=np.double, order='C')
    bayy = kron_cython.rowwise_kron_cython(Sx, Sy)
    del Sx, Sy

    Jx = np.zeros((bayy.shape[1],) * 2)
    Jy = np.zeros((bayy.shape[1],) * 2)

    k = 0
    for k1 in range(bayy.shape[1]):
        for k2 in range(k1, bayy.shape[1]):
            t0 = perf_counter()
            Jx[k1, k2] = sumtriangles(XY, bxxa[:, k1] * bxxa[:, k2], tri.simplices)[0]
            Jy[k1, k2] = sumtriangles(XY, bayy[:, k1] * bayy[:, k2], tri.simplices)[0]
            t1 = perf_counter()
            print('%d/%d' % (k, bayy.shape[1] * (bayy.shape[1] + 1) / 2), t1 - t0)
            k += 1

    Jx = Jx + np.triu(Jx, 1).T
    Jy = Jy + np.triu(Jy, 1).T

    return [Jx, Jy]


def smPenalty_ND_spline(*Ms):
    """
    Description:
    ============
    This function compute the penalty matrix for N-Dim variables. lam are the parameters controlling wiggling in
    different dimensions. if None, it is set to one for all dim.
    For the n-dim spline, form a grid of parameters B_{i1,...,in}, the penalty for the spine wiggling
        vec(B)^T \cdot J vec(B)
    Ms[j] will be the matrix for the wiggling penalty for the jth 1Dim coordinate of the ND-spline
    """
    N = len(Ms)
    J = []

    for k in range(N):

        KP = Ms[k]
        for j in range(k):
            I = sparse.csr_matrix(np.eye(Ms[j].shape[0], dtype=np.float64))
            KP = sparse.kron(I, KP)
        for j in range(k + 1, N):
            I = sparse.csr_matrix(np.eye(Ms[j].shape[0]), dtype=np.float64)
            KP = sparse.kron(KP, I)
        J += [KP.toarray()]
    return J


def rowWiseKron(A, C):
    R = sparse.csr_matrix(np.zeros((A.shape[0], A.shape[1] * C.shape[1])))
    is_A_sparse = type(A) is sparse.csr.csr_matrix
    is_C_sparse = type(C) is sparse.csr.csr_matrix
    if is_A_sparse:
        A = np.array(A.toarray(), dtype=np.double)
    if is_C_sparse:
        C = np.array(C.toarray(), dtype=np.double)

    is_A_sparse = type(A) is sparse.csr.csr_matrix
    is_C_sparse = type(C) is sparse.csr.csr_matrix

    for i in range(A.shape[0]):
        if is_A_sparse and is_C_sparse:
            R[i, :] = sparse.kron(A[i, :], C[i, :])
        elif is_A_sparse:
            R[i, :] = np.kron(A[i, :].toarray().flatten(), C[i, :])
        elif is_C_sparse:
            R[i, :] = np.kron(A[i, :], C[i, :].toarray().flatten())
        else:
            if not use_fast_kron:
                R[i, :] = np.kron(A[i, :], C[i, :])
            else:
                rowA = A[i, :].reshape(1, A.shape[1])
                rowC = C[i, :].reshape(1, C.shape[1])
                R[i, :] = kron_cython.kron_cython(rowA, rowC)
    return R


def rowWiseKron_fullPython(a, b):
    i = a.shape[0]
    j = a.shape[1]
    l = b.shape[1]

    result = np.zeros((i, j * l), dtype=float)
    a_reshape = np.zeros((1, j), dtype=np.double)
    b_reshape = np.zeros((1, l), dtype=np.double)

    for m in range(i):
        for p in range(j):
            a_reshape[0, p] = a[m, p]
        for q in range(l):
            b_reshape[0, q] = b[m, q]
        result[m, :] = np.kron(a_reshape, b_reshape)

    return result


def multiRowWiseKron(*M, sparseX=True):
    KP = M[0]
    if len(M) == 1:
        if type(KP) is sparse.csr.csr_matrix and sparseX:
            return KP
        elif sparseX:
            return sparse.csr_matrix(KP, dtype=np.float64)
        else:
            return KP

    if type(KP) != np.ndarray:
        KP = KP.toarray()
    KP = np.array(KP, dtype=np.double, order='C')
    for X in M[1:]:
        if type(X) != np.ndarray:
            X = X.toarray()
        X = np.array(X, dtype=np.double, order='C')
        if use_fast_kron:
            KP = kron_cython.rowwise_kron_cython(KP, X)
        else:
            KP = rowWiseKron_fullPython(KP, X)

    if sparseX:
        return sparse.csr_matrix(KP, dtype=np.float64)
    else:
        return KP


def basisAndPenalty(x, knots, xmin=None, xmax=None, penalty_type='EqSpaced', der=1, n_points=10 ** 4, is_cyclic=None,
                    ord=4, sparseX=True, measure=None, ord_AD=3, ad_knots=4,
                    domain_fun=lambda x: np.ones(x.shape, dtype=bool), compute_pen=True,
                    prercomp_SandB=None):
    if penalty_type == 'EqSpaced':
        return basisAndPenalty_EqSpaced(x, knots, is_cyclic=None, ord=ord, sparseX=sparseX, split_range=None)

    elif penalty_type == 'diff':
        return basisAndPenalty_diff(x, knots, is_cyclic=is_cyclic, outer_ok=True, order=ord, sparseX=sparseX,
                                    split_range=None)

    elif penalty_type == 'der':
        return basisAndPenalty_deriv(x, knots, xmin, xmax, n_points, ord=ord, der=der, outer_ok=True,
                                     is_cyclic=is_cyclic, sparseX=sparseX,
                                     measure=measure)
    elif penalty_type == 'adaptive':
        return basisAndPenalty_Adaptive(x, knots, xmin, xmax, n_points, ord_AD=ord_AD, ad_smooth_basis_size=ad_knots,
                                        ord=ord, der=der, outer_ok=True, is_cyclic=is_cyclic, sparseX=sparseX,
                                        )
    elif penalty_type == 'der_2Ddomain':
        return basisAndPenalty_disk(x, knots, 200, ord=ord, der=der, outer_ok=True,
                                    is_cyclic=is_cyclic, sparseX=sparseX, extra_pen=1, domain_fun=domain_fun,
                                    compute_pen=compute_pen,
                                    prercomp_SandB=prercomp_SandB)


def basisAndPenalty_deriv(x, knots, xmin, xmax, n_points, ord=4, der=1, outer_ok=True,
                          is_cyclic=False, sparseX=True, extra_pen=1, measure=None):
    """

    :param x: input covariate, can be >1 dim (exponential increase number of param with dim)
    :param knots: knot vec (if multidim use a tensor product spline)
    :param xmin: inf of covariate domain (usually max(knots))
    :param xmax: sup of covariate domain (usually max(knots))
    :param n_points: points used for approx spline integral
    :param ord: oder of the spline
    :param der: order of the derivative inside the penalty
    :param outer_ok: bool. accept x outside the knot range
    :param is_cyclic: bool. if var is cyclic
    :param sparseX: return a sparse repr of X
    :param extra_pen: 0 if no extra pen, 1 if add null space pen
    :param measure: default 1, or a function for non-uniform penalization
    :return:
    """
    FLOAT_EPS = np.finfo(float).eps
    dim_spline = len(x)
    if is_cyclic is None:
        is_cyclic = np.zeros(dim_spline, dtype=bool)

    Xs = []
    Bs = []
    Ms = []
    basis_dim = []
    for k in range(dim_spline):
        if is_cyclic[k]:
            Xs += [cSplineDes(knots[k], x[k], ord=ord, der=0)]
        else:
            Xs += [splineDesign(knots[k], x[k], ord=ord, der=0, outer_ok=True)]
        M0, B0 = smPenalty_1D_derBased(knots[k], xmin[k], xmax[k], n_points, ord=ord, der=der, outer_ok=outer_ok,
                                       cyclic=is_cyclic[k],
                                       measure=measure)

        Bs += [B0]
        Ms += [M0]

        basis_dim += [Xs[k].shape[1]]

    B_list = Bs
    if dim_spline > 1:
        S_list = smPenalty_ND_spline(*Ms)
    else:
        S_list = []
        for M0 in Ms:
            S_list += [M0.toarray()]

    if extra_pen:
        S_tens = np.zeros(((len(S_list),) + S_list[0].shape))
        S_tens[:, :, :] = S_list
        S = S_tens.sum(axis=0)
        eig, U = np.linalg.eigh(S)
        zero_idx = np.abs(eig) < FLOAT_EPS * np.max(eig)
        Utilde = U[:, zero_idx]
        S_list += [np.dot(Utilde, Utilde.T)]
        B_list += [Utilde.T]

    X = multiRowWiseKron(*Xs, sparseX=sparseX)

    return X, B_list, S_list, basis_dim


def basisAndPenalty_disk(x, knots, n_points, ord=4, der=1, outer_ok=True,
                         is_cyclic=False, sparseX=True, extra_pen=1, domain_fun=lambda x: np.ones(x.shape, dtype=bool),
                         compute_pen=True,
                         prercomp_SandB=None):
    """

    :param x: input covariate, can be >1 dim (exponential increase number of param with dim)
    :param knots: knot vec (if multidim use a tensor product spline)
    :param xmin: inf of covariate domain (usually max(knots))
    :param xmax: sup of covariate domain (usually max(knots))
    :param n_points: points used for approx spline integral
    :param ord: oder of the spline
    :param der: order of the derivative inside the penalty
    :param outer_ok: bool. accept x outside the knot range
    :param is_cyclic: bool. if var is cyclic
    :param sparseX: return a sparse repr of X
    :param extra_pen: 0 if no extra pen, 1 if add null space pen
    :param measure: default 1, or a function for non-uniform penalization
    :return:
    """
    FLOAT_EPS = np.finfo(float).eps
    dim_spline = len(x)
    if is_cyclic is None:
        is_cyclic = np.zeros(dim_spline, dtype=bool)
    assert (len(x) == 2)
    Xs = []
    Bs = []
    Ms = []
    basis_dim = []
    set_zero = ~domain_fun(x)

    for k in range(dim_spline):
        if is_cyclic[k]:
            Xs += [cSplineDes(knots[k], x[k], ord=ord, der=0)]
        else:
            Xs += [splineDesign(knots[k], x[k], ord=ord, der=0, outer_ok=True)]


        basis_dim += [Xs[k].shape[1]]
    if compute_pen:
        if prercomp_SandB is None:
            S_list = smPenalty_Disk(knots, n_points, ord=ord, der=der, outer_ok=outer_ok,
                                    cyclic=is_cyclic, check_tri=False, domain_fun=domain_fun)
            Bs = [smoothPen_sqrt(S_list[0]), smoothPen_sqrt(S_list[1])]
            B_list = Bs
            if extra_pen:
                S_tens = np.zeros(((len(S_list),) + S_list[0].shape))
                S_tens[:, :, :] = S_list
                S = S_tens.sum(axis=0)
                eig, U = np.linalg.eigh(S)
                zero_idx = np.abs(eig) < FLOAT_EPS * np.max(eig)
                Utilde = U[:, zero_idx]
                S_list += [np.dot(Utilde, Utilde.T)]
                B_list += [Utilde.T]
        else:
            S_list = prercomp_SandB['S_list']
            B_list = prercomp_SandB['B_list']


    else:
        B_list = None
        S_list = None

    X = multiRowWiseKron(*Xs, sparseX=False)
    X[set_zero] = 0
    if sparseX:
        X = sparse.csr_matrix(X, dtype=np.float64)

    return X, B_list, S_list, basis_dim


def basisAndPenalty_Adaptive(x, knots, xmin, xmax, n_points, ord_AD=3, ad_smooth_basis_size=4, ord=4, der=1,
                             outer_ok=True,
                             is_cyclic=False, sparseX=True, extra_pen=1):
    """
    Create an adaptive penalty by expanding the penalty measure with a low-dim spline basis
    useful if the smoothness level is non-constant.
    :param x:
    :param knots:
    :param xmin:
    :param xmax:
    :param n_points:
    :param ord_AD:
    :param ad_smooth_basis_size:
    :param ord:
    :param der:
    :param outer_ok:
    :param is_cyclic:
    :param sparseX:
    :param extra_pen:
    :return:
    """
    FLOAT_EPS = np.finfo(float).eps
    dim_spline = len(x)
    assert (dim_spline == 1)
    if is_cyclic is None:
        is_cyclic = np.zeros(dim_spline, dtype=bool)

    Xs = []

    if is_cyclic[0]:
        Xs += [cSplineDes(knots[0], x[0], ord=ord, der=0)]
    else:
        Xs += [splineDesign(knots[0], x[0], ord=ord, der=0, outer_ok=True)]
    S_list, B_list = adaptiveSmoother_1D_derBased(knots[0], xmin[0], xmax[0], n_points, ord_AD=ord_AD,
                                                  ad_smooth_basis_size=ad_smooth_basis_size, ord=ord, der=der,
                                                  outer_ok=outer_ok, cyclic=is_cyclic[0])

    basis_dim = [Xs[0].shape[1]]
    if extra_pen:
        S_tens = np.zeros(((len(S_list),) + S_list[0].shape))
        S_tens[:, :, :] = S_list
        S = S_tens.sum(axis=0)
        eig, U = np.linalg.eigh(S)
        zero_idx = np.abs(eig) < FLOAT_EPS * np.max(eig)
        Utilde = U[:, zero_idx]
        S_list += [np.dot(Utilde, Utilde.T)]
        B_list += [Utilde.T]

    X = multiRowWiseKron(*Xs, sparseX=sparseX)

    return X, B_list, S_list, basis_dim


def basisAndPenalty_diff(x, knots, is_cyclic=None, order=4, outer_ok=True, sparseX=True, split_range=None):
    """
    Description
    ===========
    This function compute the spline design matrix and the penalty matrix of an arbitrary dimensional spline.
    High dimensional splines will result in huge design matrix (see the row wise kron product).

    penaltyType = 'Equispaced'


    """
    dim_spline = len(x)
    assert (split_range is None)
    if is_cyclic is None:
        is_cyclic = np.zeros(dim_spline, dtype=bool)

    Xs = []
    Bs = []
    Ms = []
    basis_dim = []
    for k in range(dim_spline):
        M, B = non_eqSpaced_diff_pen(knots[k], order, outer_ok=outer_ok, cyclic=is_cyclic[k])
        if is_cyclic[k]:
            Xs += [cSplineDes(knots[k], x[k], ord=order, der=0)]
            # Xs[k] will be of shape (n samples x spline dimension)

        else:
            # this will raise valueError if x is not contained in the knots
            Xs += [splineDesign(knots[k], x[k], ord=order, der=0, outer_ok=True)]

        Bs += [B]
        Ms += [M]
        basis_dim += [Xs[k].shape[1]]

    B_list = Bs
    S_list = smPenalty_ND_spline(*Ms)

    X = multiRowWiseKron(*Xs, sparseX=sparseX)

    return X, B_list, S_list, basis_dim


def basisAndPenalty_EqSpaced(x, knots, is_cyclic=None, ord=4, sparseX=True, split_range=None):
    """
    Description
    ===========
    This function compute the spline design matrix and the penalty matrix of an arbitrary dimensional spline.
    High dimensional splines will result in huge design matrix (see the row wise kron product).

    penaltyType = 'Equispaced'


    """
    dim_spline = len(x)
    assert (split_range is None)
    if is_cyclic is None:
        is_cyclic = np.zeros(dim_spline, dtype=bool)

    Xs = []
    Bs = []
    Ms = []
    basis_dim = []
    for k in range(dim_spline):
        if is_cyclic[k]:
            Xs += [cSplineDes(knots[k], x[k], ord=ord, der=0)]
            # Xs[k] will be of shape (n samples x spline dimension)
            Bs += [smPenalty_1D_cyclicBSpline(Xs[k].shape[1])]
            Ms += [np.dot(Bs[-1].T, Bs[-1])]

        else:
            # this will raise valueError if x is not contained in the knots
            Xs += [splineDesign(knots[k], x[k], ord=ord, der=0, outer_ok=True)]
            # trivial case of a single step element
            if Xs[k].shape[1] == 1:
                Bs += [sparse.csr_matrix([[1.]], dtype=np.float64)]
            else:
                Bs += [smPenalty_1D_BSpline(Xs[k].shape[1])]
            Ms += [np.dot(Bs[-1].T, Bs[-1])]

        basis_dim += [Xs[k].shape[1]]

    B_list = Bs
    S_list = smPenalty_ND_spline(*Ms)

    X = multiRowWiseKron(*Xs, sparseX=sparseX)

    return X, B_list, S_list, basis_dim


def basis_temporal(X, basis_kernel, trial_idx, pre_trial_dur, post_trial_dur, time_bin, sparseX=True):
    # # x list of length 1 (1-dim temporal filter)
    if trial_idx is None:
        if len(X) > 1:
            raise ValueError('temporal kernel signal to be filtered should be 1-dim')
        x = X[0]
        Xc = np.zeros((x.shape[0], basis_kernel.shape[1]))
        for k in range(basis_kernel.shape[1]):
            kern_vec = basis_kernel[:, k].flatten()
            Xc[:, k] = signal.fftconvolve(x, kern_vec, mode='same')
    else:
        x = X[0]
        Xc = np.zeros((x.shape[0], basis_kernel.shape[1]))
        skip_start = int(np.ceil(pre_trial_dur / time_bin))
        skip_end = int(np.ceil(post_trial_dur / time_bin))
        if (trial_idx is None) or len(trial_idx) == 0:
            raise ValueError('must indicate trial indices for temp kernel')
        for k in range(basis_kernel.shape[1]):
            kern_vec = basis_kernel[:, k].flatten()
            for tr in np.unique(trial_idx):
                sel = np.where(trial_idx == tr)[0]
                sel = sel[skip_start:sel.shape[0] - skip_end]
                sel_conv = signal.fftconvolve(x[sel], kern_vec, mode='same')
                Xc[sel, k] = sel_conv
    if sparseX:
        return sparse.csr_matrix(Xc)
    else:
        return Xc


def fit_penalised_LS(y, X, M, sp):
    """
    Description
    ===========
    This function wants as an input a variable ab endogenous variable y (1D), the exogenous X (len(y),n_predictors),
    the penalty matrix M, and a scalar sp representing how much the penalty is weighted. It fits a penalized Least Square
    of the form
    B_hat =  argmin( ||y - B \cdot X)||^{2}_2 + sp \cdot B^T \cdot M \cdot B
    by agumenting y and X appropriately and fit a regular OLS
    """
    if type(M) is sparse.csr.csr_matrix and type(X) is sparse.csr.csr_matrix:
        Xagu = np.vstack((X.toarray(), np.sqrt(sp) * M.toarray()))

    elif type(M) is sparse.csr.csr_matrix:
        Xagu = np.vstack((X, np.sqrt(sp) * M.toarray()))

    elif type(X) is sparse.csr.csr_matrix:
        Xagu = np.vstack((X.toarray(), np.sqrt(sp) * M))

    else:
        Xagu = np.vstack((X, np.sqrt(sp) * M))

    yagu = np.hstack((y, np.zeros(M.shape[0])))
    Xagu = sm.add_constant(Xagu)
    model = sm.OLS(yagu, Xagu)
    model = model.fit()
    return model


class covarate_smooth(object):
    def __init__(self, x_cov, ord=4, knots=None, knots_num=15, perc_out_range=0.0, is_cyclic=None, lam=None,
                 is_temporal_kernel=False,
                 kernel_direction=0, kernel_length=21, knots_percentiles=(0, 100), penalty_type='EqSpaced', der=None,
                 trial_idx=None, time_bin=None, pre_trial_dur=None, post_trial_dur=None, penalty_measure=None,
                 event_input=True,
                 ord_AD=3, ad_knots=4, domain_fun=lambda x: np.ones(x.shape, dtype=bool), prercomp_SandB=None,
                 repeat_extreme_knots=True):
        """
            x_cov: n-dim sampled points in which to evaluate the basis function
            ord: number of coefficient of the spline (spline degree + 1)
            knots: list of knots to be used to constuct the spline basis
            knots_num: if knots are not given, number of equispaced knots to be generated
            perc_out_range: percentage of knots that are outside the x_cov range
            is_cyclic: (None set all covariates to non-cyclic), boolean vector flagging which coordinate is cyclic
            lam: smooth weight per coordinate
            is_temporal_kernek: boolean, if true convolve x with basis
            kernel_direction: used if is_temporal_kernel is true
                values:
                    - 0: if bidirectional
                    - -1: if negative
                    - 1: if positive
            measure: a function defined on [0,1] that will be rescaled linearly to [min(knots),max(knots)]

        """
        self.der = der
        self._x = np.array(x_cov, dtype=np.double)
        self.dim = self._x.shape[0]
        self._ord = ord
        self.nan_filter = np.array(np.sum(np.isnan(self._x), axis=0), dtype=bool)
        self.penalty_type = penalty_type
        self.trial_idx = trial_idx
        self.time_bin = time_bin
        self.pre_trial_dur = pre_trial_dur
        self.post_trial_dur = post_trial_dur
        self.is_event_input = event_input
        self.ord_AD = ord_AD
        self.ad_knots = ad_knots
        self.domain_fun = domain_fun
        if self.pre_trial_dur is None:
            self.pre_trial_dur = 0
        if self.post_trial_dur is None:
            self.post_trial_dur = 0

        if np.isscalar(self.der) and self.der < 2 or self.penalty_type == 'EqSpaced':
            self.extra_pen = 0  # no extra lambda term
        else:
            self.extra_pen = 1

        self.is_temporal_kernel = is_temporal_kernel
        self.kernel_direction = kernel_direction

        # flag which coord are cyclic
        self.set_cyclic(is_cyclic)

        if is_temporal_kernel:
            self._set_knots_temporal(knots_num, kernel_length, kernel_direction, knots=knots)

            self.eval_basis = self._eval_basis_temporal
            self.set_knots = self._set_knots_temporal
            self.eval_basis_and_penalty = self._eval_basis_and_penalty_temporal

        else:
            # set knots
            self._set_knots_spatial(knots, knots_num=knots_num,
                                    perc_out_range=perc_out_range,
                                    percentiles=knots_percentiles,
                                    is_cyclic=is_cyclic,
                                    repeat_extreme_knots=repeat_extreme_knots)

            self.eval_basis = self._eval_basis_spatial
            self.set_knots = self._set_knots_spatial
            self.eval_basis_and_penalty = lambda: self._eval_basis_and_penalty_spatial(prercomp_SandB=prercomp_SandB)

        # set measure for non-uniform penalties
        if not penalty_measure is None:
            self.measure = penalty_measure
        else:  # uniform penalizaiton
            self.measure = lambda x: 1

        # eval the smooths and the penalty
        self.X, self.B_list, self.S_list, self.colMean_X, self.basis_dim, self.basis_kernel = self.eval_basis_and_penalty()

        self.set_lam(lam)

    def __eq__(self, other):
        is_equal = True
        if self.dim != other.dim:
            return False
        for cc in range(self.dim):
            is_equal = is_equal and (all(self._x[cc] == other._x[cc]))

        is_equal = is_equal and self._ord == other._ord
        is_equal = is_equal and self.is_temporal_kernel == other.is_temporal_kernel
        if 'kernel_direction' in self.__dict__.keys() and 'kernel_direction' in other.__dict__.keys():
            is_equal = is_equal and self.kernel_direction == other.kernel_direction

        is_equal = is_equal and all(self.is_cyclic == other.is_cyclic)
        for cc in range(self.dim):
            is_equal = is_equal and all(self.knots[cc] == other.knots[cc])

        if self.time_pt_for_kernel is None:
            is_equal = is_equal and (other.time_pt_for_kernel is None)
        else:
            is_equal = is_equal and all(self.time_pt_for_kernel == other.time_pt_for_kernel)
        is_equal = is_equal and all((self.X == other.X).__dict__['data'])
        is_equal = is_equal and self.basis_dim == other.basis_dim
        if self.basis_kernel is None:
            is_equal = is_equal and (other.basis_kernel is None)
        else:
            is_equal = is_equal and all((self.basis_kernel == other.basis_kernel).__dict__['data'])

        cc = 0
        for B in self.B_list:
            is_equal = is_equal and all((B == other.B_list[cc]).__dict__['data'])
            cc += 1
        return is_equal

    def _set_knots_spatial(self, knots, knots_num=None, perc_out_range=None,
                           is_cyclic=[False], percentiles=(2, 98), repeat_extreme_knots=False):
        """
            Set new knots
        """
        if knots is None:
            self.knots = self.computeKnots(knots_num, perc_out_range, percentiles=percentiles)
        else:
            if len(knots) != self.dim:
                raise ValueError('need a knot for every dimention of the covariate smooth')
            self.knots = np.zeros(self.dim, dtype=object)
            for i in range(self.dim):
                if (not is_cyclic[i]) and repeat_extreme_knots:
                    if any(knots[i][:self._ord] != knots[i][0]):
                        knots[i] = np.hstack(([knots[i][0]] * (self._ord - 1), knots[i]))
                    if any(knots[i][-self._ord:] != knots[i][-1]):
                        knots[i] = np.hstack((knots[i], [knots[i][-1]] * (self._ord - 1)))
                self.knots[i] = np.array(knots[i])
        self.time_pt_for_kernel = None

    def _set_knots_temporal(self, knots_num, kernel_length, kernel_direction, knots=None):
        if kernel_length % 2 == 0:
            kernel_length += 1
        if not knots is None:
            if len(knots) != 1:
                raise ValueError(
                    'temporal kernel have 1D response funciton, a list containing one input vector is required')
            knots = knots[0]

        repeats = self._ord - 1
        kernel_length = kernel_length #+ self._ord + 1
        if kernel_direction == 0:
            if knots is None:
                times = np.linspace(1 - kernel_length, kernel_length - 1, kernel_length)
                knots = np.hstack(([-(kernel_length - 1)] * repeats,
                                   np.linspace(-(kernel_length - 1), (kernel_length - 1), knots_num),
                                   [(kernel_length - 1)] * repeats))
            else:
                times = np.linspace(knots[0], knots[-1], kernel_length)

        elif kernel_direction == 1:
            if knots is None:
                int_knots = np.linspace(0.000001, knots_num, knots_num)
                knots = np.hstack(([int_knots[0]] * repeats, int_knots, [(int_knots[-1])] * repeats))

            times_pos = np.linspace(0, knots[-1], (kernel_length + 1) // 2)
            times_neg = np.linspace(-knots[-1], -times_pos[1], (kernel_length - 1) // 2)
            times = np.hstack((times_neg, times_pos))

        elif kernel_direction == -1:
            if knots is None:
                int_knots = np.linspace(0.000001, knots_num, knots_num)
                knots = np.hstack(([int_knots[0]] * repeats, int_knots, [(int_knots[-1])] * repeats))
                knots = -knots[::-1]

            # else:
            #     int_knots = np.copy(knots)

            times_neg = np.linspace(knots[0], 0, (kernel_length + 1) // 2)
            times_pos = np.linspace(-times_neg[-2], -knots[0], (kernel_length - 1) // 2)
            times = np.hstack((times_neg, times_pos))

        self.knots = np.array([knots])
        self.time_pt_for_kernel = times

    def set_cyclic(self, is_cyclic):
        """
            Set whose coordinate is cyclic
        """
        if is_cyclic is None:
            self.is_cyclic = np.zeros(self.dim, dtype=bool)
        elif self.is_temporal_kernel:
            self.is_cyclic = np.array([False])
        else:
            is_cyclic = np.array(is_cyclic)
            if not is_cyclic.dtype.type is np.bool_:
                raise ValueError('is_cyclic must be numpy array of bool')
            if is_cyclic.shape[0] != self.dim:
                raise ValueError('is_cyclic must have a value for every covariate')
            self.is_cyclic = is_cyclic

    def set_lam(self, lam):
        """
            Set smoothing penalty per coordinate
        """
        if lam is None:
            self.lam = 0.05 * np.ones(len(self.S_list))
        elif np.isscalar(lam):
            self.lam = lam * np.ones(len(self.S_list))
        else:
            if len(lam) != len(self.S_list):
                print('lam len:', len(lam), 'pen len', len(self.S_list))
                raise ValueError('Smoothing penalty should correspond to the penalty matrix that are linearly summed')
            self.lam = np.array(lam)

    def computeKnots(self, knots_num, perc_out_range, percentiles=(2, 98)):
        """
            Compute equispaced knots based on input data values (cover all the data range)
        """
        knots = np.zeros(self.dim, dtype=object)
        i = 0
        #print(percentiles)
        for xx in self._x:
            if self.is_cyclic[i]:
                perc = [0,100]
            else:
                perc = percentiles
            # select range
            # centered knots
            min_x = np.nanpercentile(xx, perc[0])
            max_x = np.nanpercentile(xx, perc[1])

            # any out of range?
            pp = (max_x.max() - min_x.min()) * perc_out_range
            knots[i] = np.linspace(min_x - pp, max_x + pp, knots_num)
            if not self.is_cyclic[i]:
                kn0 = knots[i][0]
                knend = knots[i][-1]
                knots[i] = np.hstack(([kn0] * (self._ord - 1), knots[i], [knend] * (self._ord - 1)))
            i += 1
        return knots

    def _eval_basis_and_penalty_spatial(self, prercomp_SandB=None):
        """
                    Evaluate the basis in the datum and compute the penalty and X col means
        """
        self.xmin = np.zeros(len(self.knots))
        self.xmax = np.zeros(len(self.knots))

        for cc in range(self.knots.shape[0]):
            self.xmin[cc] = self.knots[cc][0]
            self.xmax[cc] = self.knots[cc][-1]
        X, B, S, basis_dim = basisAndPenalty(self._x, self.knots, is_cyclic=self.is_cyclic, ord=self._ord,
                                             penalty_type=self.penalty_type, xmin=self.xmin, xmax=self.xmax,
                                             der=self.der,
                                             measure=self.measure, ord_AD=self.ord_AD,
                                             ad_knots=self.ad_knots, domain_fun=self.domain_fun,
                                             prercomp_SandB=prercomp_SandB)
        colMean_X = np.mean(np.array(X[:, :-1].toarray()[~self.nan_filter, :], dtype=np.double), axis=0)
        return X, B, S, colMean_X, basis_dim, None

    def _eval_basis_and_penalty_temporal(self):

        self.xmin = np.zeros(len(self.knots))
        self.xmax = np.zeros(len(self.knots))

        for cc in range(self.knots.shape[0]):
            self.xmin[cc] = self.knots[cc][0]
            self.xmax[cc] = self.knots[cc][-1]

        self.basis_kernel, B, S, basis_dim = basisAndPenalty(np.array([self.time_pt_for_kernel]), self.knots,
                                                             is_cyclic=self.is_cyclic, ord=self._ord,
                                                             der=self.der, xmin=self.xmin, xmax=self.xmax,
                                                             penalty_type=self.penalty_type,
                                                             measure=self.measure, ord_AD=self.ord_AD,
                                                             ad_knots=self.ad_knots, domain_fun=None)
        X = self._eval_basis_temporal(self._x, self.trial_idx, self.pre_trial_dur, self.post_trial_dur, self.time_bin)
        colMean_X = np.mean(np.array(X[:, :-1].toarray()[~self.nan_filter, :], dtype=np.double), axis=0)
        return X, B, S, colMean_X, basis_dim, self.basis_kernel

    def set_new_covariate(self, x_cov, knots=None, knots_num=None, perc_out_range=None, kernel_length=None,
                          kernel_direction=None):
        """
            Set new kovariates and refresh the results
        """
        if self.is_temporal_kernel:
            self.set_knots(knots_num, kernel_length, kernel_direction)
        else:
            self.set_knots(knots, knots_num=knots_num, perc_out_range=perc_out_range)
        self._x = np.array(x_cov)
        self.nan_filter = np.array(np.sum(np.isnan(self._x), axis=0), dtype=bool)
        self.X, self.B_list, self.S_list, self.colMean_X, self.basis_dim, self.basis_kernel = self.eval_basis_and_penalty()

    def compute_Bx(self):
        if self.dim == 1 and (self.penalty_type in ['EqSpaced', 'diff'] or self.der <= 1):
            Bx = np.sqrt(self.lam[0]) * self.B_list[0]

        else:
            Sx = 0
            cc = 0
            for S in self.S_list:
                Sx = Sx + S * self.lam[cc]
                cc += 1

            if np.sum(self.lam) == 0:
                Bx = np.zeros(Sx.shape)
            else:
                try:
                    Bx = np.linalg.cholesky(Sx).T
                except np.linalg.LinAlgError:
                    try:
                        eig, U = np.linalg.eigh(Sx)
                    except Exception as e:
                        raise (e)

                    sort_col = np.argsort(eig)[::-1]
                    eig = eig[sort_col]
                    U = U[:, sort_col]
                    # matrix is sym should be positive
                    eig = np.abs(eig)
                    i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
                    eig = np.delete(eig, i_rem, 0)
                    Bx = np.zeros(U.shape)
                    mask = np.arange(U.shape[1])
                    mask = mask[np.delete(mask, i_rem, 0)]
                    Bx[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
                    Bx = Bx.T

        return Bx

    def additive_model_preprocessing(self, penal_only=False, sparsebl=True):
        Bx = self.compute_Bx()
        if self.X.shape[1] != 1:
            if penal_only:
                return Bx[:, :-1]
            # preprocess X in order to remove the undetermined intercept fit in an additive model
            X = self.X[:, :-1]
    
            Bx = Bx[:, :-1]
            if type(X) is sparse.csr_matrix:
                X = X.toarray() - self.colMean_X
            else:
                X = X - self.colMean_X
        else:
            if penal_only:
                return Bx
            # preprocess X in order to remove the undetermined intercept fit in an additive model
            X = self.X
    
            Bx = Bx
            if type(X) is sparse.csr_matrix:
                X = X.toarray()
            else:
                X = X
        # nan time points set to zero so that <X, \beta> do not contribute
        X[self.nan_filter, :] = 0
        if sparsebl:
            X = sparse.csr_matrix(X, dtype=np.float64)
        return X, Bx

    def mean_center(self, X):

        return np.array(X[:, :-1] - np.mean(X[~self.nan_filter, :-1], axis=0))

    def _eval_basis_spatial(self, X):
        """
            Evaluate the basis function
        """
        fX, _, _, _ = basisAndPenalty(X, self.knots, is_cyclic=self.is_cyclic, ord=self._ord, der=self.der,
                                      penalty_type=self.penalty_type, xmin=self.xmin, xmax=self.xmax,
                                      measure=self.measure, compute_pen=False)
        return fX

    def _eval_basis_temporal(self, X, trial_idx, pre_trial_dur, post_trial_dur, time_bin):
        # x list of length 1 (1-dim temporal filter)
        if len(X) > 1:
            raise ValueError('temporal kernel signal to be filtered should be 1-dim')
        x = X[0]
        Xc = np.zeros((x.shape[0], self.basis_kernel.shape[1]))
        skip_start = int(np.ceil(pre_trial_dur / time_bin))
        skip_end = int(np.ceil(post_trial_dur / time_bin))
        if (trial_idx is None) or len(trial_idx) == 0:
            raise ValueError('must indicate trial indices for temp kernel')
        for k in range(self.basis_kernel.shape[1]):
            kern_vec = self.basis_kernel[:, k].toarray().flatten()
            for tr in np.unique(trial_idx):
                sel = np.where(trial_idx == tr)[0]
                sel = sel[skip_start:sel.shape[0] - skip_end]
                sel_conv = signal.fftconvolve(x[sel], kern_vec, mode='same')
                Xc[sel, k] = sel_conv

        return sparse.csr_matrix(Xc)


class smooths_handler(object):
    def __init__(self):
        self.smooths_dict = {}
        self.smooths_var = []

    def add_smooth(self, name, x_cov, ord=4, lam=None, knots=None, knots_num=15, perc_out_range=0.1, is_cyclic=None,
                   is_temporal_kernel=False, kernel_direction=0, kernel_length=21, penalty_type='EqSpaced', der=None,
                   knots_percentiles=(2, 98), trial_idx=None, time_bin=0.006, pre_trial_dur=None, post_trial_dur=None,
                   penalty_measure=None, event_input=True, ord_AD=None, ad_knots=None,
                   domain_fun=lambda x: np.zeros(x.shape, dtype=bool),
                   prercomp_SandB=None, repeat_extreme_knots=True):
        """
        :param name: string, name of the variable
        :param x_cov: list containing the input variable (the list will contain 1 vector per dimension of the variable)
        :param ord: int, the order of the base spline, the number of coefficient in the polinomial (ord =4 is cubic spline)
        :param lam: float, or list of float (smoothing  coefficients, leave None)
        :param knots: list or None. If list, each element of the list is a vector of knots locations for a specific dimension of the variable
        :param knots_num: int, used if no knots are specified, number of knots to be used
        :param perc_out_range: (set this to 0), obsolete... float between 0 and 1, percentage of knots out of the variable range (set to 0. is
        :param is_cyclic: list of bool, True if a variable dimension is cyclic
        :param is_temporal_kernel: bool, true if it is a temporal event, false if it is a spatial variable
        :param kernel_direction: 0,1,-1 directionality of the kernel
        :param kernel_length: int, length in time points of the kernel
        :param penalty_type: 'der' or 'EqSpaced', determine how to penalize for wiggliness ('der' is energy based, 'EqSpaced' difference based)
        :param der: 1 or 2. order of the derivative that should be penalized (set 2)
        :param knots_percentiles: tuple of two floats between 0 and 100 (for spatial variables) , if no knots are passed,
                                "knots_num" knots are equispaced between the specified percentiles of the input variable
        :param trial_idx: vector of int, only for temporal, index of the trial. must be of the same length of the input variable vector
        :param time_bin: float, time binning in sec
        :param pre_trial_dur: float, pre-trial duration in sec
        :param post_trial_dur: float, post-trial duration in sec
        :return:
        """
        if name in self.smooths_var:
            print('Name "%s" already used. Overwriting' % name)
            self.smooths_var.remove(name)
            self.smooths_dict.pop(name)
        self.smooths_var += [name]
        self.smooths_dict[name] = covarate_smooth(x_cov, ord=ord, knots=knots, knots_num=knots_num,
                                                  perc_out_range=perc_out_range, is_cyclic=is_cyclic, lam=lam,
                                                  is_temporal_kernel=is_temporal_kernel,
                                                  kernel_direction=kernel_direction,
                                                  kernel_length=kernel_length, penalty_type=penalty_type, der=der,
                                                  knots_percentiles=knots_percentiles, trial_idx=trial_idx,
                                                  time_bin=time_bin, pre_trial_dur=pre_trial_dur,
                                                  post_trial_dur=post_trial_dur,
                                                  penalty_measure=penalty_measure,
                                                  event_input=event_input, ord_AD=ord_AD,
                                                  ad_knots=ad_knots, domain_fun=domain_fun,
                                                  prercomp_SandB=prercomp_SandB,
                                                  repeat_extreme_knots=repeat_extreme_knots)
        return True

    def __getitem__(self, name):
        return self.smooths_dict[name]

    def __eq__(self, other):
        is_eq = True
        for name in self.smooths_var:
            print('check', name)
            is_eq = is_eq and (self.smooths_dict[name] == other.smooths_dict[name])
        return is_eq

    def set_smooth_penalties(self, smooth_pen, list_cov=None):
        if list_cov is None:
            list_cov = self.smooths_var
        ## smooth_pen is a vector with all the penalties in order per variable
        tot_smooths_required = 0
        for cov_name in list_cov:
            tot_smooths_required = (tot_smooths_required + len(
                self.smooths_dict[cov_name].S_list))  # self.smooths_dict[cov_name].dim +\
            # self.smooths_dict[cov_name].extra_pen)
        if len(smooth_pen) != tot_smooths_required:
            raise ValueError(
                'smooth_pen length must match the covariates number, (if the mean funciton mu(x) : R^n --> R, smooth_pen must be of length n')

        cc = 0
        for cov_name in list_cov:
            smooth_num = len(self.smooths_dict[cov_name].S_list)
            lam = smooth_pen[cc:cc + smooth_num]
            cc = cc + smooth_num
            # set the new penalties (make sure that a new penalty matrix is created, lambda is always used
            self.smooths_dict[cov_name].set_lam(lam)
        return True

    def get_sm_ols_endog_and_exog(self, name, y):
        """
        this function returns the matrices needed for fitting a smoother (1 covariate model y=f(x)+ noise, with noise
        normally distributed)
        :param name: smooth that needs to be regressed
        :param y: endog 1D variable
        :return:
        """
        X, B_list = self.smooths_dict[name].X, self.smooths_dict[name].B_list
        # set nan to zero
        nan_filter = self.smooths_dict[name].nan_filter
        X[nan_filter, :] = 0

        Bx = self.self.smooths_dict[name].compute_Bx()
        if type(Bx) is sparse.csr.csr_matrix and type(X) is sparse.csr.csr_matrix:
            Xagu = np.vstack((X.toarray(), Bx.toarray()))

        elif type(Bx) is sparse.csr.csr_matrix:
            Xagu = np.vstack((X, Bx.toarray()))

        elif type(X) is sparse.csr.csr_matrix:
            Xagu = np.vstack((X.toarray(), Bx))

        else:
            Xagu = np.vstack((X, Bx))

        yagu = np.hstack((y, np.zeros(Bx.shape[0])))

        return Xagu, yagu

    def get_additive_model_endog_and_exog(self, name_list, y):
        Xagu, yagu, index_cov = self.get_general_additive_model_endog_and_exog(name_list, y, weights=None)
        return Xagu, yagu, index_cov

    def get_general_additive_model_endog_and_exog(self, name_list, y, weights=None):
        """
        Cycle over variables and prepare the agumented matrix to be used in the ols
        :param name_list:
        :param y:
        :param sp_list:
        :return:
        """
        first = True
        index_cov = {}
        count = 1
        cov_num = 0
        if not weights is None:
            N = weights.shape[0]
            w_mat = sparse.dia_matrix((np.sqrt(weights), [0]), (N, N), dtype=np.float64)

        for name in name_list:
            sm_cov = self.smooths_dict[name]

            if len(name_list) > 0:
                X, M = sm_cov.additive_model_preprocessing()
            else:
                X = sm_cov.X
                X[sm_cov.nan_filter, :] = 0
                M = sm_cov.compute_Bx()

            if not weights is None:
                X = w_mat * X

            # save the indices that will be related to a specific covariate in the full regression
            index_cov[name] = np.arange(count, count + X.shape[1])
            # update the starting index
            count += X.shape[1]

            if type(M) is np.ndarray:
                M = sparse.csr_matrix(M)
            hstack_M = sparse.hstack
            vstack_M = sparse.vstack
            zeros = lambda shape: sparse.csr_matrix(np.zeros(shape))
            if type(X) is sparse.csr.csr_matrix:
                hstack_X = sparse.hstack
            else:
                hstack_X = np.hstack

            if first:
                first = False
                fullX = hstack_X((np.ones((X.shape[0], 1)), X.copy()))
                fullM = hstack_M((np.zeros((M.shape[0], 1)), M.copy()))
            else:
                fullX = hstack_X((fullX, X))
                # add zeros at the RHS
                zero_pad = fullM.shape[1]

                fullM = hstack_M((fullM, zeros((fullM.shape[0], M.shape[1]))))
                # add zeros at the LHS and the new marix
                M_zeropad = hstack_M((zeros((M.shape[0], zero_pad)), M))
                # attach to the matrix
                fullM = vstack_M((fullM, M_zeropad))

            cov_num += 1

        if type(fullM) is sparse.csr.csr_matrix or type(fullM) is sparse.coo.coo_matrix:
            fullM = fullM.toarray()
        if type(fullX) is sparse.csr.csr_matrix or type(fullX) is sparse.coo.coo_matrix:
            fullX = fullX.toarray()

        Xagu = np.vstack((fullX, fullM))

        if not weights is None:
            yagu = weights * y
        else:
            yagu = y

        yagu = np.hstack((yagu, np.zeros(fullM.shape[0])))
        return Xagu, yagu, index_cov

    def get_exog_mat_fast(self, name_list):
        # calculate col number
        # t0 = perf_counter()
        num_col = 1
        for name in name_list:
            num_col += self.smooths_dict[name].X.shape[1] - 1
        fullX = np.ones((self.smooths_dict[name].X.shape[0], num_col))
        index_cov = {}
        count = 1
        for name in name_list:

            sm_cov = self.smooths_dict[name]

            if len(name_list) > 0:
                X, _ = sm_cov.additive_model_preprocessing(sparsebl=False)
            else:
                X = sm_cov.X
                X[sm_cov.nan_filter, :] = 0

            # save the indices that will be related to a specific covariate in the full regression
            index_cov[name] = np.arange(count, count + X.shape[1])
            # update the starting index
            count += X.shape[1]

            fullX[:, index_cov[name]] = X
        # t1 = perf_counter()
        # print('hstack:', t1 - t0, 'sec')

        return fullX, index_cov

    def get_exog_mat(self, name_list):
        first = True
        index_cov = {}
        count = 1
        #t0 = perf_counter()
        # # calculate col number
        # num_col = 1
        # for name in name_list:
        #     num_col += self.smooths_dict[name].X.shape[1] - 1
        # X = np.zeros((self.smooths_dict[name].X.shape[1],num_col))

        for name in name_list:

            sm_cov = self.smooths_dict[name]

            if len(name_list) > 0:
                X, _ = sm_cov.additive_model_preprocessing(sparsebl=False)
            else:
                X = sm_cov.X
                X[sm_cov.nan_filter, :] = 0

            # save the indices that will be related to a specific covariate in the full regression
            index_cov[name] = np.arange(count, count + X.shape[1])
            # update the starting index
            count += X.shape[1]

            if type(X) is sparse.csr.csr_matrix:
                hstack_X = sparse.hstack
            else:
                #print('full matrix stack')
                hstack_X = np.hstack

            if first:
                first = False
                fullX = hstack_X((np.ones((X.shape[0], 1)), X.copy()))
            else:
                fullX = hstack_X((fullX, X))
        #t1 = perf_counter()
        #print('hstack:', t1 - t0, 'sec')
        #t0 = perf_counter()
        if type(fullX) is sparse.csr.csr_matrix or type(fullX) is sparse.coo.coo_matrix:
            fullX = fullX.toarray()
        t1 = perf_counter()
        #print('tranform to full matrix: ', t1 - t0, 'sec')

        return fullX, index_cov

    def get_penalty_agumented(self, name_list):
        """
        Cycle over variables and prepare the agumented matrix to be used in the ols
        :param name_list:
        :param y:
        :param sp_list:
        :return:
        """
        first = True
        cov_num = 0

        for name in name_list:
            sm_cov = self.smooths_dict[name]

            if len(name_list) > 0:
                M = sm_cov.additive_model_preprocessing(penal_only=True)
            else:
                M = sm_cov.compute_Bx()

            if type(M) is np.ndarray:
                M = sparse.csr_matrix(M)
            hstack_M = sparse.hstack
            vstack_M = sparse.vstack
            zeros = lambda shape: sparse.csr_matrix(np.zeros(shape))

            if first:
                first = False
                fullM = hstack_M((np.zeros((M.shape[0], 1)), M.copy()))
            else:
                # add zeros at the RHS
                zero_pad = fullM.shape[1]
                fullM = hstack_M((fullM, zeros((fullM.shape[0], M.shape[1]))))
                # add zeros at the LHS and the new marix
                M_zeropad = hstack_M((zeros((M.shape[0], zero_pad)), M))
                # attach to the matrix
                fullM = vstack_M((fullM, M_zeropad))

            cov_num += 1

        if type(fullM) is sparse.csr.csr_matrix or type(fullM) is sparse.coo.coo_matrix:
            fullM = fullM.toarray()

        return fullM


def matrix_transform(*M):
    mat_list = []
    for R in M:
        mat_list += [np.matrix(R)]
    return mat_list


def compute_Sjs(sm_handler, var_list):
    S_all = []
    tot_dim = 1
    ii = 0
    if len(var_list) > 0:
        ii = 1
    for var in var_list:
        tot_dim += sm_handler[var].X.shape[1] - ii * (sm_handler[var].X.shape[1] != 1)

    cc = 1
    for var in var_list:
        dim = len(sm_handler[var].S_list)

        for k in range(dim):
            S = np.zeros((tot_dim, tot_dim))
            Sk = sm_handler[var].S_list[k]
            shapeS = Sk.shape[0]
            Sk = Sk[:shapeS - ii, :shapeS - ii]
            S[cc: cc + Sk.shape[0], cc:cc + Sk.shape[0]] = Sk
            S_all += [S]
        cc += Sk.shape[0]

    return S_all


def checkGrad(grad, grad_app, tol=10 ** -3, print_res=False):
    DEN = (np.linalg.norm(grad) + np.linalg.norm(grad_app))
    if DEN == 0:
        check = 0
    else:
        check = np.linalg.norm(grad_app - grad) / (np.linalg.norm(grad) + np.linalg.norm(grad_app))
    if print_res == True:
        print('check:', check)
    return check > tol


def approx_grad(x0, dim, func, epsi):
    grad = np.zeros(shape=dim)
    for j in range(grad.shape[0]):
        if np.isscalar(x0):
            ej = epsi
        else:
            ej = np.zeros(x0.shape[0])
            ej[j] = epsi
        grad[j] = (func(x0 + ej) - func(x0 - ej)) / (2 * epsi)
    return grad


