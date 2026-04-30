import numpy as np
import scipy.linalg as sp_linalg
import scipy.stats as sts
from .utils.linalg_utils import inner1d_sum


def symmetrize_tensor(S_tens):
    new_tens = np.zeros(S_tens.shape)
    cc = 0
    for S in S_tens:
        Ssym = np.triu(S) + np.triu(S, 1).T
        new_tens[cc] = Ssym
        cc += 1

    return new_tens


def transform_Slam(S_tensor, rho):
    """
    Compute log|sum_i lams[i] * S_tensor[i]|_+ stably (Wood 2011, Appendix B).

    Parameters
    ----------
    lams     : (M,) array of positive smoothing parameters
    S_tensor : (M, q, q) array of positive semi-definite matrices

    Returns
    -------
    log_det : float
    S_full  : (q, q) ndarray  — transformed penalty sum, use for Cholesky / derivatives
    S_i_out : (M, q, q) ndarray — individually transformed S_i (same basis as S_full)
    Q_s     : (q, q) ndarray  — accumulated rotation; S_full = Q_s.T @ sum(lams*S_i) @ Q_s
    """
    lams = np.exp(rho)
    S_tensor = np.stack(S_tensor)
    M, q = S_tensor.shape[0], S_tensor.shape[1]

    eps = np.finfo(np.float64).eps
    eps_split = eps ** (1 / 3.0)
    eps_rank = eps ** 0.8

    # S_bar[j]  : current working matrix for gamma[j]; shape shrinks each iteration
    # gamma[j]  : original index into lams for position j in S_bar
    S_bar = list(S_tensor.copy())
    S_i_out = list(S_tensor.copy())  # stays (q, q) throughout

    Q_s = np.eye(q)
    gamma = list(range(M))
    K = 0

    while True:
        Q = q - K
        n = len(gamma)

        # All terms already assigned to alpha in a prior iteration; nothing left.
        if n == 0:
            break

        # Step 1
        frobs = np.array([np.sqrt((S_bar[j] ** 2).sum()) for j in range(n)])
        lams_g = np.array([lams[gamma[j]] for j in range(n)])
        omegas = frobs * lams_g
        max_omega = omegas.max()

        # Step 2
        alpha_pos = np.where(omegas >= max_omega * eps_split)[0]
        gp_pos = np.where(omegas < max_omega * eps_split)[0]

        # Step 3: formal rank of the Frobenius-normalised dominant sum
        S_alpha = np.stack([S_bar[j] for j in alpha_pos])
        frobs_a = frobs[alpha_pos]
        ev = np.linalg.eigvalsh(np.sum(S_alpha / frobs_a[:, None, None], axis=0))
        r = int(np.sum(ev > ev[-1] * eps_rank))

        # Step 4: termination
        if r == Q:
            break

        # Step 5: use S_bar (current working matrices), not original S_tensor
        lams_a = lams_g[alpha_pos]
        _, U_eig = np.linalg.eigh(np.sum(lams_a[:, None, None] * S_alpha, axis=0))
        U_eig = U_eig[:, ::-1]  # descending eigenvalue order
        Ur = U_eig[:, :r]
        Un = U_eig[:, r:]

        T_gp = np.eye(q);
        T_gp[K:, K:] = U_eig  # [[I_K, 0], [0, U]]
        T_al = np.zeros((q, q))
        if K > 0:
            T_al[:K, :K] = np.eye(K)
        T_al[K:, K:K + r] = Ur  # [[I_K, 0, 0], [0, Ur, 0]]

        # Step 6
        Q_s = Q_s @ T_gp

        # Step 7
        for j in alpha_pos:
            idx = gamma[j]
            S_i_out[idx] = T_al.T @ S_i_out[idx] @ T_al
        for j in gp_pos:
            idx = gamma[j]
            S_i_out[idx] = T_gp.T @ S_i_out[idx] @ T_gp

        # Step 8
        new_S_bar = [Un.T @ S_bar[j] @ Un for j in gp_pos]
        new_gamma = [gamma[j] for j in gp_pos]

        # Step 9
        K += r
        S_bar = new_S_bar
        gamma = new_gamma

    # Assemble from individually-transformed matrices so each matrix element
    # receives contributions from only one lambda scale (no cross-scale cancellation).
    S_full = sum(lams[i] * S_i_out[i] for i in range(M))
    S_full = 0.5 * (S_full + S_full.T)  # numerical symmetry after accumulated rotations
    S_i_out = np.stack(S_i_out)

    return S_full, S_i_out


def _slam_chol_and_inv(S_transf, lam):
    """Shared helper: build S_lam, its Cholesky factor, and S_lam^{-1}.

    Returns (log_det, Sinv) where log_det = log|S_lam| and Sinv = S_lam^{-1}
    (pseudoinverse when S_lam is singular, via eigendecomposition fallback).
    L is lower-triangular from Cholesky; its inverse is computed via a
    triangular solve rather than pinv to avoid an unnecessary SVD.
    """
    Slam = np.einsum("ijk,i", S_transf, lam)
    try:
        Pinv = np.diag(1.0 / np.sqrt(np.abs(np.diag(Slam))))
        P    = np.diag(np.sqrt(np.abs(np.diag(Slam))))
        L    = np.linalg.cholesky(np.einsum("ij,jh,hk->ik", Pinv, Slam, Pinv))
        log_det = 2.0 * np.sum(np.log(np.diag(L))) + 2.0 * np.sum(np.log(np.diag(P)))
        # L is lower-triangular and full-rank: triangular solve, not pinv.
        # Sinv = Pinv @ Linv.T @ Linv @ Pinv; since Pinv is diagonal this
        # reduces to two element-wise scalings (~5 µs vs ~26 ms for einsum).
        Linv = sp_linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        p_inv = np.diag(Pinv)
        Sinv = (p_inv[:, None] * Linv.T) @ (Linv * p_inv[None, :])
    except np.linalg.LinAlgError:
        Slam = np.triu(Slam) + np.triu(Slam, 1).T
        d_tild, U_tild = np.linalg.eigh(Slam)
        idx     = d_tild > np.finfo(float).eps
        log_det = np.sum(np.log(d_tild[idx]))
        Utmp    = U_tild[:, idx] * (1.0 / np.sqrt(d_tild[idx]))
        Sinv    = np.dot(Utmp, Utmp.T)
    return log_det, Sinv


def logDet_Slam(rho, S_transf, compute_grad=False, S_all=None):
    lam = np.exp(rho)
    if compute_grad:
        _, S_transf = transform_Slam(S_all, rho)
    Slam = np.einsum("ijk,i", S_transf, lam)
    try:
        Pinv = np.diag(1.0 / np.sqrt(np.abs(np.diag(Slam))))
        P    = np.diag(np.sqrt(np.abs(np.diag(Slam))))
        L    = np.linalg.cholesky(np.einsum("ij,jh,hk->ik", Pinv, Slam, Pinv))
        log_det = 2.0 * np.sum(np.log(np.diag(L))) + 2.0 * np.sum(np.log(np.diag(P)))
    except np.linalg.LinAlgError:
        Slam = np.triu(Slam) + np.triu(Slam, 1).T
        d_tild, _ = np.linalg.eigh(Slam)
        log_det = np.sum(np.log(d_tild[d_tild > np.finfo(float).eps]))
    return log_det


def grad_logDet_Slam(rho, S_transf, compute_grad=False, S_all=None):
    lam = np.exp(rho)
    if compute_grad:
        _, S_transf = transform_Slam(S_all, rho)
    _, Sinv = _slam_chol_and_inv(S_transf, lam)
    grad_det = np.zeros((rho.shape[0],))
    for j in range(rho.shape[0]):
        grad_det[j] = lam[j] * inner1d_sum(Sinv, S_transf[j].T)
    return grad_det


def hes_logDet_Slam(rho, S_transf):
    lam = np.exp(rho)

    _, Sinv = _slam_chol_and_inv(S_transf, lam)

    hes_det = np.zeros((rho.shape[0], rho.shape[0]))
    tmp_dict = {}
    for i in range(rho.shape[0]):
        # Sinv_Si = np.einsum('ij,jk->ik', Sinv, S_transf[i])
        for j in range(rho.shape[0]):
            if i == 0:
                Sinv_Sj = np.einsum(
                    "ij,jk->ik", Sinv, S_transf[j]
                )  # use symmetry to half the time
                tmp_dict[j] = Sinv_Sj
            else:
                Sinv_Sj = tmp_dict[j]
            Sinv_Si = tmp_dict[i]
            hes_det[i, j] = -lam[j] * lam[i] * inner1d_sum(Sinv_Si, Sinv_Sj.T)
            if i == j:
                hes_det[i, j] = hes_det[i, j] + lam[i] * inner1d_sum(
                    Sinv, S_transf[i].T
                )
    return hes_det


if __name__ == "__main__":
    from gam_data_handlers import *

    np.random.seed(4)

    tp = 1 * 10**3
    x1, x2, x3 = (
        np.random.uniform(0.05, 1, size=tp),
        np.random.uniform(0, 1, size=tp),
        np.random.uniform(-2, 2, size=tp),
    )
    xs = [x1, x2, x3]

    # define smooth handler
    sm_handler = smooths_handler()
    sm_handler.add_smooth(
        "1d_var",
        [x1],
        ord=4,
        knots=None,
        knots_num=10,
        perc_out_range=0.0,
        is_cyclic=[False],
        lam=None,
        penalty_type="der",
        der=2,
    )
    sm_handler.add_smooth(
        "1d_var2",
        [x2],
        ord=4,
        knots=None,
        knots_num=10,
        perc_out_range=0.0,
        is_cyclic=[False],
        lam=None,
        penalty_type="der",
        der=2,
    )
    sm_handler.add_smooth(
        "1d_var3",
        [x3],
        ord=4,
        knots=None,
        knots_num=15,
        perc_out_range=0.0,
        is_cyclic=[False],
        lam=None,
        penalty_type="der",
        der=2,
    )

    sm_handler.add_smooth(
        "1d_var4",
        [x3],
        ord=4,
        knots=None,
        knots_num=15,
        perc_out_range=0.0,
        is_cyclic=[False],
        lam=None,
        penalty_type="der",
        der=2,
    )
    var_list = ["1d_var", "1d_var2", "1d_var3", "1d_var4"]
    # Define a gamma variable

    rho = np.array([14, 14.1, -13, 1] * 2)
    S_all = compute_Sjs(sm_handler, var_list)
    # S_all[1] = S_all[1]#*10**8
    # S_all[0] = S_all[0]# * 10 ** 8
    # S_all[2] = S_all[2]# *10**-2

    Slam = create_Slam(rho, sm_handler, var_list)
    Slam_trans, S_transf = transform_Slam(S_all, rho)

    func = lambda rho: logDet_Slam(rho, S_transf, compute_grad=True, S_all=S_all)
    grad_log_det = grad_logDet_Slam(rho, S_transf)
    app_grad = approx_grad(rho, grad_log_det.shape, func, 10**-4)

    func2 = lambda rho: grad_logDet_Slam(rho, S_transf, compute_grad=True, S_all=S_all)
    hes_log_det = hes_logDet_Slam(rho, S_transf)
    app_grad2 = approx_grad(rho, hes_log_det.shape, func2, 10**-4)
