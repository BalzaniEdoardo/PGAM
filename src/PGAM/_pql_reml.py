"""Linearized REML score for PQL smoothing-parameter selection.

At each outer PIRLS iteration the working model is

    y' = sqrt(W) z,   X' = sqrt(W) X

treated as an ordinary Gaussian regression.  The restricted marginal
log-likelihood is

    l_r(rho) = -alpha/2 - log|X'TX' + S_lam|/2 + log|S_lam|_+/2

where alpha = ||y' - A y'||^2 is the same penalised RSS used in GCV and
A = X'(X'TX' + S_lam)^{-1} X'T.  Minimising -l_r drives REML smoothing
selection on the linearised model (Wood 2017 §6.5, strategy 2).

Gradient of -l_r w.r.t. rho_j:

    d(-l_r)/d rho_j = alpha_grad_j / 2
                    + lam_j * tr((X'TX'+S_lam)^{-1} S_j)   [= lam_j tr(Mj)]
                    - d log|S_lam|_+ / d rho_j / 2

The log-det terms and their derivatives reuse deriv_det_Slam.py machinery;
only tr(Mj) = tr((X'TX'+S_lam)^{-1} S_j) is new (one np.trace call per j).
"""

import numpy as np
import scipy.linalg as linalg

from .gam_data_handlers import compute_Sjs, approx_grad
from .deriv_det_Slam import transform_Slam, logDet_Slam, grad_logDet_Slam
from .utils.linalg_utils import inner1d_sum


def reml_objective(
    rho,
    X,
    Q,
    R,
    endog,
    sm_handler,
    var_list,
    return_type="eval_grad",
    S_all=None,
    S_transf=None,
):
    """Linearized REML score for the PQL working model (to minimise).

    Parameters
    ----------
    rho : ndarray, shape (M,)
        Log-smoothing parameters.
    X, Q, R, endog :
        Whitened quantities from the current PIRLS step: X = sqrt(W) X_orig,
        endog = sqrt(W) z.  Only the first n_obs = X.shape[0] rows of endog
        are used.
    sm_handler, var_list :
        Smooth handler and variable list.
    return_type : {"eval", "eval_grad"}
        How much to compute.
    S_all : list of ndarray or None
        Raw penalty matrices from compute_Sjs (plain ndarrays, not np.matrix).
        Computed internally if None.
    S_transf : ndarray or None
        Second output of transform_Slam(S_all, rho0), the Sj matrices
        projected onto the range of S_lam.  Pre-compute this once per PIRLS
        step (using the starting rho0) and pass it here so the expensive
        transform is not repeated on every optimizer call.  If None it is
        computed internally.

    Returns
    -------
    "eval"      -> float
    "eval_grad" -> (float, ndarray shape (M,))
    """
    sm_handler.set_smooth_penalties(np.exp(rho), var_list)
    n_obs = X.shape[0]
    B = np.array(sm_handler.get_penalty_agumented(var_list), dtype=np.float64)

    U, s, V_T = linalg.svd(np.vstack((R, B)))

    i_rem = np.where(s < 1e-8 * s.max())[0]
    s   = np.delete(s, i_rem, 0)
    U   = np.delete(U, i_rem, 1)
    V_T = np.delete(V_T, i_rem, 0)

    U1 = U[: R.shape[0], : s.shape[0]]
    y  = endog[:n_obs].reshape((n_obs, 1))

    dinv = 1.0 / s

    # REML RSS: y^T (I - A) y  =  ||y||^2 - ||y1||^2
    #
    # The hat matrix A = X(X'X + S_lam)^{-1}X' is NOT idempotent for penalised
    # regression, so ||y - Ay||^2  !=  y^T(I-A)y in general.  From the SVD:
    #   y^T A y = ||U1^T Q^T y||^2  (using A = Q U1 U1^T Q^T)
    # so the REML RSS is  ||y||^2 - ||y1||^2  where y1 = U1^T Q^T y.
    y1        = U1.T @ (Q.T @ y)              # shape (k, 1)
    RSS_reml  = y.T @ y - y1.T @ y1           # (1, 1) array

    # log|X'TX' + S_lam| = 2 * sum log(s_k)  (from SVD of [R; B])
    log_det_XtXpSl = 2.0 * np.sum(np.log(s))

    if S_all is None:
        S_all = compute_Sjs(sm_handler, var_list)
    if S_transf is None:
        _, S_transf = transform_Slam(S_all, rho)

    log_det_Sl = logDet_Slam(rho, S_transf)

    reml_val = 0.5 * RSS_reml.item() + 0.5 * log_det_XtXpSl - 0.5 * log_det_Sl

    if return_type == "eval":
        return reml_val

    # --- gradient ---
    # d[y^T(I-A)y]/d rho_j = -d[y^T A y]/d rho_j
    # Since y^T A y = y1^T y1 and dA/d rho_j = -lam_j Q U1 Dinv V Sj V^T Dinv U1^T Q^T:
    #   d[y^T A y]/d rho_j = -lam_j y1^T Mj y1
    # => d[y^T(I-A)y]/d rho_j =  lam_j y1^T Mj y1
    lams = np.exp(rho)
    p    = len(rho)

    grad_log_det_Sl = grad_logDet_Slam(rho, S_transf)
    reml_grad = np.zeros(p)

    for j in range(p):
        Mj = (dinv[:, None] * V_T) @ S_all[j] @ (dinv * V_T.T)   # (k, k)

        # d[y^T(I-A)y]/d rho_j  =  lam_j * y1^T Mj y1
        RSS_grad_j = lams[j] * (y1.T @ Mj @ y1).item()

        # d log|X'TX'+S_lam|/d rho_j  =  lam_j tr(Mj)
        # Derivation: d/d rho_j [2 sum log s_k] = 2 sum (1/s_k) * lam_j v_k^T Sj v_k / (2 s_k)
        #           = lam_j sum v_k^T Sj v_k / s_k^2  =  lam_j tr((X'X+Slam)^{-1} Sj)
        logdet_XtX_grad_j = lams[j] * np.trace(Mj)

        reml_grad[j] = (
            0.5 * RSS_grad_j
            + 0.5 * logdet_XtX_grad_j
            - 0.5 * grad_log_det_Sl[j]
        )

    return reml_val, reml_grad


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def prepare_S_transf(S_all, rho):
    """Project each Sj onto the range of S_lam.

    Wraps transform_Slam and returns only S_transf (the second output),
    which is the quantity that must be pre-computed once per PIRLS step and
    then passed to reml_objective as S_transf.

    Parameters
    ----------
    S_all : list of ndarray
        Output of compute_Sjs — one (p x p) penalty matrix per smoothing
        parameter.
    rho : ndarray, shape (M,)
        Current log-smoothing parameters (used to determine the range of S_lam).

    Returns
    -------
    S_transf : ndarray, shape (M, q, q)
        Projected penalty matrices.  q = rank(S_lam).
    """
    _, S_transf = transform_Slam(S_all, rho)
    return S_transf


# ---------------------------------------------------------------------------
# Self-test / regression baseline for transform_Slam
# ---------------------------------------------------------------------------

def _naive_log_det_Sl(rho, S_all):
    """Reference log|S_lam|_+ via plain eigendecomposition (no transform)."""
    lam = np.exp(rho)
    Slam = sum(lam[j] * S_all[j] for j in range(len(rho)))
    Slam = 0.5 * (Slam + Slam.T)
    e = np.linalg.eigvalsh(Slam)
    return float(np.sum(np.log(e[e > 1e-10 * e.max()])))


def _naive_grad_log_det_Sl(rho, S_all):
    """Reference gradient of log|S_lam|_+ via plain eigendecomposition."""
    lam = np.exp(rho)
    Slam = sum(lam[j] * S_all[j] for j in range(len(rho)))
    Slam = 0.5 * (Slam + Slam.T)
    e, U = np.linalg.eigh(Slam)
    keep = e > 1e-10 * e.max()
    U_r  = U[:, keep]
    e_r  = e[keep]
    grad = np.zeros(len(rho))
    for j in range(len(rho)):
        # d log|S_lam|_+ / d rho_j = lam_j * tr(S_lam^{-} S_j)
        proj = U_r.T @ S_all[j] @ U_r   # (k x k)
        grad[j] = lam[j] * float(np.sum(np.diag(proj) / e_r))
    return grad


if __name__ == "__main__":
    import statsmodels.api as sm
    from .gam_data_handlers import smooths_handler

    np.random.seed(42)
    n = 2000
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)

    sm_h = smooths_handler()
    sm_h.add_smooth("x1", [x1], knots_num=8, penalty_type="diff")
    sm_h.add_smooth("x2", [x2], knots_num=8, penalty_type="diff")
    var_list = ["x1", "x2"]

    rho = np.array([1.5, -0.5])
    S_all = compute_Sjs(sm_h, var_list)

    # --- regression baseline: transform_Slam vs naive ---
    _, S_transf = transform_Slam(S_all, rho)
    ld_fast  = logDet_Slam(rho, S_transf)
    ld_naive = _naive_log_det_Sl(rho, S_all)
    print(f"log|S_lam|_+  fast={ld_fast:.6f}  naive={ld_naive:.6f}  "
          f"diff={abs(ld_fast - ld_naive):.2e}")

    gd_fast  = grad_logDet_Slam(rho, S_transf)
    gd_naive = _naive_grad_log_det_Sl(rho, S_all)
    for j in range(len(rho)):
        print(f"  grad[{j}]  fast={gd_fast[j]:.6f}  naive={gd_naive[j]:.6f}  "
              f"diff={abs(gd_fast[j] - gd_naive[j]):.2e}")

    # --- gradient check for reml_objective ---
    family = sm.families.Poisson(link=sm.families.links.Log())
    from ._pql_gcv import weights_and_data

    mu  = family.starting_mu(np.random.poisson(2, n).astype(float))
    wd  = weights_and_data(np.random.poisson(2, n).astype(float), family)
    z, w = wd.get_params(mu)
    sm_h.set_smooth_penalties(np.exp(rho), var_list)
    pen = sm_h.get_penalty_agumented(var_list)
    X_raw, _ = sm_h.get_exog_mat(var_list)

    Xw  = (np.sqrt(w)[:, None] * X_raw)
    yw  = np.sqrt(w) * z
    Q, R = np.linalg.qr(Xw, "reduced")
    endog_aug = np.zeros(Xw.shape[0] + pen.shape[0])
    endog_aug[:n] = yw

    _, S_transf0 = transform_Slam(S_all, rho)

    val, grad = reml_objective(
        rho, Xw, Q, R, endog_aug, sm_h, var_list,
        return_type="eval_grad", S_all=S_all, S_transf=S_transf0,
    )
    func = lambda r: reml_objective(
        r, Xw, Q, R, endog_aug, sm_h, var_list,
        return_type="eval", S_all=S_all, S_transf=S_transf0,
    )
    grad_app = approx_grad(rho, grad.shape, func, 1e-4)
    max_err = np.max(np.abs(grad - grad_app) / (np.abs(grad_app) + 1e-8))
    print(f"\nreml_objective gradient check: max rel err = {max_err:.2e}")
    if max_err > 1e-3:
        print("  WARNING: gradient may be wrong")
    else:
        print("  OK")
