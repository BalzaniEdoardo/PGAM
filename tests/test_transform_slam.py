"""
Correctness tests for transform_Slam (Wood 2011 Appendix B).

Covers:
  1. Toy extreme-lambda case: log_det vs ground truth
  2. S_full symmetry after assembly
  3. No NaN/Inf in outputs
  4. Log-det consistency vs naive eigvalsh (moderate lambda)
  5. Empty-gamma termination (n==0 guard, previously crashed with ValueError)
  6. Extreme rho stress test
  7. Pipeline smoke-test: hess_laplace_appr_REML runs without crash
"""

import numpy as np
import pytest
import statsmodels.api as sm

from PGAM.gam_data_handlers import smooths_handler, compute_Sjs
from PGAM.deriv_det_Slam import transform_Slam, logDet_Slam
from PGAM.der_wrt_smoothing import (
    d2variance_family,
    deriv3_link,
    hess_laplace_appr_REML,
    mle_gradient_bassed_optim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gam_S_all():
    rng = np.random.default_rng(7)
    n, n_smooth, n_knots = 300, 4, 8
    sm_h = smooths_handler()
    var_list = []
    for i in range(n_smooth):
        x = rng.uniform(0, 1, n)
        name = f"x{i}"
        sm_h.add_smooth(name, [x], knots_num=n_knots, penalty_type="diff")
        var_list.append(name)
    S_all = compute_Sjs(sm_h, var_list)
    rho_mod = rng.uniform(-3, 3, len(S_all))
    rho_ext = np.tile([-20.0, 15.0, -5.0, 0.0], len(S_all))[:len(S_all)]
    return S_all, rho_mod, rho_ext


# ---------------------------------------------------------------------------
# 1. Toy extreme-lambda ground-truth (from wood2011_algo_notes.md)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_toy_extreme_lambda_log_det(seed):
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((5, 5))
    _, U = np.linalg.eigh(rand.T @ rand)
    S_tensor = [U[:, :3] @ U[:, :3].T, U[:, 3:] @ U[:, 3:].T]

    log_lams = np.array([-20.0, 15.0])
    true_log_det = 3 * log_lams[0] + 2 * log_lams[1]  # = -30

    _, S_i_out = transform_Slam(S_tensor, log_lams)
    log_det = logDet_Slam(log_lams, S_i_out)

    assert abs(log_det - true_log_det) < 1e-8, (
        f"seed={seed}: log_det error = {log_det - true_log_det:.2e}"
    )


# ---------------------------------------------------------------------------
# 2. S_full symmetry
# ---------------------------------------------------------------------------

def test_S_full_symmetric_moderate(gam_S_all):
    S_all, rho_mod, _ = gam_S_all
    S_full, _ = transform_Slam(S_all, rho_mod)
    assert np.max(np.abs(S_full - S_full.T)) < 1e-12


def test_S_full_symmetric_extreme(gam_S_all):
    S_all, _, rho_ext = gam_S_all
    S_full, _ = transform_Slam(S_all, rho_ext)
    assert np.max(np.abs(S_full - S_full.T)) < 1e-12


# ---------------------------------------------------------------------------
# 3. No NaN/Inf
# ---------------------------------------------------------------------------

def test_finite_moderate(gam_S_all):
    S_all, rho_mod, _ = gam_S_all
    S_full, S_i_out = transform_Slam(S_all, rho_mod)
    assert np.isfinite(S_full).all()
    assert np.isfinite(S_i_out).all()


def test_finite_extreme(gam_S_all):
    S_all, _, rho_ext = gam_S_all
    S_full, S_i_out = transform_Slam(S_all, rho_ext)
    assert np.isfinite(S_full).all()
    assert np.isfinite(S_i_out).all()


# ---------------------------------------------------------------------------
# 4. Log-det consistency vs naive eigvalsh (moderate lambda only)
# ---------------------------------------------------------------------------

def test_logdet_vs_naive_eigvalsh(gam_S_all):
    S_all, rho_mod, _ = gam_S_all
    _, S_i_out = transform_Slam(S_all, rho_mod)
    log_det_algo = logDet_Slam(rho_mod, S_i_out)

    lams = np.exp(rho_mod)
    Slam_naive = np.einsum("ijk,i->jk", np.stack(S_all), lams)
    ev = np.linalg.eigvalsh(Slam_naive)
    log_det_naive = float(np.log(ev[ev > ev[-1] * 1e-12]).sum())

    assert abs(log_det_algo - log_det_naive) < 1e-6


# ---------------------------------------------------------------------------
# 5. Empty-gamma termination (n==0 guard) — previously crashed with ValueError
# ---------------------------------------------------------------------------

def test_empty_gamma_no_crash():
    """When gp_pos is empty in every iteration gamma empties before r==Q."""
    rng = np.random.default_rng(99)
    # Rank-1 matrices with identical column space → all dominant from iter 1
    v = rng.standard_normal(6)
    S_tensor = [np.outer(v, v), np.outer(v, v) * 0.5]
    rho = np.array([2.0, 1.0])
    S_full, S_i_out = transform_Slam(S_tensor, rho)
    assert np.isfinite(S_full).all()


# ---------------------------------------------------------------------------
# 6. Pipeline smoke-test: hess_laplace_appr_REML must not crash
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_gam():
    rng = np.random.default_rng(42)
    n, n_smooth, n_knots = 500, 3, 7
    sm_h = smooths_handler()
    var_list = []
    for i in range(n_smooth):
        name = f"x{i}"
        sm_h.add_smooth(name, [rng.uniform(0, 1, n)], knots_num=n_knots, penalty_type="diff")
        var_list.append(name)
    X, _ = sm_h.get_exog_mat(var_list)

    base = sm.families.Poisson(link=sm.families.links.Log())
    base.link = deriv3_link(base.link, run_tests=False)
    family = d2variance_family(base, run_tests=False)

    rho = rng.uniform(-2, 2, n_smooth)
    sm_h.set_smooth_penalties(np.exp(rho), var_list)
    beta_true = rng.normal(0, 0.3, X.shape[1])
    beta_true[0] = 0.5
    y = rng.poisson(family.link.inverse(X @ beta_true))

    beta_hat = mle_gradient_bassed_optim(
        rho, sm_h, var_list, y, X, family,
        phi_est=1.0, method="Newton-CG", num_random_init=1, tol=1e-6,
    )[0]
    S_all = compute_Sjs(sm_h, var_list)
    return rho, beta_hat, S_all, y, X, family, sm_h, var_list


def test_hess_laplace_no_crash(fitted_gam):
    rho, beta_hat, S_all, y, X, family, sm_h, var_list = fitted_gam
    hess, _ = hess_laplace_appr_REML(
        rho, beta_hat, S_all, y, X, family, 1.0, sm_h, var_list,
        compute_grad=False, fixRand=True, return_intermediates=True,
    )
    assert np.isfinite(hess).all()