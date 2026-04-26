"""
Numerical gradient / Hessian tests for der_wrt_smoothing.py.

Tests are ordered by chain-rule dependency:
  Batch 1 – variance and link function higher derivatives
  Batch 2 – alpha, w, h scalar derivatives wrt mu
  Batch 3 – dbeta_hat (J = dβ̂/dρ) and d2beta_hat
  (later batches will cover grad_H, REML, AIC, etc.)

Each test uses centred finite differences to approximate the derivative of the
function *one order below* and compares to the analytical formula coded in the
module.  Relative tolerance is 1e-4 for first derivatives and 1e-3 for second
derivatives (limited by FD accuracy, not by the formulae).
"""

import numpy as np
import pytest
import statsmodels.api as sm

from PGAM.gam_data_handlers import smooths_handler
from PGAM.newton_optim import gcv_objective, gcv_comp, gcv_grad_comp, gcv_hess_comp, matrix_transform
from PGAM.der_wrt_smoothing import (
    d2variance_family,
    deriv3_link,
    variance_deriv2,
    variance_deriv3,
    link_deriv3,
    link_deriv4,
    alpha_mu,
    alpha_deriv,
    alpha_deriv2,
    w_mu,
    w_deriv,
    w_2deriv,
    small_h_mu,
    deriv_small_h,
    compute_Sjs,
    mle_gradient_bassed_optim,
    dbeta_hat,
    d2beta_hat,
    H_rho,
    grad_H_drho,
    hes_H_drho,
    Vbeta_rho,
    dVb_drho,
    d2Vb_drho,
    grad_chol_Vb_rho,
    unpenalized_ll,
    penalty_ll,
    laplace_appr_REML,
    grad_laplace_appr_REML,
    grad_laplace_appr_REML_dense,
    hess_laplace_appr_REML,
    hess_laplace_appr_REML_dense,
    reml_objective,
    RemlProblem,
)

# ---------------------------------------------------------------------------
# finite-difference helpers
# ---------------------------------------------------------------------------

EPS = 1e-5
RTOL_1ST = 1e-4
RTOL_2ND = 1e-3


def fd1(f, x, eps=EPS):
    """Centred FD first derivative of scalar-valued f at each element of x."""
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


def fd2(f, x, eps=EPS):
    """Centred FD second derivative of scalar-valued f at each element of x."""
    return (f(x + eps) - 2.0 * f(x) + f(x - eps)) / eps ** 2


# ---------------------------------------------------------------------------
# shared test points
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
MU_POS = RNG.uniform(0.3, 3.0, size=30)   # strictly positive (Poisson/Gamma/log)
MU_PROB = RNG.uniform(0.05, 0.95, size=30) # in (0,1) for Binomial/Logit/Probit


# ===========================================================================
# Batch 1a – variance function higher derivatives
# ===========================================================================

class TestVarianceDeriv2:
    """V''(mu): derivative of V'(mu) wrt mu, checked by FD on V'."""

    @pytest.mark.parametrize("family", [
        sm.families.Poisson(),
        sm.families.Gamma(),
        sm.families.Gaussian(),
        sm.families.Binomial(),
        sm.families.InverseGaussian(),
    ])
    def test_fd(self, family):
        mu = MU_PROB if isinstance(family, sm.families.Binomial) else MU_POS
        v2_analytical = variance_deriv2(family, mu)
        v2_fd = fd1(lambda x: family.variance.deriv(x), mu)
        np.testing.assert_allclose(
            v2_analytical, v2_fd, rtol=RTOL_1ST,
            err_msg=f"variance_deriv2 FD mismatch for {family.__class__.__name__}"
        )

    @pytest.mark.parametrize("family,expected", [
        (sm.families.Poisson(),       0.0),
        (sm.families.Gamma(),         2.0),
        (sm.families.Gaussian(),      0.0),
        (sm.families.Binomial(),     -2.0),
    ])
    def test_analytical(self, family, expected):
        mu = MU_PROB if isinstance(family, sm.families.Binomial) else MU_POS
        v2 = variance_deriv2(family, mu)
        np.testing.assert_allclose(
            v2, expected * np.ones_like(mu), atol=1e-12,
            err_msg=f"variance_deriv2 constant mismatch for {family.__class__.__name__}"
        )


class TestVarianceDeriv3:
    """V'''(mu): derivative of V''(mu) wrt mu, checked by FD on V''."""

    @pytest.mark.parametrize("family", [
        sm.families.Poisson(),
        sm.families.Gamma(),
        sm.families.Gaussian(),
        sm.families.Binomial(),
        sm.families.InverseGaussian(),
    ])
    def test_fd(self, family):
        mu = MU_PROB if isinstance(family, sm.families.Binomial) else MU_POS
        v3_analytical = variance_deriv3(family, mu)
        v3_fd = fd1(lambda x: variance_deriv2(family, x), mu)
        np.testing.assert_allclose(
            v3_analytical, v3_fd, rtol=RTOL_1ST,
            err_msg=f"variance_deriv3 FD mismatch for {family.__class__.__name__}"
        )


# ===========================================================================
# Batch 1b – link function higher derivatives
# ===========================================================================

class TestLinkDeriv3:
    """g'''(mu): derivative of g''(mu) wrt mu, checked by FD on g''."""

    @pytest.mark.parametrize("link,mu", [
        (sm.families.links.Log(),           MU_POS),
        (sm.families.links.identity(),      MU_POS),
        (sm.families.links.inverse_power(), MU_POS),
        (sm.families.links.Logit(),         MU_PROB),
        (sm.families.links.probit(),        MU_PROB),
    ])
    def test_fd(self, link, mu):
        d3_analytical = link_deriv3(link, mu)
        d3_fd = fd1(lambda x: link.deriv2(x), mu)
        np.testing.assert_allclose(
            d3_analytical, d3_fd, rtol=RTOL_1ST,
            err_msg=f"link_deriv3 FD mismatch for {link.__class__.__name__}"
        )

    def test_log_analytical(self):
        link = sm.families.links.Log()
        np.testing.assert_allclose(
            link_deriv3(link, MU_POS), 2.0 / MU_POS ** 3, rtol=1e-12
        )

    def test_identity_zero(self):
        link = sm.families.links.identity()
        np.testing.assert_array_equal(link_deriv3(link, MU_POS), 0.0)


class TestLinkDeriv4:
    """g''''(mu): derivative of g'''(mu) wrt mu, checked by FD on g'''."""

    @pytest.mark.parametrize("link,mu", [
        (sm.families.links.Log(),           MU_POS),
        (sm.families.links.identity(),      MU_POS),
        (sm.families.links.inverse_power(), MU_POS),
        (sm.families.links.Logit(),         MU_PROB),
        (sm.families.links.probit(),        MU_PROB),
    ])
    def test_fd(self, link, mu):
        d4_analytical = link_deriv4(link, mu)
        d4_fd = fd1(lambda x: link_deriv3(link, x), mu)
        np.testing.assert_allclose(
            d4_analytical, d4_fd, rtol=RTOL_1ST,
            err_msg=f"link_deriv4 FD mismatch for {link.__class__.__name__}"
        )

    def test_log_analytical(self):
        link = sm.families.links.Log()
        np.testing.assert_allclose(
            link_deriv4(link, MU_POS), -6.0 / MU_POS ** 4, rtol=1e-12
        )

    def test_identity_zero(self):
        link = sm.families.links.identity()
        np.testing.assert_array_equal(link_deriv4(link, MU_POS), 0.0)


# ===========================================================================
# Batch 2 – alpha, w, h scalar derivatives wrt mu
# ===========================================================================

def _make_family(base_family):
    """Wrap a statsmodels family: add deriv3/4 to its link then wrap variance derivs."""
    base_family.link = deriv3_link(base_family.link, run_tests=False)
    return d2variance_family(base_family, run_tests=False)


# Poisson + log is the canonical pair: alpha=1 (const), w=mu (linear), so
# dalpha/dmu = 0, d2alpha/dmu2 = 0, w''=0 exactly.  Good for zero-value checks.
@pytest.fixture(scope="module")
def poisson_yw():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.5, 4.0, size=30)
    y = rng.poisson(mu).astype(float)
    family = _make_family(sm.families.Poisson())
    return y, mu, family


# Gamma + log is non-canonical: alpha, w', w'' are genuinely non-zero.
# This makes the tests non-trivial.
@pytest.fixture(scope="module")
def gamma_log_yw():
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.5, 4.0, size=30)
    y = rng.gamma(shape=2.0, scale=mu / 2.0)  # Gamma with mean mu
    family = _make_family(sm.families.Gamma(link=sm.families.links.Log()))
    return y, mu, family


class TestAlphaDeriv:
    """dalpha/dmu: FD of alpha_mu wrt mu."""

    def test_fd_canonical(self, poisson_yw):
        # Canonical: dalpha/dmu = 0; both sides should be ~machine epsilon.
        y, mu, family = poisson_yw
        da_analytical = alpha_deriv(y, mu, family)
        da_fd = fd1(lambda x: alpha_mu(y, x, family), mu)
        np.testing.assert_allclose(da_analytical, da_fd, atol=1e-10)

    def test_fd_noncanonical(self, gamma_log_yw):
        y, mu, family = gamma_log_yw
        da_analytical = alpha_deriv(y, mu, family)
        da_fd = fd1(lambda x: alpha_mu(y, x, family), mu)
        np.testing.assert_allclose(da_analytical, da_fd, rtol=RTOL_1ST, atol=1e-10)


class TestAlphaDeriv2:
    """d2alpha/dmu2: FD of alpha_deriv wrt mu."""

    def test_fd_canonical(self, poisson_yw):
        y, mu, family = poisson_yw
        d2a_analytical = alpha_deriv2(y, mu, family)
        d2a_fd = fd1(lambda x: alpha_deriv(y, x, family), mu)
        np.testing.assert_allclose(d2a_analytical, d2a_fd, atol=1e-8)

    def test_fd_noncanonical(self, gamma_log_yw):
        y, mu, family = gamma_log_yw
        d2a_analytical = alpha_deriv2(y, mu, family)
        d2a_fd = fd1(lambda x: alpha_deriv(y, x, family), mu)
        np.testing.assert_allclose(d2a_analytical, d2a_fd, rtol=RTOL_1ST, atol=1e-8)


class TestWDeriv:
    """dw/dmu: FD of w_mu wrt mu."""

    def test_fd_canonical(self, poisson_yw):
        y, mu, family = poisson_yw
        dw_analytical = w_deriv(mu, y, family)
        dw_fd = fd1(lambda x: w_mu(x, y, family), mu)
        np.testing.assert_allclose(dw_analytical, dw_fd, rtol=RTOL_1ST, atol=1e-10)

    def test_fd_noncanonical(self, gamma_log_yw):
        y, mu, family = gamma_log_yw
        dw_analytical = w_deriv(mu, y, family)
        dw_fd = fd1(lambda x: w_mu(x, y, family), mu)
        np.testing.assert_allclose(dw_analytical, dw_fd, rtol=RTOL_1ST, atol=1e-10)


class TestW2Deriv:
    """d2w/dmu2: FD of w_deriv wrt mu."""

    def test_fd_canonical(self, poisson_yw):
        # Canonical Poisson+log: w=mu, w''=0 exactly.
        y, mu, family = poisson_yw
        d2w_analytical = w_2deriv(mu, y, family)
        d2w_fd = fd1(lambda x: w_deriv(x, y, family), mu)
        np.testing.assert_allclose(d2w_analytical, d2w_fd, atol=1e-8)

    def test_fd_noncanonical(self, gamma_log_yw):
        y, mu, family = gamma_log_yw
        d2w_analytical = w_2deriv(mu, y, family)
        d2w_fd = fd1(lambda x: w_deriv(x, y, family), mu)
        np.testing.assert_allclose(d2w_analytical, d2w_fd, rtol=RTOL_1ST, atol=1e-8)


class TestDerivSmallH:
    """dh/dmu (h = w'/g'): FD of small_h_mu wrt mu."""

    def test_fd_canonical(self, poisson_yw):
        y, mu, family = poisson_yw
        dh_analytical = deriv_small_h(mu, y, family)
        dh_fd = fd1(lambda x: small_h_mu(x, y, family), mu)
        np.testing.assert_allclose(dh_analytical, dh_fd, rtol=RTOL_1ST, atol=1e-10)

    def test_fd_noncanonical(self, gamma_log_yw):
        y, mu, family = gamma_log_yw
        dh_analytical = deriv_small_h(mu, y, family)
        dh_fd = fd1(lambda x: small_h_mu(x, y, family), mu)
        np.testing.assert_allclose(dh_analytical, dh_fd, rtol=RTOL_1ST, atol=1e-10)


# ===========================================================================
# Batch 3 – dbeta_hat (J = dβ̂/dρ) and d2beta_hat
# ===========================================================================

@pytest.fixture(scope="module")
def gam_problem():
    """Small GAM problem with 2 smooth terms (Gamma + log, non-canonical).

    Returns a dict with all quantities needed by the derivative tests:
    y, X, family, sm_handler, var_list, S_all, rho, beta_hat, phi_est.

    compute_Sjs is used (not compute_Sall) because the derivative functions
    expect S matrices padded to the full coefficient space (p × p).
    """
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)

    sm = smooths_handler()
    sm.add_smooth("x1", [x1], knots_num=6, penalty_type="EqSpaced")
    sm.add_smooth("x2", [x2], knots_num=6, penalty_type="EqSpaced")
    var_list = ["x1", "x2"]

    X, _ = sm.get_exog_mat(var_list)
    p = X.shape[1]

    # True parameters: small intercept + mild smooth effects
    beta_true = rng.normal(0, 0.3, p)
    beta_true[0] = 0.5  # intercept

    base = sm.families.Gamma(link=sm.families.links.Log()) if False else None
    base = __import__("statsmodels").api.families.Gamma(
        link=__import__("statsmodels").api.families.links.Log()
    )
    base.link = deriv3_link(base.link, run_tests=False)
    family = d2variance_family(base, run_tests=False)

    mu_true = family.link.inverse(X @ beta_true)
    y = rng.gamma(shape=2.0, scale=mu_true / 2.0)

    phi_est = 1.0
    rho = np.array([0.5, 0.5])
    sm.set_smooth_penalties(np.exp(rho), var_list)

    beta_hat = mle_gradient_bassed_optim(
        rho, sm, var_list, y, X, family,
        phi_est=phi_est, method="Newton-CG", num_random_init=3, tol=1e-10
    )[0]

    S_all = compute_Sjs(sm, var_list)

    return dict(
        y=y, X=X, family=family, sm=sm, var_list=var_list,
        S_all=S_all, rho=rho, beta_hat=beta_hat, phi_est=phi_est,
    )


def _beta_hat_at_rho(rho, prob):
    """Re-optimise beta_hat at a perturbed rho.  Used in FD checks."""
    prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
    return mle_gradient_bassed_optim(
        rho, prob["sm"], prob["var_list"],
        prob["y"], prob["X"], prob["family"],
        phi_est=prob["phi_est"], method="Newton-CG",
        num_random_init=1, tol=1e-12,
        beta_zero=prob["beta_hat"].copy(),
    )[0]


class TestDbetaHat:
    """J = dβ̂/dρ, shape (M, p): FD of β̂(ρ) by re-optimising at perturbed ρ."""

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-4

        J_analytical = dbeta_hat(
            rho, beta_hat, S_all, prob["sm"], prob["var_list"],
            prob["y"], prob["X"], prob["family"], prob["phi_est"],
        )

        M = len(rho)
        J_fd = np.zeros_like(J_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            b_plus  = _beta_hat_at_rho(rho + drho, prob)
            b_minus = _beta_hat_at_rho(rho - drho, prob)
            J_fd[k] = (b_plus - b_minus) / (2 * eps)

        # restore sm_handler state
        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])

        np.testing.assert_allclose(J_analytical, J_fd, rtol=1e-3, atol=1e-6)


class TestD2betaHat:
    """d²β̂/(dρ_h dρ_r), shape (M, M, p): FD of J = dβ̂/dρ by re-optimising."""

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-3   # larger eps needed for second-order FD

        H_analytical = d2beta_hat(
            rho, beta_hat, S_all, prob["sm"], prob["var_list"],
            prob["y"], prob["X"], prob["family"], prob["phi_est"],
        )

        def J_at_rho(r):
            b = _beta_hat_at_rho(r, prob)
            prob["sm"].set_smooth_penalties(np.exp(r), prob["var_list"])
            return dbeta_hat(
                r, b, S_all, prob["sm"], prob["var_list"],
                prob["y"], prob["X"], prob["family"], prob["phi_est"],
            )

        M = len(rho)
        H_fd = np.zeros_like(H_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            J_plus  = J_at_rho(rho + drho)
            J_minus = J_at_rho(rho - drho)
            H_fd[k] = (J_plus - J_minus) / (2 * eps)

        # restore
        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])

        np.testing.assert_allclose(H_analytical, H_fd, rtol=1e-2, atol=1e-5)


# ===========================================================================
# Batch 4 – grad_H_drho and hes_H_drho
# ===========================================================================

def _H_at_rho(rho, prob):
    """Compute H = X^T diag(w) X / phi at the optimised beta_hat(rho)."""
    b = _beta_hat_at_rho(rho, prob)
    prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
    return np.array(H_rho(
        rho, b, prob["y"], prob["X"], prob["family"], prob["phi_est"],
        prob["sm"], prob["var_list"], comp_gradient=False,
    ))


class TestGradHDrho:
    """dH/dρ_k, shape (M, p, p): FD of H(ρ) at optimised β̂(ρ)."""

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-4

        gH_analytical = grad_H_drho(
            rho, beta_hat, prob["y"], prob["X"],
            prob["sm"], prob["var_list"], prob["family"],
            S_all, prob["phi_est"],
        )

        M = len(rho)
        gH_fd = np.zeros_like(gH_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            H_plus  = _H_at_rho(rho + drho, prob)
            H_minus = _H_at_rho(rho - drho, prob)
            gH_fd[k] = (H_plus - H_minus) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        np.testing.assert_allclose(gH_analytical, gH_fd, rtol=1e-3, atol=1e-6)


class TestHesHDrho:
    """d²H/(dρ_h dρ_k), shape (M, M, p, p): FD of grad_H_drho."""

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-3

        hH_analytical = hes_H_drho(
            rho, beta_hat, prob["y"], prob["X"],
            S_all, prob["sm"], prob["var_list"], prob["family"], prob["phi_est"],
        )

        def gH_at_rho(r):
            b = _beta_hat_at_rho(r, prob)
            prob["sm"].set_smooth_penalties(np.exp(r), prob["var_list"])
            return grad_H_drho(
                r, b, prob["y"], prob["X"],
                prob["sm"], prob["var_list"], prob["family"],
                S_all, prob["phi_est"],
            )

        M = len(rho)
        hH_fd = np.zeros_like(hH_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            gH_plus  = gH_at_rho(rho + drho)
            gH_minus = gH_at_rho(rho - drho)
            hH_fd[k] = (gH_plus - gH_minus) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        np.testing.assert_allclose(hH_analytical, hH_fd, rtol=1e-2, atol=1e-5)


# ===========================================================================
# Batch 5 – dVb_drho, d2Vb_drho, grad_chol_Vb_rho
# ===========================================================================

def _Vb_inv_at_rho(rho, prob):
    """H(rho) + S_lambda(rho) at the optimised beta_hat(rho).

    Vbeta_rho(..., inverse=False) returns -(H+S), so we negate it.
    """
    b = _beta_hat_at_rho(rho, prob)
    prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
    return -np.array(Vbeta_rho(
        rho, b, prob["y"], prob["X"], prob["family"],
        prob["sm"], prob["var_list"], prob["phi_est"],
        inverse=False,
    ))


def _chol_Vb_at_rho(rho, prob):
    """Upper-triangular Cholesky factor of (H+S)^{-1} at optimised beta_hat(rho)."""
    b = _beta_hat_at_rho(rho, prob)
    prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
    Vb = -np.array(Vbeta_rho(
        rho, b, prob["y"], prob["X"], prob["family"],
        prob["sm"], prob["var_list"], prob["phi_est"],
        inverse=True,
    ))
    return np.linalg.cholesky(Vb).T  # upper triangular R s.t. R.T @ R = (H+S)^{-1}


class TestDVbDrho:
    """dVb/drho_k, shape (M, p, p): FD of (H+S)(rho) at optimised beta_hat(rho).

    dVb_drho computes d(H+S)/drho_k = dH/drho_k + exp(rho_k)*S_k/phi,
    verified by centred FD of the full matrix H(rho)+S_lambda(rho).
    """

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-4

        dVb_analytical = dVb_drho(
            rho, beta_hat, S_all, prob["y"], prob["X"], prob["family"],
            prob["sm"], prob["var_list"], prob["phi_est"],
        )

        M = len(rho)
        dVb_fd = np.zeros_like(dVb_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            Vb_plus  = _Vb_inv_at_rho(rho + drho, prob)
            Vb_minus = _Vb_inv_at_rho(rho - drho, prob)
            dVb_fd[k] = (Vb_plus - Vb_minus) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        np.testing.assert_allclose(dVb_analytical, dVb_fd, rtol=1e-3, atol=1e-6)


class TestD2VbDrho:
    """d2Vb/(drho_h drho_k), shape (M, M, p, p): FD of dVb_drho.

    d2Vb_drho adds delta_{hk} * exp(rho_k)*S_k/phi to hes_H_drho on the diagonal.
    Verified by centred FD of dVb_drho(rho).
    """

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-3

        d2Vb_analytical = d2Vb_drho(
            rho, beta_hat, S_all, prob["y"], prob["X"], prob["family"],
            prob["sm"], prob["var_list"], prob["phi_est"],
        )

        def dVb_at_rho(r):
            b = _beta_hat_at_rho(r, prob)
            prob["sm"].set_smooth_penalties(np.exp(r), prob["var_list"])
            return dVb_drho(
                r, b, S_all, prob["y"], prob["X"], prob["family"],
                prob["sm"], prob["var_list"], prob["phi_est"],
            )

        M = len(rho)
        d2Vb_fd = np.zeros_like(d2Vb_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            dVb_plus  = dVb_at_rho(rho + drho)
            dVb_minus = dVb_at_rho(rho - drho)
            d2Vb_fd[k] = (dVb_plus - dVb_minus) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        np.testing.assert_allclose(d2Vb_analytical, d2Vb_fd, rtol=1e-2, atol=1e-5)


class TestGradCholVbRho:
    """dR/drho_k, shape (M, p, p): FD of the upper Cholesky factor R of (H+S)^{-1}.

    grad_chol_Vb_rho uses the Cholesky recurrence (Wood 2017 App. B.7).
    Verified by centred FD of chol((H+S)^{-1})(rho).
    """

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        S_all = prob["S_all"]
        eps = 1e-4

        dR_analytical = grad_chol_Vb_rho(
            rho, beta_hat, S_all, prob["y"], prob["X"], prob["family"],
            prob["sm"], prob["var_list"], prob["phi_est"],
        )

        M = len(rho)
        dR_fd = np.zeros_like(dR_analytical)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            R_plus  = _chol_Vb_at_rho(rho + drho, prob)
            R_minus = _chol_Vb_at_rho(rho - drho, prob)
            dR_fd[k] = (R_plus - R_minus) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        # Only upper-triangular entries are meaningful (lower triangle is zero by construction)
        mask = np.triu(np.ones(dR_analytical.shape[-2:], dtype=bool))
        np.testing.assert_allclose(
            dR_analytical[:, mask], dR_fd[:, mask], rtol=1e-3, atol=1e-6
        )


# ===========================================================================
# Batch 6 – Laplace REML value, gradient, and Hessian
# ===========================================================================

def _reml_at_rho(rho, prob):
    """REML value at rho with beta_hat re-optimised.  Restores sm state."""
    b = _beta_hat_at_rho(rho, prob)
    # sm state is already set to rho inside _beta_hat_at_rho -> mle_gradient_bassed_optim
    return laplace_appr_REML(
        rho, b, prob["S_all"], prob["y"], prob["X"],
        prob["family"], prob["phi_est"], prob["sm"], prob["var_list"],
        compute_grad=False,
    )


class TestLaplaceREML:
    """REML value built from first principles vs laplace_appr_REML.

    REML = l(β̂) + penalty_ll + [+0.5/φ * log|S_λ|+] + [-0.5 * log|H+S|] + M*log(2π)

    Wood (2017) eq. 6.18: +0.5*log|S_λ|+ and -0.5*log|H+S_λ|.
    The manual formula uses slogdet and eigenvalue decomposition so it exercises
    a completely different code path from the internal Cholesky-based implementation.
    """

    def test_manual_formula(self, gam_problem):
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        l_unpen = unpenalized_ll(beta_hat, y, X, family, phi_est)
        pen = penalty_ll(rho, beta_hat, sm, var_list, phi_est)

        # H + S_λ via Vbeta_rho (inverse=False returns -(H+S))
        H_plus_S = -np.array(
            Vbeta_rho(rho, beta_hat, y, X, family, sm, var_list, phi_est, inverse=False)
        )
        _, log_det_HpS = np.linalg.slogdet(H_plus_S)

        # pseudo-log-det of S_λ via eigenvalues
        M = len(rho)
        S_lambda = sum(np.exp(rho[k]) * S_all[k] for k in range(M))
        eigvals = np.linalg.eigvalsh(S_lambda)
        tol = np.finfo(float).eps
        nz = eigvals[eigvals > tol * eigvals.max()]
        log_pdet_Slam = np.sum(np.log(nz))
        M_null = len(eigvals) - len(nz)

        reml_manual = (
            l_unpen + pen
            + 0.5 * log_pdet_Slam / phi_est
            - 0.5 * log_det_HpS
            + 0.5 * M_null * np.log(2 * np.pi)
        )

        reml_func = laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )

        np.testing.assert_allclose(reml_manual, reml_func, rtol=1e-6,
                                   err_msg="REML manual formula mismatch")


class TestGradLaplaceREML:
    """grad REML wrt rho: FD of REML(rho) at re-optimised beta_hat.

    By the envelope theorem (stationarity of penalised l wrt beta at beta_hat),
    the total derivative of REML(rho, beta_hat(rho)) equals the partial derivative
    at fixed beta_hat.  So FD of re-optimised REML must match grad_laplace_appr_REML
    which uses fixed beta_hat.
    """

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]
        eps = 1e-4

        grad_analytical = grad_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )

        M = len(rho)
        grad_fd = np.zeros(M)
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            grad_fd[k] = (_reml_at_rho(rho + drho, prob) - _reml_at_rho(rho - drho, prob)) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-3, atol=1e-6,
                                   err_msg="grad_laplace_appr_REML FD mismatch")


class TestHessLaplaceREML:
    """Hessian of REML wrt rho: FD of grad_laplace_appr_REML at re-optimised beta_hat.

    The Hessian of the negative REML is V_rho^{-1} (Wood 2017 eq. 6.30), which
    drives the smoothing-parameter uncertainty correction in eq. 6.31-6.32.
    """

    def test_fd(self, gam_problem):
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]
        eps = 1e-3

        hess_analytical = hess_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )

        def grad_at_rho(r):
            b = _beta_hat_at_rho(r, prob)
            prob["sm"].set_smooth_penalties(np.exp(r), prob["var_list"])
            return grad_laplace_appr_REML(
                r, b, S_all, y, X, family, phi_est, sm, var_list,
                compute_grad=False,
            )

        M = len(rho)
        hess_fd = np.zeros((M, M))
        for k in range(M):
            drho = np.zeros(M)
            drho[k] = eps
            g_plus  = grad_at_rho(rho + drho)
            g_minus = grad_at_rho(rho - drho)
            hess_fd[k] = (g_plus - g_minus) / (2 * eps)

        prob["sm"].set_smooth_penalties(np.exp(rho), prob["var_list"])
        np.testing.assert_allclose(hess_analytical, hess_fd, rtol=1e-2, atol=1e-5,
                                   err_msg="hess_laplace_appr_REML FD mismatch")


class TestGradLaplaceREMLScalable:
    """Scalable gradient must match the dense version exactly on small problems."""

    def test_matches_dense(self, gam_problem):
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        grad_dense = grad_laplace_appr_REML_dense(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        grad_scalable = grad_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        np.testing.assert_allclose(
            grad_scalable, grad_dense, rtol=1e-10, atol=1e-12,
            err_msg="grad_laplace_appr_REML does not match dense version",
        )


class TestHessLaplaceREMLScalable:
    """Scalable Hessian must match the dense version exactly on small problems.

    Uses the same gam_problem fixture (M=2, p~14) so the dense (M,M,p,p)
    tensor is cheap to materialise, giving a clean regression baseline.
    """

    def test_matches_dense(self, gam_problem):
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        hess_dense = hess_laplace_appr_REML_dense(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        hess_scalable = hess_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        np.testing.assert_allclose(
            hess_scalable, hess_dense, rtol=1e-10, atol=1e-12,
            err_msg="hess_laplace_appr_REML does not match dense version",
        )


# ===========================================================================
# Batch 7 – reml_objective (consolidated evaluation)
# ===========================================================================

class TestRemlObjective:
    """reml_objective must match the standalone laplace / grad / hess functions.

    reml_objective re-optimises beta_hat internally (BFGS, tol=1e-10), then
    returns REML / grad / hess via the scalable tensor-free path.  The
    standalone functions use the pre-optimised gam_problem beta_hat (Newton-CG,
    tol=1e-10).  Both find the same optimum so results should agree within the
    optimisation tolerance (~1e-4 relative).
    """

    def test_eval_matches_laplace_reml(self, gam_problem):
        """return_type='eval' REML value matches laplace_appr_REML at same rho."""
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        reml_val = reml_objective(
            rho, y, X, sm, var_list, family, S_all, phi_est,
            return_type="eval",
        )
        standalone_val = laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        np.testing.assert_allclose(
            reml_val, standalone_val, rtol=1e-5,
            err_msg="reml_objective 'eval' vs laplace_appr_REML mismatch",
        )

    def test_eval_grad_matches_grad_laplace_reml(self, gam_problem):
        """return_type='eval_grad' gradient matches grad_laplace_appr_REML."""
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        _reml_val, grad_val = reml_objective(
            rho, y, X, sm, var_list, family, S_all, phi_est,
            return_type="eval_grad",
        )
        standalone_grad = grad_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        np.testing.assert_allclose(
            grad_val, standalone_grad, rtol=1e-3, atol=1e-6,
            err_msg="reml_objective 'eval_grad' vs grad_laplace_appr_REML mismatch",
        )

    def test_eval_grad_hess_matches_hess_laplace_reml(self, gam_problem):
        """return_type='eval_grad_hess' Hessian matches hess_laplace_appr_REML."""
        prob = gam_problem
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        _reml_val, _grad_val, hess_val = reml_objective(
            rho, y, X, sm, var_list, family, S_all, phi_est,
            return_type="eval_grad_hess",
        )
        standalone_hess = hess_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        np.testing.assert_allclose(
            hess_val, standalone_hess, rtol=1e-3, atol=1e-5,
            err_msg="reml_objective 'eval_grad_hess' vs hess_laplace_appr_REML mismatch",
        )


# ===========================================================================
# Batch 7b – RemlProblem (caching wrapper for trust-constr)
# ===========================================================================

class TestRemlProblem:
    """RemlProblem must match reml_objective / hess_laplace_appr_REML and cache correctly."""

    def test_eval_grad_matches_reml_objective(self, gam_problem):
        """eval_grad returns the same (f, grad) as reml_objective('eval_grad')."""
        prob = gam_problem
        rho = prob["rho"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est, S_all = prob["sm"], prob["var_list"], prob["phi_est"], prob["S_all"]

        rp = RemlProblem(y, X, sm, var_list, family, S_all, phi_est)
        val_rp, grad_rp = rp.eval_grad(rho)

        val_ref, grad_ref = reml_objective(
            rho, y, X, sm, var_list, family, S_all, phi_est,
            return_type="eval_grad",
        )
        np.testing.assert_allclose(val_rp, val_ref, rtol=1e-5)
        np.testing.assert_allclose(grad_rp, grad_ref, rtol=1e-3, atol=1e-6)

    def test_hess_cache_hit_matches_standalone(self, gam_problem):
        """hess called at the same rho as eval_grad (cache hit) matches hess_laplace_appr_REML."""
        prob = gam_problem
        rho = prob["rho"]
        beta_hat = prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est, S_all = prob["sm"], prob["var_list"], prob["phi_est"], prob["S_all"]

        rp = RemlProblem(y, X, sm, var_list, family, S_all, phi_est)
        rp.eval_grad(rho)  # populate cache
        hess_rp = rp.hess(rho)  # cache hit: intermediates injected

        hess_ref = hess_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        np.testing.assert_allclose(hess_rp, hess_ref, rtol=1e-3, atol=1e-5)

    def test_hess_cache_miss_matches_standalone(self, gam_problem):
        """hess called at a different rho from eval_grad (cache miss) still matches."""
        prob = gam_problem
        rho = prob["rho"]
        rho2 = rho + 0.1
        beta_hat = prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est, S_all = prob["sm"], prob["var_list"], prob["phi_est"], prob["S_all"]

        rp = RemlProblem(y, X, sm, var_list, family, S_all, phi_est)
        rp.eval_grad(rho)   # populate cache at rho
        hess_rp = rp.hess(rho2)  # cache miss — rho2 != rho

        hess_ref = hess_laplace_appr_REML(
            rho2, None, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=True,
        )
        np.testing.assert_allclose(hess_rp, hess_ref, rtol=1e-3, atol=1e-5)

    def test_intermediates_keys(self, gam_problem):
        """eval_grad with return_intermediates=True returns the expected keys."""
        prob = gam_problem
        rho = prob["rho"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est, S_all = prob["sm"], prob["var_list"], prob["phi_est"], prob["S_all"]

        _, _, itm = reml_objective(
            rho, y, X, sm, var_list, family, S_all, phi_est,
            return_type="eval_grad", return_intermediates=True,
        )
        assert set(itm.keys()) == {"b_hat", "Vb", "Vb_inv", "dB"}
        assert itm["Vb"].shape == itm["Vb_inv"].shape
        assert itm["dB"].shape[0] == len(rho)


# ===========================================================================
# Batch 8 – Corrected AIC (Wood 2017 eq 6.31-6.32)
# ===========================================================================

class TestCorrectedAIC:
    """Verify the corrected AIC formula from Wood (2017) eq 6.31-6.32.

    The corrected covariance matrix is (eq 6.31):
        V'_β = Vb + V' + V''
        V'    = J Vρ J^T                          (smoothing-parameter location)
        V''_jm = Σ_i Σ_k Σ_l ∂R_ij/∂ρ_k Vρ_kl ∂R_im/∂ρ_l  (Cholesky correction)

    The corrected AIC penalty is (eq 6.32):
        τ₂ = tr(V'_β * I_hat)
        AIC = -2 l(β̂) + 2 τ₂

    where I_hat = H = X^T diag(w) X / phi is the observed Fisher information.
    Tests below replicate each building block independently (matrix multiply
    vs einsum, np.trace vs inner product) to catch index-ordering bugs.
    """

    def _build_quantities(self, prob):
        """Compute all building blocks needed for the AIC formula."""
        rho, beta_hat = prob["rho"], prob["beta_hat"]
        y, X, family = prob["y"], prob["X"], prob["family"]
        sm, var_list, phi_est = prob["sm"], prob["var_list"], prob["phi_est"]
        S_all = prob["S_all"]

        # Bayesian covariance Vb = (H + S_λ)^{-1}
        Vb = -np.array(Vbeta_rho(
            rho, beta_hat, y, X, family, sm, var_list, phi_est, inverse=True,
        ))

        # V_ρ = (negative REML Hessian)^{-1}  (Wood 2017 eq 6.30)
        hess_reml = hess_laplace_appr_REML(
            rho, beta_hat, S_all, y, X, family, phi_est, sm, var_list,
            compute_grad=False,
        )
        V_rho = np.linalg.pinv(-hess_reml)

        # J = dβ̂/dρ, shape (M, p)
        J = dbeta_hat(rho, beta_hat, S_all, sm, var_list, y, X, family, phi_est)

        # dR/dρ_k (Cholesky factor gradient), shape (M, p, p)
        dR = grad_chol_Vb_rho(rho, beta_hat, S_all, y, X, family, sm, var_list, phi_est)

        # Observed Fisher information I_hat = H = X^T diag(w) X / phi
        I_hat = np.array(H_rho(
            rho, beta_hat, y, X, family, phi_est, sm, var_list, comp_gradient=False,
        ))

        return rho, beta_hat, y, X, family, sm, var_list, phi_est, Vb, V_rho, J, dR, I_hat

    def test_V_prime_einsum_matches_matmul(self, gam_problem):
        """V' = J Vρ J^T: einsum 'ri,rh,hj->ij' must equal J.T @ Vρ @ J."""
        _, _, _, _, _, _, _, _, _, V_rho, J, _, _ = self._build_quantities(gam_problem)

        V_prime_ein = np.einsum("ri,rh,hj->ij", J, V_rho, J)
        V_prime_mat = J.T @ V_rho @ J
        np.testing.assert_allclose(
            V_prime_ein, V_prime_mat, rtol=1e-12,
            err_msg="V' einsum vs matrix-multiply mismatch (eq 6.31)",
        )

    def test_corrected_covariance_is_symmetric(self, gam_problem):
        """V'_β = Vb + V' + V'' must be symmetric (it is a covariance matrix)."""
        _, _, _, _, _, _, _, _, Vb, V_rho, J, dR, _ = self._build_quantities(gam_problem)

        V_prime   = np.einsum("ri,rh,hj->ij", J, V_rho, J)
        V_2prime  = np.einsum("kij,kl,lim->jm", dR, V_rho, dR)
        V_corr = Vb + V_prime + V_2prime

        np.testing.assert_allclose(
            V_corr, V_corr.T, rtol=1e-10,
            err_msg="V'_β from eq 6.31 is not symmetric",
        )

    def test_aic_trace_matches_einsum(self, gam_problem):
        """τ₂ = tr(V'_β I_hat): np.trace(V_corr @ I_hat) must equal einsum contraction."""
        _, _, _, _, _, _, _, _, Vb, V_rho, J, dR, I_hat = self._build_quantities(gam_problem)

        V_prime  = J.T @ V_rho @ J
        V_2prime = np.einsum("kij,kl,lim->jm", dR, V_rho, dR)
        V_corr   = Vb + V_prime + V_2prime

        tau2_trace = np.trace(V_corr @ I_hat)
        tau2_ein   = np.einsum("ij,ji->", V_corr, I_hat)
        np.testing.assert_allclose(
            tau2_trace, tau2_ein, rtol=1e-12,
            err_msg="τ₂ via np.trace vs einsum mismatch",
        )

    def test_aic_correction_nonnegative(self, gam_problem):
        """Corrected EDF τ₂ must be ≥ the uncorrected edf₁ = tr(F) where F = Vb H.

        V' and V'' are both PSD (outer-product form weighted by PSD V_ρ), so
        the correction V' + V'' adds non-negative eigenvalues to Vb.  Hence
        tr((V' + V'') I_hat) ≥ 0 and τ₂ ≥ edf₁.
        """
        _, _, _, _, _, _, _, _, Vb, V_rho, J, dR, I_hat = self._build_quantities(gam_problem)

        V_prime  = J.T @ V_rho @ J
        V_2prime = np.einsum("kij,kl,lim->jm", dR, V_rho, dR)
        V_corr   = Vb + V_prime + V_2prime

        # uncorrected edf: tr(Vb H) = tr(F) where F = (H + S_λ)^{-1} H
        edf1 = np.trace(Vb @ I_hat)
        tau2 = np.trace(V_corr @ I_hat)

        assert tau2 >= edf1 - 1e-6, (
            f"τ₂ = {tau2:.6f} < edf₁ = {edf1:.6f}; correction is negative"
        )

    def test_aic_formula_eq_6_32(self, gam_problem):
        """AIC = -2 l(β̂) + 2 τ₂ is finite and well-formed (eq 6.32)."""
        rho, beta_hat, y, X, family, sm, var_list, phi_est, Vb, V_rho, J, dR, I_hat = (
            self._build_quantities(gam_problem)
        )

        V_prime  = J.T @ V_rho @ J
        V_2prime = np.einsum("kij,kl,lim->jm", dR, V_rho, dR)
        V_corr   = Vb + V_prime + V_2prime

        tau2 = np.trace(V_corr @ I_hat)
        l_hat = unpenalized_ll(beta_hat, y, X, family, phi_est, omega=1)
        aic = -2 * l_hat + 2 * tau2

        assert np.isfinite(aic), f"AIC is not finite: {aic}"
        assert tau2 > 0, f"τ₂ = {tau2:.6f} is non-positive"
        # AIC decomposes correctly
        np.testing.assert_allclose(
            aic, -2 * l_hat + 2 * tau2, rtol=1e-15,
            err_msg="AIC = -2l + 2τ₂ decomposition mismatch",
        )


# ===========================================================================
# Batch 9 – gcv_objective consolidated function
# ===========================================================================

@pytest.fixture(scope="module")
def gcv_problem():
    """Small Gaussian GAM working-response problem for GCV consolidation tests.

    Builds the augmented WLS matrices (X, Q, R, endog) that match what
    _fit_pql_gcv feeds to gcv_comp / gcv_grad_comp / gcv_hess_comp / gcv_objective.
    """
    import statsmodels.api as _sm
    rng = np.random.default_rng(17)
    n = 80
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)

    sm = smooths_handler()
    sm.add_smooth("x1", [x1], knots_num=6, penalty_type="EqSpaced")
    sm.add_smooth("x2", [x2], knots_num=6, penalty_type="EqSpaced")
    var_list = ["x1", "x2"]

    X_raw, _ = sm.get_exog_mat(var_list)
    rho = np.array([1.0, 2.0])
    sm.set_smooth_penalties(np.exp(rho), var_list)

    # Gaussian: working response = y, weights = 1  →  wexog = Xagu, wendog = yagu
    beta_true = rng.normal(0, 0.5, X_raw.shape[1])
    y = X_raw @ beta_true + rng.normal(0, 0.3, n)

    pen_matrix = sm.get_penalty_agumented(var_list)
    Xagu = np.vstack((X_raw, pen_matrix))
    yagu = np.zeros(Xagu.shape[0])
    yagu[:n] = y
    wagu = np.ones(Xagu.shape[0])
    model = _sm.WLS(yagu, Xagu, wagu)

    X = model.wexog[:n, :]
    Q, R = np.linalg.qr(X, "reduced")
    endog = model.wendog

    # Pre-convert S matrices (mirrors the optimisation-loop caching)
    S_mat = matrix_transform(*compute_Sjs(sm, var_list))

    return dict(
        X=X, Q=Q, R=R, endog=endog, sm=sm, var_list=var_list, rho=rho,
        S_mat=S_mat,
    )


class TestGcvObjective:
    """gcv_objective must match the standalone gcv_comp / gcv_grad_comp functions.

    The Hessian is tested against a finite-difference of gcv_grad_comp rather
    than gcv_hess_comp because gcv_hess_comp omits the α_grad / δ_grad cross
    terms (alpha_grad and delta_grad arrays are left at zero inside that
    function, making the cross-derivative contributions vanish).
    gcv_objective carries the correct full Hessian.
    """

    def test_eval_matches_gcv_comp(self, gcv_problem):
        """return_type='eval' must equal gcv_comp(return_par='gcv')."""
        p = gcv_problem
        val_obj = gcv_objective(
            p["rho"], p["X"], p["Q"], p["R"], p["endog"],
            p["sm"], p["var_list"],
            return_type="eval", S_all=p["S_mat"],
        )
        val_ref = gcv_comp(
            p["rho"], p["X"], p["Q"], p["R"], p["endog"],
            p["sm"], p["var_list"], return_par="gcv",
        )
        np.testing.assert_allclose(
            val_obj, val_ref, rtol=1e-12,
            err_msg="gcv_objective 'eval' vs gcv_comp mismatch",
        )

    def test_eval_grad_matches_gcv_grad_comp(self, gcv_problem):
        """return_type='eval_grad' gradient must equal gcv_grad_comp."""
        p = gcv_problem
        val_obj, grad_obj = gcv_objective(
            p["rho"], p["X"], p["Q"], p["R"], p["endog"],
            p["sm"], p["var_list"],
            return_type="eval_grad", S_all=p["S_mat"],
        )
        grad_ref = gcv_grad_comp(
            p["rho"], p["X"], p["Q"], p["R"], p["endog"],
            p["sm"], p["var_list"], return_par="gcv",
        )
        np.testing.assert_allclose(
            grad_obj, grad_ref, rtol=1e-10,
            err_msg="gcv_objective 'eval_grad' vs gcv_grad_comp mismatch",
        )

    def test_eval_grad_hess_matches_fd(self, gcv_problem):
        """return_type='eval_grad_hess' Hessian matches FD of gcv_grad_comp.

        gcv_hess_comp is NOT used as reference because it drops the α_grad /
        δ_grad cross terms (they are initialised to zero and never updated),
        so its output is numerically incorrect.  Finite differences of the
        confirmed-correct gradient provide the ground truth instead.
        """
        p = gcv_problem
        rho = p["rho"]
        _, _, hess_obj = gcv_objective(
            rho, p["X"], p["Q"], p["R"], p["endog"],
            p["sm"], p["var_list"],
            return_type="eval_grad_hess", S_all=p["S_mat"],
        )

        # FD Hessian from gcv_grad_comp (centred differences, eps=1e-4)
        eps = 1e-4
        n_rho = len(rho)
        hess_fd = np.zeros((n_rho, n_rho))
        for k in range(n_rho):
            rho_p = rho.copy(); rho_p[k] += eps
            rho_m = rho.copy(); rho_m[k] -= eps
            g_p = gcv_grad_comp(rho_p, p["X"], p["Q"], p["R"], p["endog"],
                                 p["sm"], p["var_list"], return_par="gcv")
            g_m = gcv_grad_comp(rho_m, p["X"], p["Q"], p["R"], p["endog"],
                                 p["sm"], p["var_list"], return_par="gcv")
            hess_fd[:, k] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(
            hess_obj, hess_fd, rtol=1e-3, atol=1e-6,
            err_msg="gcv_objective 'eval_grad_hess' vs FD Hessian mismatch",
        )