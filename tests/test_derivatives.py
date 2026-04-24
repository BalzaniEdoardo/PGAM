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