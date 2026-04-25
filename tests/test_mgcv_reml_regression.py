"""
Regression test: Laplace-REML formula against mgcv's C-level computation.

The fixture files in tests/data/ were produced by _scripts/gen_mgcv_reml_data.R,
which fits a Poisson GAM with mgcv (method="REML") and records:
  - the design matrix X, response y, full-dim penalty matrices S_j, and rho
    extracted from the fitted model
  - the ground-truth REML value  V_r = -mod$gcv.ubre  from mgcv's internal
    C code (gam.fit5)

This test is NOT circular: the reference comes from mgcv's implementation,
our value is assembled from the same building blocks used by laplace_appr_REML
(unpenalized_ll, logDet_Slam, Vbeta_rho) applied to the mgcv-extracted inputs.

To regenerate the fixtures:
    Rscript _scripts/gen_mgcv_reml_data.R
"""
import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from PGAM.der_wrt_smoothing import (
    unpenalized_ll,
    penalty_ll_Slam,
    d2variance_family,
    deriv3_link,
    Vbeta_rho,
)
from PGAM.deriv_det_Slam import transform_Slam, logDet_Slam

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_fixtures():
    scalars_df = pd.read_csv(os.path.join(DATA_DIR, "mgcv_reml_scalars.csv"))
    scalars    = dict(zip(scalars_df["name"], scalars_df["value"]))

    n    = int(scalars["n"])
    p    = int(scalars["p"])
    M_sp = int(scalars["M_sp"])

    Mp        = int(scalars["Mp"])        # null-space dim from mgcv's totalPenaltySpace
    y         = np.array([scalars[f"y_{i+1}"]    for i in range(n)])
    beta      = np.array([scalars[f"beta_{i+1}"] for i in range(p)])
    rho       = np.array([scalars[f"rho_{i+1}"]  for i in range(M_sp)])
    REML_mgcv = float(scalars["REML_mgcv"])

    X_df = pd.read_csv(os.path.join(DATA_DIR, "mgcv_reml_X.csv"))
    X    = X_df.values.astype(float)

    S_df  = pd.read_csv(os.path.join(DATA_DIR, "mgcv_reml_S.csv"))
    S_all = []
    for k in range(1, M_sp + 1):
        block = S_df[S_df["s_idx"] == k].drop(columns="s_idx").values.astype(float)
        S_all.append(block)

    return X, y, beta, rho, S_all, Mp, REML_mgcv


def _make_poisson_family():
    base      = sm.families.Poisson(link=sm.families.links.Log())
    base.link = deriv3_link(base.link, run_tests=False)
    return d2variance_family(base, run_tests=False)


class _MinimalSmHandler:
    """Minimal sm_handler interface for Vbeta_rho: only set/get penalty needed."""

    def __init__(self, S_all):
        self._S_all = S_all
        self._lam   = np.ones(len(S_all))

    def set_smooth_penalties(self, lam, var_list):
        self._lam = np.asarray(lam, dtype=float)

    def get_penalty_agumented(self, var_list):
        S_lam = sum(self._lam[j] * self._S_all[j] for j in range(len(self._S_all)))
        d, V  = np.linalg.eigh(S_lam)
        d     = np.clip(d, 0.0, None)
        return (V * np.sqrt(d)).T          # Cholesky-like square root


@pytest.fixture(scope="module")
def mgcv_data():
    pytest.importorskip("pandas")
    if not os.path.isdir(DATA_DIR):
        pytest.skip("Fixture data not found; run _scripts/gen_mgcv_reml_data.R")
    missing = [f for f in ("mgcv_reml_scalars.csv", "mgcv_reml_X.csv",
                            "mgcv_reml_S.csv")
               if not os.path.isfile(os.path.join(DATA_DIR, f))]
    if missing:
        pytest.skip(f"Missing fixture files: {missing}")
    return _load_fixtures()


class TestMgcvRemlRegression:  # noqa: D101
    """Our Laplace-REML formula vs mgcv ground truth (rtol=1e-4).

    Assembles V_r from the same building blocks as laplace_appr_REML:
        V_r = unpenalized_ll
              + penalty_ll_Slam          (-0.5/phi * beta^T S_lambda beta)
              + 0.5/phi * logDet_Slam    (+0.5 * log|S_lambda|+)
              + log_det_sum              (-0.5 * log|H + S_lambda|)
              + 0.5 * M_null * log(2*pi)
    """

    def test_reml_matches_mgcv(self, mgcv_data):
        X, y, beta, rho, S_all, Mp, REML_mgcv = mgcv_data
        phi_est    = 1.0
        family     = _make_poisson_family()
        sm_handler = _MinimalSmHandler(S_all)
        var_list   = [f"s{k}" for k in range(len(S_all))]

        ll  = unpenalized_ll(beta, y, X, family, phi_est)

        lam   = np.exp(rho)
        S_lam = sum(lam[j] * S_all[j] for j in range(len(S_all)))
        pen   = penalty_ll_Slam(S_lam, beta, phi_est)

        _, S_transf = transform_Slam(S_all, rho)
        log_pdet_S  = logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all)
        log_det_Slam_term = 0.5 * log_pdet_S / phi_est

        _, log_det_HpS = Vbeta_rho(
            rho, beta, y, X, family, sm_handler, var_list, phi_est,
            inverse=False, compute_grad=False, return_logdet=True,
        )
        log_det_sum = -0.5 * log_det_HpS

        reml_py = (
            ll + pen
            + log_det_Slam_term
            + log_det_sum
            + 0.5 * Mp * np.log(2 * np.pi)
        )

        np.testing.assert_allclose(
            reml_py, REML_mgcv, rtol=1e-4,
            err_msg=(
                f"Python V_r={reml_py:.6f} vs mgcv V_r={REML_mgcv:.6f}"
            ),
        )
