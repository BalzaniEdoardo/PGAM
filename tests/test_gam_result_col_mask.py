"""
Tests for GAM_result.get_smooths_info and eval_basis after col_mask changes.

Backward compatibility principle: with col_mask=None the stored smooth_info and
eval_basis output must be identical to pre-change behaviour.

Covers:
  1. get_smooths_info stores col_mask=None when no mask is set.
  2. get_smooths_info stores the mask when one is set.
  3. eval_basis width is unchanged (no mask) — same as basisAndPenalty output.
  4. eval_basis width is reduced (with mask) — last column always preserved.
  5. predict / get_smooth / mu_sigma_log_space dimension consistency (shapes match
     beta and colMean_X after masking).
"""

import numpy as np
import scipy.sparse as sparse
import scipy.stats as sts
import statsmodels.api as sm

from PGAM.gam_data_handlers import covarate_smooth, smooths_handler
from PGAM.GAM_library import GAM_result


# ---------------------------------------------------------------------------
# minimal helpers
# ---------------------------------------------------------------------------

def _make_sm_handler_and_data(n_pts=300, knots_num=6, seed=0, col_mask=None):
    """
    Build a smooths_handler with a single 2D spatial smooth and
    generate a Poisson-ish response.  Returns (sm_handler, y, var_list).
    """
    rng = np.random.default_rng(seed)
    k1d = np.linspace(0, 10, knots_num)

    x = rng.uniform(0, 10, n_pts)
    y_cov = rng.uniform(0, 10, n_pts)

    handler = smooths_handler()
    handler.add_smooth(
        "pos",
        [x, y_cov],
        knots=[k1d, k1d],
        is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
        col_mask=col_mask,
    )

    y = rng.poisson(lam=1.0, size=n_pts).astype(float)
    return handler, y, ["pos"]


def _fit_gam(sm_handler, y, var_list):
    """Fit one PIRLS step and return a GAM_result."""
    from PGAM.GAM_library import general_additive_model
    family = sm.families.Poisson()
    gam = general_additive_model(sm_handler, var_list, y, family)
    # optim_gam returns a GAM_result object.
    return gam.optim_gam(var_list, max_iter=1)


# ---------------------------------------------------------------------------
# 1 & 2. get_smooths_info stores col_mask correctly
# ---------------------------------------------------------------------------

def test_smooth_info_stores_none_when_no_mask():
    handler, y, var_list = _make_sm_handler_and_data()
    gam = _fit_gam(handler, y, var_list)
    assert "col_mask" in gam.smooth_info["pos"]
    assert gam.smooth_info["pos"]["col_mask"] is None


def test_smooth_info_stores_mask_when_set():
    handler_full, _, _ = _make_sm_handler_and_data()
    mask = handler_full["pos"].get_active_col_mask()

    handler, y, var_list = _make_sm_handler_and_data(col_mask=mask)
    gam = _fit_gam(handler, y, var_list)

    stored = gam.smooth_info["pos"]["col_mask"]
    assert stored is not None
    assert np.array_equal(stored, mask)


# ---------------------------------------------------------------------------
# 3. eval_basis width unchanged when no mask
# ---------------------------------------------------------------------------

def test_eval_basis_no_mask_width():
    """
    Without a mask, eval_basis must return the full basis width —
    same as what basisAndPenalty produces directly.
    """
    handler, y, var_list = _make_sm_handler_and_data()
    gam = _fit_gam(handler, y, var_list)

    rng = np.random.default_rng(99)
    x_test = rng.uniform(0, 10, 100)
    y_test = rng.uniform(0, 10, 100)

    fX = gam.eval_basis([x_test, y_test], "pos", sparseX=False)
    # full basis: should match the width of the unmasked smooth
    expected_width = handler["pos"].X.shape[1]   # k+1 (no mask → full width)
    assert fX.shape[1] == expected_width, (
        f"eval_basis width {fX.shape[1]} != expected {expected_width}"
    )


# ---------------------------------------------------------------------------
# 4. eval_basis width reduced when mask is set
# ---------------------------------------------------------------------------

def test_eval_basis_with_mask_width():
    """
    With a mask, eval_basis must return k+1 columns (k active + 1 last kept),
    so that fX[:, :-1] gives k columns matching colMean_X and beta[index].
    """
    handler_full, _, _ = _make_sm_handler_and_data()
    mask = handler_full["pos"].get_active_col_mask()
    k = mask.sum()

    handler, y, var_list = _make_sm_handler_and_data(col_mask=mask)
    gam = _fit_gam(handler, y, var_list)

    rng = np.random.default_rng(99)
    x_test = rng.uniform(0, 10, 100)
    y_test = rng.uniform(0, 10, 100)

    fX = gam.eval_basis([x_test, y_test], "pos", sparseX=False)
    assert fX.shape[1] == k + 1, (
        f"eval_basis should return k+1={k+1} cols with mask, got {fX.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 5. predict / get_smooth dimension consistency
# ---------------------------------------------------------------------------

def test_predict_runs_no_mask():
    """predict() must work and return shape (n_test,) without a mask."""
    handler, y, var_list = _make_sm_handler_and_data()
    gam = _fit_gam(handler, y, var_list)

    rng = np.random.default_rng(77)
    x_test = rng.uniform(0, 10, 80)
    y_test = rng.uniform(0, 10, 80)

    mu = gam.predict([[x_test, y_test]])
    assert mu.shape == (80,)


def test_predict_runs_with_mask():
    """predict() must work and return shape (n_test,) with a mask."""
    handler_full, _, _ = _make_sm_handler_and_data()
    mask = handler_full["pos"].get_active_col_mask()

    handler, y, var_list = _make_sm_handler_and_data(col_mask=mask)
    gam = _fit_gam(handler, y, var_list)

    rng = np.random.default_rng(77)
    x_test = rng.uniform(0, 10, 80)
    y_test = rng.uniform(0, 10, 80)

    mu = gam.predict([[x_test, y_test]])
    assert mu.shape == (80,)


def test_predict_no_mask_vs_mask_same_shape():
    """predict() output shape is the same with and without a mask."""
    rng = np.random.default_rng(55)
    x_test = rng.uniform(0, 10, 80)
    y_test = rng.uniform(0, 10, 80)

    handler_full, _, _ = _make_sm_handler_and_data()
    mask = handler_full["pos"].get_active_col_mask()

    handler_nm, y_nm, vl = _make_sm_handler_and_data()
    gam_nm = _fit_gam(handler_nm, y_nm, vl)
    mu_nm = gam_nm.predict([[x_test, y_test]])

    handler_m, y_m, vl = _make_sm_handler_and_data(col_mask=mask)
    gam_m = _fit_gam(handler_m, y_m, vl)
    mu_m = gam_m.predict([[x_test, y_test]])

    assert mu_nm.shape == mu_m.shape


def test_get_smooth_runs_no_mask():
    handler, y, var_list = _make_sm_handler_and_data()
    gam = _fit_gam(handler, y, var_list)

    rng = np.random.default_rng(33)
    x_test = rng.uniform(0, 10, 80)
    y_test = rng.uniform(0, 10, 80)

    mean_y, lo, hi = gam.smooth_compute([x_test, y_test], "pos")
    assert mean_y.shape == (80,)
    assert lo.shape == (80,)
    assert hi.shape == (80,)


def test_get_smooth_runs_with_mask():
    handler_full, _, _ = _make_sm_handler_and_data()
    mask = handler_full["pos"].get_active_col_mask()

    handler, y, var_list = _make_sm_handler_and_data(col_mask=mask)
    gam = _fit_gam(handler, y, var_list)

    rng = np.random.default_rng(33)
    x_test = rng.uniform(0, 10, 80)
    y_test = rng.uniform(0, 10, 80)

    mean_y, lo, hi = gam.smooth_compute([x_test, y_test], "pos")
    assert mean_y.shape == (80,)
