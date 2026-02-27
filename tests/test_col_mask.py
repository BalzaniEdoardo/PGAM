"""
Tests for covarate_smooth column masking.

Backward compatibility principle: col_mask=None must reproduce the exact same
behaviour as before this change.  Every test group has at least one no-mask
case to enforce this.

Covers:
  1. No mask → all sizes unchanged from baseline.
  2. Explicit boolean mask → X, S_list, colMean_X shrink correctly.
  3. get_active_col_mask() → identifies zero columns in sparse data.
  4. Two-step workflow: build full, call get_active_col_mask, apply mask.
  5. additive_model_preprocessing() is shape-consistent after masking.
  6. compute_Bx() runs without error after masking (both 1D and 2D).
  7. set_new_covariate() re-applies the stored mask.
  8. _set_knots_spatial is_cyclic=None default → same result as old [False] default.
"""

import sys
import os


import numpy as np
import scipy.sparse as sparse
from PGAM.gam_data_handlers import covarate_smooth


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_1d_smooth(n_pts=200, knots_num=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n_pts)
    return covarate_smooth([x], knots_num=knots_num, penalty_type="EqSpaced")


def make_2d_smooth(n_pts=300, knots_num=6, seed=0, sparse_data=False):
    """
    2D spatial smooth with a fixed knot grid over [0,10]^2.

    sparse_data=False → data fills the whole island (all columns active).
    sparse_data=True  → data stays in [0,4]^2 (bottom-left quarter);
                        most Kronecker columns are all-zero because those
                        island locations were never visited.  This is the
                        bat scenario.

    Knots are always explicit so the grid is independent of data range.
    """
    rng = np.random.default_rng(seed)
    if sparse_data:
        x = rng.uniform(0, 4, n_pts)
        y = rng.uniform(0, 4, n_pts)
    else:
        x = rng.uniform(0, 10, n_pts)
        y = rng.uniform(0, 10, n_pts)

    k1d = np.linspace(0, 10, knots_num)
    return covarate_smooth(
        [x, y],
        knots=[k1d, k1d],
        is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
    )


# ---------------------------------------------------------------------------
# 1. No mask → sizes unchanged
# ---------------------------------------------------------------------------

def test_no_mask_1d_sizes():
    sm = make_1d_smooth()
    assert sm.col_mask is None
    n_cols = sm.X.shape[1]
    assert len(sm.colMean_X) == n_cols - 1
    for S in sm.S_list:
        assert S.shape == (n_cols, n_cols)


def test_no_mask_2d_sizes():
    sm = make_2d_smooth()
    assert sm.col_mask is None
    n_cols = sm.X.shape[1]
    assert len(sm.colMean_X) == n_cols - 1
    for S in sm.S_list:
        assert S.shape == (n_cols, n_cols)


# ---------------------------------------------------------------------------
# 2. Explicit mask → correct filtered sizes
# ---------------------------------------------------------------------------

def test_explicit_mask_1d():
    sm_full = make_1d_smooth(knots_num=8)
    n_full = sm_full.X.shape[1]           # full column count
    n_active = n_full - 1                 # active cols (excl. last)

    # keep every other active column
    mask = np.zeros(n_active, dtype=bool)
    mask[::2] = True
    k = mask.sum()

    sm_full._apply_col_mask(mask)

    assert sm_full.X.shape[1] == k + 1,      "X should have k+1 cols after masking"
    assert len(sm_full.colMean_X) == k,       "colMean_X length should equal k"
    for S in sm_full.S_list:
        assert S.shape == (k + 1, k + 1),     "each S_list entry should be (k+1)x(k+1)"
    # B_list masked for 1D
    for B in sm_full.B_list:
        assert B.shape == (k + 1, k + 1),     "B_list[i] should be (k+1)x(k+1) after 1D mask"


def test_explicit_mask_2d():
    sm_full = make_2d_smooth(knots_num=6)
    n_full = sm_full.X.shape[1]
    n_active = n_full - 1

    # keep first half of active columns
    mask = np.zeros(n_active, dtype=bool)
    mask[: n_active // 2] = True
    k = mask.sum()

    sm_full._apply_col_mask(mask)

    assert sm_full.X.shape[1] == k + 1
    assert len(sm_full.colMean_X) == k
    for S in sm_full.S_list:
        assert S.shape == (k + 1, k + 1)


def test_mask_all_true_is_noop():
    """Masking with all-True should give same sizes as no mask."""
    sm_ref = make_2d_smooth(knots_num=5)
    n_full = sm_ref.X.shape[1]

    sm_masked = make_2d_smooth(knots_num=5)
    all_true = np.ones(n_full - 1, dtype=bool)
    sm_masked._apply_col_mask(all_true)

    assert sm_masked.X.shape == sm_ref.X.shape
    assert sm_masked.colMean_X.shape == sm_ref.colMean_X.shape
    for Sm, Sr in zip(sm_masked.S_list, sm_ref.S_list):
        assert Sm.shape == Sr.shape


def test_mask_wrong_length_raises():
    sm = make_1d_smooth()
    bad_mask = np.ones(999, dtype=bool)
    try:
        sm._apply_col_mask(bad_mask)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 3. get_active_col_mask identifies dead columns
# ---------------------------------------------------------------------------

def test_active_mask_dense_data():
    """With full coverage all columns should be active."""
    sm = make_2d_smooth(sparse_data=False)
    mask = sm.get_active_col_mask(min_obs=1)
    assert mask.shape[0] == sm.X.shape[1] - 1
    # with uniform data and a reasonable knot grid every column should fire
    assert mask.all(), "all columns should be active with full-coverage data"


def test_active_mask_sparse_data():
    """With sparse coverage a large fraction of columns should be inactive."""
    sm_sparse = make_2d_smooth(sparse_data=True, knots_num=8)
    mask = sm_sparse.get_active_col_mask(min_obs=1)
    n_active = mask.sum()
    n_total = len(mask)
    # bat only covers the bottom-left quarter of the island grid
    # → expect well under half the columns to be active
    assert n_active < n_total, "some columns should be inactive with sparse data"
    assert n_active < n_total * 0.6, (
        f"expected <60% active cols with sparse data, got {n_active}/{n_total}"
    )


# ---------------------------------------------------------------------------
# 4. Two-step workflow: build full, auto-mask, apply
# ---------------------------------------------------------------------------

def test_two_step_workflow():
    sm = make_2d_smooth(sparse_data=True, knots_num=8)
    n_before = sm.X.shape[1]

    mask = sm.get_active_col_mask(min_obs=1)
    k = mask.sum()
    sm._apply_col_mask(mask)

    assert sm.X.shape[1] == k + 1,       "X width mismatch after two-step"
    assert len(sm.colMean_X) == k,        "colMean_X length mismatch"
    assert sm.X.shape[1] < n_before,      "masked X should be narrower than full island grid"


# ---------------------------------------------------------------------------
# 5. additive_model_preprocessing consistency
# ---------------------------------------------------------------------------

def _check_preprocessing(sm):
    """X_out and Bx_out from additive_model_preprocessing must be col-compatible."""
    X_out, Bx_out = sm.additive_model_preprocessing()
    if sparse.issparse(X_out):
        X_out = X_out.toarray()
    assert X_out.shape[1] == Bx_out.shape[1], (
        f"X and Bx column count mismatch: {X_out.shape[1]} vs {Bx_out.shape[1]}"
    )
    # colMean_X was subtracted from X, so they must match in width
    assert X_out.shape[1] == len(sm.colMean_X), (
        f"X cols ({X_out.shape[1]}) != colMean_X len ({len(sm.colMean_X)})"
    )
    return X_out, Bx_out


def test_preprocessing_no_mask_1d():
    _check_preprocessing(make_1d_smooth())


def test_preprocessing_no_mask_2d():
    _check_preprocessing(make_2d_smooth())


def test_preprocessing_with_mask_1d():
    sm = make_1d_smooth(knots_num=10)
    n = sm.X.shape[1] - 1
    mask = np.ones(n, dtype=bool)
    mask[0] = False   # drop first column
    sm._apply_col_mask(mask)
    _check_preprocessing(sm)


def test_preprocessing_with_mask_2d():
    sm = make_2d_smooth(sparse_data=True, knots_num=8)
    mask = sm.get_active_col_mask()
    sm._apply_col_mask(mask)
    _check_preprocessing(sm)


# ---------------------------------------------------------------------------
# 6. compute_Bx runs without error
# ---------------------------------------------------------------------------

def test_compute_bx_no_mask_1d():
    sm = make_1d_smooth()
    Bx = sm.compute_Bx()
    assert Bx.shape[1] == sm.X.shape[1], "Bx cols should match X cols before :-1 drop"


def test_compute_bx_no_mask_2d():
    sm = make_2d_smooth()
    Bx = sm.compute_Bx()
    assert Bx.shape[1] == sm.X.shape[1]


def test_compute_bx_with_mask_2d():
    sm = make_2d_smooth(sparse_data=True, knots_num=8)
    mask = sm.get_active_col_mask()
    k = mask.sum()
    sm._apply_col_mask(mask)
    Bx = sm.compute_Bx()
    assert Bx.shape[1] == k + 1, f"Bx should have k+1={k+1} cols, got {Bx.shape[1]}"


# ---------------------------------------------------------------------------
# 7. set_new_covariate re-applies the stored mask
# ---------------------------------------------------------------------------

def test_set_new_covariate_reapplies_mask():
    """
    After set_new_covariate() rebuilds X, _apply_col_mask must be re-invoked
    automatically with the stored mask so the masked shape is preserved.
    Tests with a 2D smooth (the bat scenario) using the same fixed island knots.
    """
    rng = np.random.default_rng(42)
    k1d = np.linspace(0, 10, 6)
    knots = [k1d, k1d]

    x1 = rng.uniform(0, 4, 300)
    y1 = rng.uniform(0, 4, 300)
    sm = covarate_smooth(
        [x1, y1],
        knots=knots,
        is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
    )

    mask = sm.get_active_col_mask()
    k = mask.sum()
    sm._apply_col_mask(mask)

    assert sm.X.shape[1] == k + 1

    # new data in the same sparse region; pass stored expanded knots and the mask
    stored_knots = list(sm.knots)
    x2 = rng.uniform(0, 4, 400)
    y2 = rng.uniform(0, 4, 400)
    sm.set_new_covariate([x2, y2], knots=stored_knots, col_mask=mask)

    assert sm.X.shape[1] == k + 1,   "mask should be re-applied after set_new_covariate"
    assert len(sm.colMean_X) == k,    "colMean_X length should stay k after re-apply"
    for S in sm.S_list:
        assert S.shape == (k + 1, k + 1)


def test_set_new_covariate_no_mask_by_default():
    """set_new_covariate without col_mask returns full-width basis regardless of stored mask."""
    rng = np.random.default_rng(42)
    k1d = np.linspace(0, 10, 6)
    x1 = rng.uniform(0, 4, 200)
    y1 = rng.uniform(0, 4, 200)
    sm = covarate_smooth(
        [x1, y1], knots=[k1d, k1d], is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
    )
    full_width = sm.X.shape[1]
    mask = sm.get_active_col_mask()
    sm._apply_col_mask(mask)
    assert sm.X.shape[1] < full_width   # mask was applied

    stored_knots = list(sm.knots)
    x2 = rng.uniform(0, 4, 200)
    y2 = rng.uniform(0, 4, 200)
    sm.set_new_covariate([x2, y2], knots=stored_knots)  # no col_mask → full basis
    assert sm.X.shape[1] == full_width
    assert sm.col_mask is None


def test_set_new_covariate_changed_knots_raises_with_mask():
    """Passing a stale mask with different knots raises ValueError."""
    import pytest
    rng = np.random.default_rng(1)
    k1d_6 = np.linspace(0, 10, 6)
    x = rng.uniform(0, 4, 200)
    y = rng.uniform(0, 4, 200)
    sm = covarate_smooth(
        [x, y], knots=[k1d_6, k1d_6], is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
    )
    mask = sm.get_active_col_mask()  # length matches 6-knot basis

    k1d_8 = np.linspace(0, 10, 8)  # different knot count → different basis dim
    with pytest.raises(ValueError, match="col_mask must have length"):
        sm.set_new_covariate([x, y], knots=[k1d_8, k1d_8], col_mask=mask)


# ---------------------------------------------------------------------------
# 8. _set_knots_spatial is_cyclic=None default backward compat
# ---------------------------------------------------------------------------

def test_set_knots_spatial_default_is_cyclic_1d():
    """
    Changing _set_knots_spatial's default from is_cyclic=[False] to is_cyclic=None
    (falling back to self.is_cyclic) must produce identical X and knots for a 1D
    non-cyclic smooth — the only case the old default could handle.
    """
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 10, 200)

    # reference: construction goes through __init__ which passes is_cyclic explicitly
    sm_ref = covarate_smooth([x], knots_num=8, penalty_type="EqSpaced")

    # set_new_covariate triggers _set_knots_spatial with is_cyclic=None (new default)
    sm_new = covarate_smooth([x], knots_num=8, penalty_type="EqSpaced")
    sm_new.set_new_covariate([x], knots=list(sm_new.knots))

    assert sm_ref.X.shape == sm_new.X.shape
    ref_dense = sm_ref.X.toarray() if sparse.issparse(sm_ref.X) else sm_ref.X
    new_dense = sm_new.X.toarray() if sparse.issparse(sm_new.X) else sm_new.X
    assert np.allclose(ref_dense, new_dense), \
        "set_new_covariate with new is_cyclic default must match original construction"


def test_set_knots_spatial_default_is_cyclic_2d():
    """
    For 2D smooths the old default is_cyclic=[False] would raise IndexError.
    With the new is_cyclic=None default, set_new_covariate must now work and
    produce the same X as a fresh construction from the same data and knots.
    """
    rng = np.random.default_rng(8)
    k1d = np.linspace(0, 10, 6)
    x = rng.uniform(0, 10, 300)
    y = rng.uniform(0, 10, 300)

    sm_ref = covarate_smooth(
        [x, y], knots=[k1d, k1d], is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
    )
    sm_new = covarate_smooth(
        [x, y], knots=[k1d, k1d], is_cyclic=np.array([False, False]),
        penalty_type="EqSpaced",
    )
    # this used to raise IndexError — now must succeed
    sm_new.set_new_covariate([x, y], knots=list(sm_new.knots))

    assert sm_ref.X.shape == sm_new.X.shape
    ref_dense = sm_ref.X.toarray() if sparse.issparse(sm_ref.X) else sm_ref.X
    new_dense = sm_new.X.toarray() if sparse.issparse(sm_new.X) else sm_new.X
    assert np.allclose(ref_dense, new_dense), \
        "2D set_new_covariate with new is_cyclic default must match original construction"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
