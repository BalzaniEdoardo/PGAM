import pytest
import json
import pathlib
import numpy as np
from PGAM import penalty_utils
import nemos as nmo
import jax
from jax.tree_util import tree_map,  treedef_is_leaf, tree_structure

from PGAM.basis import GAMBSplineEval
from PGAM.config import set_debug


@pytest.fixture()
def script_dir():
    return pathlib.Path(__file__).resolve().parent / "data"

@pytest.fixture()
def _tree_map_list_to_array():

    is_leaf = lambda x: treedef_is_leaf(tree_structure(x)) or isinstance(x, list)
    def map_list_to_array(params):
        if "reg_strength" in params:
            rs = params.pop("reg_strength")
        try:
            reg_strength = {"reg_strength": jax.numpy.array(rs)}
        except:
            reg_strength = {"reg_strength": [np.array(r) for r in rs]}
        params = tree_map(lambda x: x if not isinstance(x, list) else np.asarray(x), params, is_leaf=is_leaf)
        params.update(reg_strength)
        return params
    return map_list_to_array

@pytest.fixture()
def one_dim_bspline_penalty(_tree_map_list_to_array, script_dir):
    with open(script_dir / "one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    return params

@pytest.fixture()
def two_dim_bspline_penalty(_tree_map_list_to_array, script_dir):
    with open(script_dir / "two_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    return params

@pytest.fixture()
def sum_two_one_dim_bspline_penalty(_tree_map_list_to_array, script_dir):
    with open(script_dir / "sum_of_two_one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    return params


@pytest.fixture()
def sum_one_dim_two_dim_bspline_penalty(_tree_map_list_to_array, script_dir):
    with open(script_dir / "sum_of_one_dim_two_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    return params


def test_one_dim_bspline_der_2_energy_penalty(one_dim_bspline_penalty):
    """Check that the full penalty matches the original PGAM implementation."""
    basis_parms = one_dim_bspline_penalty["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_parms["knots"], basis_parms["order"], der=basis_parms["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(one_dim_bspline_penalty["n_samples"], der_basis)
    assert np.allclose(pen, one_dim_bspline_penalty["energy_penalty"])


def test_one_dim_bspline_der_2_null_space_penalty(one_dim_bspline_penalty):
    """Check that the full penalty matches the original PGAM implementation."""
    basis_params = one_dim_bspline_penalty["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_params["knots"], basis_params["order"], der=basis_params["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(one_dim_bspline_penalty["n_samples"], der_basis)
    null_pen = penalty_utils.compute_penalty_null_space(pen)
    assert np.allclose(null_pen, one_dim_bspline_penalty["null_space_penalty"])


def test_one_dim_bspline_der_2_symmetric_sqrt(one_dim_bspline_penalty):
    """Check that the full penalty matches the original PGAM implementation."""
    sqrt_pen = penalty_utils.symmetric_sqrt(one_dim_bspline_penalty["energy_penalty"])
    assert np.allclose(sqrt_pen, one_dim_bspline_penalty["sqrt_energy_penalty"])
    log_lam = np.log(one_dim_bspline_penalty["reg_strength"][0])
    scaled_pen = penalty_utils.tree_compute_sqrt_penalty([one_dim_bspline_penalty["energy_penalty"]], [np.array([log_lam])], 0, apply_identifiability=lambda x:x)
    assert np.allclose(scaled_pen, np.sqrt(np.exp(log_lam)) * one_dim_bspline_penalty["sqrt_energy_penalty"])


def test_one_dim_bspline_der_2_penalty_tensor(one_dim_bspline_penalty):
    bspline_params = one_dim_bspline_penalty["bspline_params"]
    n_basis = bspline_params["knots"].shape[0] - bspline_params["order"]
    bas = GAMBSplineEval(n_basis, order=bspline_params["order"], identifiability=False)
    pen_tensor = penalty_utils.compute_energy_penalty_tensor_additive_component(bas)
    assert np.allclose(pen_tensor[0], one_dim_bspline_penalty["energy_penalty"])
    assert np.allclose(pen_tensor[1], one_dim_bspline_penalty["null_space_penalty"])


def test_one_dim_bspline_der_2_agumented(one_dim_bspline_penalty):
    bspline_params = one_dim_bspline_penalty["bspline_params"]
    n_basis = bspline_params["knots"].shape[0] - bspline_params["order"]
    bas = GAMBSplineEval(n_basis, order=bspline_params["order"], identifiability=False)
    pen_list = penalty_utils.compute_energy_penalty_tensor(bas)

    # the first col of agumented pen in original gam was a column of 0s
    # since the intercept term was treated as a column of 1s in
    # the design matrix and was not penalized.
    # secondly, the original PGAM code had a try/except which would try
    # an unsafe Cholesky decomposition, if failed, used the safe eig
    # truncation method to get a square root of a matrix that is implemented
    # here. I.e. in order to compare we need to check the square of the matrix
    with set_debug(True):
        # this context sets the original Cholesky sqrt algorithm
        # the equivalence of the sqrt is checked with: test_orig_vs_new_sqrt
        out = penalty_utils.tree_compute_sqrt_penalty(
            pen_list,
            [jax.numpy.log(jax.numpy.asarray(one_dim_bspline_penalty["reg_strength"]))]
        )
    orig_agu_pen = one_dim_bspline_penalty["agumented_penalty"][:, 1:]
    assert np.allclose(out, orig_agu_pen)


def test_two_dim_bspline_der_2_energy_penalty(two_dim_bspline_penalty):
    """Check that the full penalty matches the original PGAM implementation."""
    basis_parms = two_dim_bspline_penalty["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_parms["knots"], basis_parms["order"], der=basis_parms["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(two_dim_bspline_penalty["n_samples"], der_basis)
    pen = penalty_utils.ndim_tensor_product_basis_penalty(pen, pen)
    assert np.allclose(pen[0], two_dim_bspline_penalty["energy_penalty_0"])
    assert np.allclose(pen[1], two_dim_bspline_penalty["energy_penalty_1"])


def test_two_dim_bspline_der_2_null_space_penalty(two_dim_bspline_penalty):
    """Check that the full penalty matches the original PGAM implementation."""
    basis_params = two_dim_bspline_penalty["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_params["knots"], basis_params["order"], der=basis_params["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(two_dim_bspline_penalty["n_samples"], der_basis)
    pen = penalty_utils.ndim_tensor_product_basis_penalty(pen, pen)
    null_pen = penalty_utils.compute_penalty_null_space(pen.mean(axis=0))
    # due to poor conditioning of the penalty and different LAPACK versions
    # used by numpy and jaxlib, cannot compare directly the null-space penalty match.
    # Instead, check that the null_pen is orthogonal to the energy penalty (which is all
    # we care about).
    assert np.allclose(np.dot(pen, null_pen), 0)
    assert np.allclose(np.dot(two_dim_bspline_penalty["energy_penalty_0"], null_pen), 0)
    assert np.allclose(np.dot(two_dim_bspline_penalty["energy_penalty_1"], null_pen), 0)


def test_two_dim_bspline_der_2_symmetric_sqrt(two_dim_bspline_penalty):
    """Check that the full penalty matches the original PGAM implementation."""
    out = two_dim_bspline_penalty["penalties_for_compute_sqrt"]
    log_lam = np.log(two_dim_bspline_penalty["reg_strength"][0])
    sqrt_orig = two_dim_bspline_penalty["sqrt_energy_penalty"]
    scaled_sqrt_pen = penalty_utils.tree_compute_sqrt_penalty(out, np.array([log_lam, log_lam, log_lam]), 0, apply_identifiability=lambda x:x)
    squared_pen = scaled_sqrt_pen.T.dot(scaled_sqrt_pen)
    squared_pen_orig = sqrt_orig.T.dot(sqrt_orig)
    assert np.allclose(squared_pen_orig, squared_pen)


def test_two_dim_bspline_der_2_penalty_tensor(two_dim_bspline_penalty):
    bspline_params = two_dim_bspline_penalty["bspline_params"]
    n_basis = bspline_params["knots"].shape[0] - bspline_params["order"]
    bas = GAMBSplineEval(n_basis, order=bspline_params["order"], identifiability=False) ** 2
    pen_tensor = penalty_utils.compute_energy_penalty_tensor_additive_component(bas)
    s_list = np.concatenate(
        (
            [two_dim_bspline_penalty["energy_penalty_0"][None],
             two_dim_bspline_penalty["energy_penalty_1"][None]]
        ),
        axis=0
    )
    assert np.allclose(pen_tensor[:2], s_list)


def test_two_dim_bspline_der_2_agumented(two_dim_bspline_penalty):
    jax.config.update("jax_enable_x64", True)
    bspline_params = two_dim_bspline_penalty["bspline_params"]
    n_basis = bspline_params["knots"].shape[0] - bspline_params["order"]
    bas = GAMBSplineEval(n_basis, order=bspline_params["order"], identifiability=False) ** 2
    pen_list = penalty_utils.compute_energy_penalty_tensor(bas, n_sample=10**6)
    # the first col of agumented pen in original gam was a column of 0s
    # since the intercept term was treated as a column of 1s in
    # the design matrix and was not penalized.
    # secondly, the original PGAM code had a try/except which would try
    # an unsafe Cholesky decomposition, if failed, used the safe eig
    # truncation method to get a square root of a matrix that is implemented
    # here. I.e. in order to compare we need to check the square of the matrix
    with set_debug(True):
        # this context sets the original Cholesky sqrt algorithm
        # the equivalence of the sqrt is checked with: test_orig_vs_new_sqrt
        out = penalty_utils.tree_compute_sqrt_penalty(
            pen_list,
            [jax.numpy.log(two_dim_bspline_penalty["reg_strength"])]
        )
    orig_agu_pen = two_dim_bspline_penalty["agumented_penalty"][:, 1:]
    assert np.allclose(out, orig_agu_pen)

def test_orig_vs_new_sqrt():
    M = np.random.randn(10, 10)
    M = M.dot(M.T)
    # compare the sqrt with the original implementation
    with set_debug(True):
        sqrtM = penalty_utils.symmetric_sqrt(M)
    sqrtM_new = penalty_utils.symmetric_sqrt(M)
    assert isinstance(sqrtM, np.ndarray)
    assert isinstance(sqrtM_new, jax.numpy.ndarray)
    assert np.allclose(sqrtM.T.dot(sqrtM), sqrtM_new.T.dot(sqrtM_new))


def test_sum_two_dim_bspline_penalty_tensor(sum_two_one_dim_bspline_penalty):
    jax.config.update("jax_enable_x64", True)
    params1 = sum_two_one_dim_bspline_penalty["bspline_1_params"]
    params2 = sum_two_one_dim_bspline_penalty["bspline_2_params"]
    n_basis1 = params1["knots"].shape[0] - params1["order"]
    n_basis2 = params2["knots"].shape[0] - params2["order"]
    bas = GAMBSplineEval(n_basis1, identifiability=False) + GAMBSplineEval(n_basis2, identifiability=False)
    reg_strength = np.log(sum_two_one_dim_bspline_penalty["reg_strength"])
    with set_debug(True):
        # use Cholesky sqrt
        pen_new = penalty_utils.compute_penalty_agumented_from_basis(bas, list(reg_strength))
    pen_orig = sum_two_one_dim_bspline_penalty["block_penalty"]
    # remove first col of zeros from orig
    assert np.allclose(pen_new, pen_orig[:, 1:])


def test_sum_two_dim_bspline_penalty_tensor(sum_one_dim_two_dim_bspline_penalty):
    jax.config.update("jax_enable_x64", True)
    params1 = sum_one_dim_two_dim_bspline_penalty["bspline_1_params"]
    params2 = sum_one_dim_two_dim_bspline_penalty["bspline_2_params"]
    n_basis1 = params1["knots"].shape[0] - params1["order"]
    n_basis2 = params2["knots"].shape[0] - params2["order"]
    bas = GAMBSplineEval(n_basis1, identifiability=False) + GAMBSplineEval(n_basis2, identifiability=False)**2
    reg_strength = [np.log(r) for r in sum_one_dim_two_dim_bspline_penalty["reg_strength"]]
    with set_debug(True):
        # use Cholesky sqrt
        pen_new = penalty_utils.compute_penalty_agumented_from_basis(bas, list(reg_strength))
    pen_orig = sum_one_dim_two_dim_bspline_penalty["block_penalty"]
    # remove first col of zeros from orig
    assert np.allclose(pen_new, pen_orig[:, 1:])



# TODO:
# - create a json for an additive basis summing 2 2D basis
# - test that the block-diagonal agumented matrix matches original implementation in all cases
# - implement the agumentation of the design matrix:
#   - compute weights based on the link function and observation model
#   - scale by weight, should match model.wexog[:n_obs, :] (line 1042 of GAM_library.py)
#   - QR decompose the scaled design
# - implement the GCV computation taking as input the QR decomp, the list of penalty tensor etc.
# - test the GCV values against the original implementation
#