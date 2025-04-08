import pytest
import json
import pathlib
import numpy as np
from PGAM import penalty_utils
import nemos as nmo
import jax
from jax.tree_util import tree_map,  treedef_is_leaf, tree_structure

from PGAM.basis import GAMBSplineEval


@pytest.fixture()
def _tree_map_list_to_array():

    is_leaf = lambda x: treedef_is_leaf(tree_structure(x)) or isinstance(x, list)
    def map_list_to_array(params):
        return tree_map(lambda x: x if not isinstance(x, list) else np.asarray(x), params, is_leaf=is_leaf)
    return map_list_to_array


def test_one_dim_bspline_der_2_energy_penalty(_tree_map_list_to_array):
    """Check that the full penalty matches the original PGAM implementation."""
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    basis_parms = params["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_parms["knots"], basis_parms["order"], der=basis_parms["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(params["n_samples"], der_basis)
    assert np.allclose(pen, params["energy_penalty"])


def test_one_dim_bspline_der_2_null_space_penalty(_tree_map_list_to_array):
    """Check that the full penalty matches the original PGAM implementation."""
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    basis_params = params["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_params["knots"], basis_params["order"], der=basis_params["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(params["n_samples"], der_basis)
    null_pen = penalty_utils.compute_penalty_null_space(pen)
    assert np.allclose(null_pen, params["null_space_penalty"])


def test_one_dim_bspline_der_2_symmetric_sqrt(_tree_map_list_to_array):
    """Check that the full penalty matches the original PGAM implementation."""
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    sqrt_pen = penalty_utils.symmetric_sqrt(params["energy_penalty"])
    assert np.allclose(sqrt_pen, params["sqrt_energy_penalty"])
    log_lam = params["reg_strength"][0]
    scaled_pen = penalty_utils.tree_compute_sqrt_penalty([params["energy_penalty"]], np.array([log_lam]), np.array([0]))
    assert np.allclose(scaled_pen, np.exp(log_lam) * params["sqrt_energy_penalty"])


def test_one_dim_bspline_der_2_penalty_tensor(_tree_map_list_to_array):
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    bspline_params = params["bspline_params"]
    n_basis = bspline_params["knots"].shape[0] - bspline_params["order"]
    bas = GAMBSplineEval(n_basis, order=bspline_params["order"], identifiability=False)
    pen_tensor = penalty_utils.compute_energy_penalty_tensor_additive_component(bas)
    assert np.allclose(pen_tensor[0], params["energy_penalty"])
    assert np.allclose(pen_tensor[1], params["null_space_penalty"])


def test_one_dim_bspline_der_2_agumented(_tree_map_list_to_array):
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "one_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    bspline_params = params["bspline_params"]
    n_basis = bspline_params["knots"].shape[0] - bspline_params["order"]
    bas = GAMBSplineEval(n_basis, order=bspline_params["order"], identifiability=False)
    pen_list = penalty_utils.compute_energy_penalty_tensor(bas)
    out = penalty_utils.tree_compute_sqrt_penalty(pen_list, [jax.numpy.log(params["reg_strength"])])
    # the first col of agumented pen in original gam was a column of 0s
    # since the intercept term was treated as a column of 1s in
    # the design matrix and was not penalized.
    # secondly, the original PGAM code had a try/except which would try
    # an unsafe Cholesky decomposition, if failed, used the safe eig
    # truncation method to get a square root of a matrix that is implemented
    # here. I.e. in order to compare we need to check the square of the matrix
    orig_agu_pen = params["agumented_penalty"][:, 1:]
    orig_agu_pen_square = orig_agu_pen.T.dot(orig_agu_pen)
    assert np.allclose(out.T.dot(out), orig_agu_pen_square)


def test_two_dim_bspline_der_2_energy_penalty(_tree_map_list_to_array):
    """Check that the full penalty matches the original PGAM implementation."""
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "two_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    basis_parms = params["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_parms["knots"], basis_parms["order"], der=basis_parms["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(params["n_samples"], der_basis)
    pen = penalty_utils.ndim_tensor_product_basis_penalty(pen, pen)
    assert np.allclose(pen[0], params["energy_penalty_0"])
    assert np.allclose(pen[1], params["energy_penalty_1"])


def test_two_dim_bspline_der_2_null_space_penalty(_tree_map_list_to_array):
    """Check that the full penalty matches the original PGAM implementation."""
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "two_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)
    basis_params = params["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_params["knots"], basis_params["order"], der=basis_params["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(params["n_samples"], der_basis)
    pen = penalty_utils.ndim_tensor_product_basis_penalty(pen, pen)
    null_pen = penalty_utils.compute_penalty_null_space(pen.mean(axis=0))
    # due to poor conditioning of the penalty and different LAPACK versions
    # used by numpy and jaxlib, cannot compare directly the null-space penalty match.
    # Instead, check that the null_pen is orthogonal to the energy penalty (which is all
    # we care about).
    assert np.allclose(np.dot(pen, null_pen), 0)
    assert np.allclose(np.dot(params["energy_penalty_0"], null_pen), 0)
    assert np.allclose(np.dot(params["energy_penalty_1"], null_pen), 0)


def test_two_dim_bspline_der_2_symmetric_sqrt(_tree_map_list_to_array):
    """Check that the full penalty matches the original PGAM implementation."""
    jax.config.update('jax_enable_x64', True)
    script_dir = pathlib.Path(__file__).resolve().parent / "data"
    with open(script_dir / "two_dim_bspline_penalty.json", "r", encoding="utf-8") as f:
        params = json.load(f)
        params = _tree_map_list_to_array(params)

    basis_params = params["bspline_params"]
    der_basis = lambda x : nmo.basis._spline_basis.bspline(x, basis_params["knots"], basis_params["order"], der=basis_params["der"], outer_ok=False)
    pen = penalty_utils.compute_energy_penalty(params["n_samples"], der_basis)
    pen = penalty_utils.ndim_tensor_product_basis_penalty(pen, pen)
    null_pen = penalty_utils.compute_penalty_null_space(pen.mean(axis=0))
    sqrt_orig = params["sqrt_energy_penalty"]
    scaled_pen = penalty_utils.tree_compute_sqrt_penalty([pen[0], pen[1], null_pen], np.array([log_lam]), np.array([0]))
    # assert np.allclose(scaled_pen, np.exp(log_lam) * params["sqrt_energy_penalty"])
