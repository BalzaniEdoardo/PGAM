import pytest
import json
import pathlib
import numpy as np
from PGAM import penalty_utils
import nemos as nmo
import jax
from jax.tree_util import tree_map,  treedef_is_leaf, tree_structure


@pytest.fixture()
def _tree_map_list_to_array():

    is_leaf = lambda x: treedef_is_leaf(tree_structure(x)) or isinstance(x, list)
    def map_list_to_array(params):
        return tree_map(lambda x: x if not isinstance(x, list) else np.asarray(x), params, is_leaf=is_leaf)
    return map_list_to_array


def test_one_dim_bspline_der_2_full_penalty(_tree_map_list_to_array):
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