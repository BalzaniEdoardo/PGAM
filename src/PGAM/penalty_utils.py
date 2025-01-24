from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from nemos.tree_utils import pytree_map_and_reduce

from nemos.basis._spline_basis import bspline

@jax.jit
def symmetric_sqrt(M):
    """Compute the square root of a symmetric matrix truncating eigs at float 32 precision."""
    eig, U = jnp.linalg.eigh(M)
    sort_col = jnp.argsort(eig)
    eig = eig[sort_col]
    U = U[:, sort_col]

    # matrix is sym should be positive
    # numerical error can make some value small and negative, so pass through an abs.
    eig = jnp.abs(eig)

    # crop the eig that are small relative to max
    eig = eig * (eig > jnp.finfo(jnp.float32).eps * eig.max())
    # compute the sqrt
    Bx = U * jnp.sqrt(eig)
    return Bx[:, ::-1].T


def compute_energy_penalty(n_samples, knots, order):
    if order < 3:
        raise ValueError("A second derivative based penalty can be computed for `order >= 3`. "
                         f"The provided order is {order} instead!")
    samples = np.linspace(0, 1, n_samples)
    eval_bas = bspline(samples, knots, order, der=2, outer_ok=False)
    indices = jnp.triu_indices(eval_bas.shape[1])
    square_bas = eval_bas[:, indices[0]] * eval_bas[:, indices[1]]
    dx = samples[1] - samples[0]
    integr = jax.scipy.integrate.trapezoid(square_bas, dx=dx, axis=0)
    energy_pen = jnp.zeros((eval_bas.shape[1], eval_bas.shape[1]))
    energy_pen = energy_pen.at[indices].set(integr)
    energy_pen = energy_pen + energy_pen.T - jnp.diag(energy_pen.diagonal())
    return energy_pen


def compute_weighted_penalty(penalty_tensor: jnp.ndarray, reg_strength: jnp.ndarray, index_map: jnp.ndarray, positive_mon_func=jnp.exp):
    """
    Compute a weighted sum of the penalties.

    Parameters
    ----------
    penalty_tensor:
        A tensor of shape (N, K, K), where K is the number of coefficients, N is the number of
        different penalty matrix available. If two variables have the same num of coefficients,
        then the derivative penalty witll be the same so we don't need to double count.
        penalty_tensor[i] contains a single block on the diagonal with the i-th penalization
    reg_strength:
        Vector of dimension (M,) with M >= N the number of variable we are regressing.
    index_map:
        Vector of integers of length M, with values from 0 to N-1. This vector is used to
        map the penalty tensor to the variable, so that penalty_tensor[index_map[i]] is
        the penalization term corresponding to the i-th regularization strength, i.e. the i-th variable.
    positive_mon_func:
        Function that makes the weights positives.

    Returns
    -------
    :
        The weighted penalty.

    """
    pos_reg = positive_mon_func(reg_strength)
    return jnp.sum(penalty_tensor[index_map] * pos_reg[:, None, None], axis=1)


def create_block_penalty(full_penalty: jnp.ndarray, start_idx: int, num_weights: int):
    block_size = full_penalty.shape[0]
    block_penalty = jnp.zeros((num_weights, num_weights)).at[
                    start_idx: start_idx+block_size, start_idx: start_idx+block_size
                    ].set(full_penalty)
    return block_penalty


def tree_create_block(tree_penalty_blocks, start_idx, block_matrix_n_rows: int):
    """
    Create the block penalty.

    Put the full penalties in the correct blocks and stack them in a tensor.
    This function should create the `penalty_tensor` that is used in
    `compute_weighted_penalty`.
    Can be pre-computed.


    Parameters
    ----------
    tree_penalty_blocks::
        A tree with the full penalties that needs to be inserted in the block diagonal.
    start_idx:
        Indices of the start of the block.
    block_matrix_n_rows:
        Number of rows and cols of the final penalty matrix.

    Returns
    -------
        An (num leaves, block_matrix_n_rows, block_matrix_n_rows) tensor with the full penalties
        in the right block. Can be precomputed.

    """
    tree_penalty_blocks = jax.tree_util.tree_leaves(tree_penalty_blocks)
    start_idx = jax.tree_util.tree_leaves(start_idx)

    return jnp.concatenate(
        jax.tree_util.tree_map(
            lambda x,y: create_block_penalty(x, y, block_matrix_n_rows)[None],
            tree_penalty_blocks,
            start_idx
        ),
        axis=0
    )
