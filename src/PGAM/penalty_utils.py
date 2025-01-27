from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from nemos.tree_utils import pytree_map_and_reduce


@jax.jit
def symmetric_sqrt(symmetric_matrix):
    """
    Compute the square root of a symmetric matrix, truncating eigenvalues at float32 precision.

    Parameters
    ----------
    symmetric_matrix :
        A symmetric matrix of shape (N, N).

    Returns
    -------
    :
        The square root of the input matrix.
    """
    eig, U = jnp.linalg.eigh(symmetric_matrix)
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


def compute_start_block(tree_penalty: Any, shift_by=0):
    """
    Compute the starting block index for penalties.

    Parameters
    ----------
    tree_penalty :
        Tree containing penalty matrices.
    shift_by : int, optional
        Initial index to shift by, default is 0.

    Returns
    -------
    :
        A tree containing the cumulative sum of block indices.
    """
    flat, struct = jax.tree_util.tree_flatten(tree_penalty)
    vals = (shift_by, *(arr.shape[0] for arr in flat[:-1]))

    def cum_sum(val_iter):
        v_prev = 0
        for v_curr in val_iter:
            yield v_curr + v_prev
            v_prev = v_curr + v_prev

    return jax.tree_util.tree_unflatten(struct, [k for k in cum_sum(vals)])


def tree_compute_sqrt_penalty(tree_penalty, reg_strength: jnp.ndarray, index_map: jnp.ndarray, shift_by: Optional[int]=0, positive_mon_func: Callable=jnp.exp):
    """
    Compute the square root of penalties in a pytree and apply weighting.

    Parameters
    ----------
    tree_penalty :
        Tree containing penalty matrices.
    reg_strength :
        Regularization strengths of shape (M,).
    index_map :
        Index mapping of length M.
    shift_by :
        Initial index to shift by, default is 0.
    positive_mon_func :
        Monotonic function to ensure positive weights, default is `jnp.exp`.

    Returns
    -------
    :
        Weighted penalty matrix.
    """
    sqrt_tree = jax.tree_util.tree_map(symmetric_sqrt, tree_penalty)
    tree_start = compute_start_block(sqrt_tree, shift_by=shift_by)
    mx_size = pytree_map_and_reduce(lambda x: x.shape[0], sum, sqrt_tree)
    return compute_weighted_penalty(
        tree_create_block(sqrt_tree, tree_start, mx_size),
        reg_strength,
        jnp.hstack(jax.tree_util.tree_leaves(index_map), dtype=int),
        positive_mon_func=positive_mon_func
    )


def compute_energy_penalty(n_samples, basis_derivative: Callable):
    """
    Compute the energy penalty for a basis derivative.

    Parameters
    ----------
    n_samples :
        Number of samples for integration.
    basis_derivative :
        Function that computes the derivative of the basis.

    Returns
    -------
    j:
        Energy penalty matrix of shape (K, K), where K is the number of basis functions.
    """
    samples = np.linspace(0, 1, n_samples)
    eval_bas = basis_derivative(samples)
    indices = jnp.triu_indices(eval_bas.shape[1])
    square_bas = eval_bas[:, indices[0]] * eval_bas[:, indices[1]]
    dx = samples[1] - samples[0]
    # write my own implementation of simpson for numerical accuracy
    integr = jax.scipy.integrate.trapezoid(square_bas, dx=dx, axis=0)
    energy_pen = jnp.zeros((eval_bas.shape[1], eval_bas.shape[1]))
    energy_pen = energy_pen.at[indices].set(integr)
    energy_pen = energy_pen + energy_pen.T - jnp.diag(energy_pen.diagonal())
    return energy_pen


def compute_penalty_null_space(penalty):
    """
    Compute the null space projection of a penalty matrix.

    Parameters
    ----------
    penalty :
        Penalty matrix of shape (K, K).

    Returns
    -------
    :
        Null space projection matrix of shape (K, K).
    """
    eig, U = jnp.linalg.eigh(penalty)
    zero_idx = jnp.abs(eig) < jnp.finfo(float).eps * jnp.max(eig)
    U = U[:, zero_idx]
    return jnp.dot(U, U.T)


def compute_weighted_penalty(penalty_tensor: jnp.ndarray, reg_strength: jnp.ndarray, index_map: jnp.ndarray, positive_mon_func=jnp.exp):
    """
    Compute a weighted sum of the penalties.

    Parameters
    ----------
    penalty_tensor :
        Tensor of shape (N, K, K), where K is the number of coefficients and N is the number of penalty matrices.
    reg_strength :
        Regularization strengths of shape (M,), where M >= N.
    index_map :
        Mapping of indices from (0, N-1) of shape (M,).
    positive_mon_func :
        Function that ensures positive weights, default is `jnp.exp`.

    Returns
    -------
    :
        Weighted penalty matrix of shape (K, K).
    """
    pos_reg = positive_mon_func(reg_strength)
    return jnp.sum(penalty_tensor[index_map] * pos_reg[:, None, None], axis=0)


def create_block_penalty(full_penalty: jnp.ndarray, start_idx: int, num_weights: int):
    """
    Create a block penalty matrix.

    Parameters
    ----------
    full_penalty :
        Penalty matrix to insert in the block.
    start_idx :
        Start index of the block.
    num_weights :
        Total number of weights.

    Returns
    -------
    :
        Block penalty matrix of shape (num_weights, num_weights).
    """
    block_size = full_penalty.shape[0]
    block_penalty = jnp.zeros((num_weights, num_weights)).at[
                    start_idx: start_idx+block_size, start_idx: start_idx+block_size
                    ].set(full_penalty)
    return block_penalty


def tree_create_block(tree_penalty_blocks, start_idx, block_size: int):
    """
    Create a block penalty tensor from a tree of penalty blocks.

    Parameters
    ----------
    tree_penalty_blocks :
        Tree containing penalty blocks.
    start_idx :
        Tree containing start indices for each block.
    block_size :
        Number of rows/columns of the final block matrix.

    Returns
    -------
    :
        Block penalty tensor of shape (num_blocks, block_matrix_n_rows, block_matrix_n_rows).
    """
    tree_penalty_blocks = jax.tree_util.tree_leaves(tree_penalty_blocks)
    start_idx = jax.tree_util.tree_leaves(start_idx)

    return jnp.concatenate(
        jax.tree_util.tree_map(
            lambda x,y: create_block_penalty(x, y, block_size)[None],
            tree_penalty_blocks,
            start_idx
        ),
        axis=0
    )
