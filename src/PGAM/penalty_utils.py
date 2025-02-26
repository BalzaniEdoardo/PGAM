from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from nemos.tree_utils import pytree_map_and_reduce
from scipy import sparse

import math

from .basis import GAMBSplineEval
from .basis._basis import GAMAdditiveBasis, GAMMultiplicativeBasis


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


def compute_energy_penalty(n_samples: int, basis_derivative: Callable):
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
    energy_pen:
        Energy penalty matrix of shape (K, K), where K is the number of basis functions.
    """
    samples = jnp.linspace(0, 1, n_samples)
    eval_bas = jnp.asarray(basis_derivative(samples))
    indices = jnp.triu_indices(eval_bas.shape[1])
    square_bas = eval_bas[:, indices[0]] * eval_bas[:, indices[1]]
    dx = samples[1] - samples[0]
    # Simpson integration of squared basis.
    integr = vmap_simpson_regular(dx, square_bas)
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


def ndim_tensor_product_basis_penalty(*penalty: jnp.ndarray) -> jnp.ndarray:
    r"""
    Create a n-dimensional smoothing penalty matrix.

    Computes a smoothing penalty matrix for n-dimensional tensor product basis as in [1]_.

    Parameters
    ----------
    penalty:
        The smoothing penalty square matrices for each coordinate.
        Usually `penalty[i]` hsd the form
        :math:`\int (\mathbf{b_i}'' \cdot \mathbf{b_i} ^{\top} '' (x_1,...,x_n))^2 dx`,
        and :math:`\mathbf{b}_i` is the vector of basis function for the i-th coordinate.

    Returns
    -------
    ndim_penalty_tensor:
        A (num_dimensions, prod_penalty_shape, prod_penalty_shape) containing a penalty
        matrix that penalize wigglieness in each dimension. prod_penalty_shape is the
        product of the dimension of the squre penalty matrices.

    References
    ----------
    .. [1] Eilers, P. H. C. and B. D. Marx (2003). Multivariate calibration with temperature
    interaction using two-dimensional penalized signal regression. Chemometrics
    and Intelligent Laboratory Systems 66, 159–174.

    """
    num_dimensions = len(penalty)
    ndim_penalties = []
    penalty = tuple(penalty)
    # from the identities for the kron prod
    identities = tuple(sparse.identity(pen.shape[0], format='csr') for pen in penalty)

    # initialize the output tensor
    final_dim = math.prod(p.shape[0] for p in penalty)
    ndim_penalty_tensor = np.zeros((num_dimensions, final_dim, final_dim))

    # apply the sparse kron between each element
    for k, pen in enumerate(penalty):
        out = pen if k == 0 else identities[0]
        for j in  range(1, num_dimensions):
            out = sparse.kron(out, pen) if j == k else sparse.kron(out, identities[j])
        ndim_penalty_tensor[k] = out.toarray() if hasattr(out, 'toarray') else out
    return ndim_penalty_tensor


def compute_energy_penalty_tensor_additive_component(
        basis_component: GAMBSplineEval | GAMMultiplicativeBasis,
        n_sample: int = 10**4,
        penalize_null_space: bool = True,
) -> jnp.ndarray:
    """
    Define a penalty tensor for an additive component.

    Parameters
    ----------
    basis_component:
        Additive component of a basis.
    n_sample:
        Number of samples for the numerical approximation of the integral.
    penalize_null_space:
        Boolean, if true penalize the null space of every energy penalty component.


    Returns
    -------

    Notes
    -----
    For second derivative based penalties:
    - For 1-dimensional predictors, it adds penalties to straight lines (degree 1 polynomials ..math:`a + b \cdot x`).
    - For 2-dimensional predictors, it adds a penalty to ..math:`a + b \cdot x + c \cdot y + d \cdot xy`.

    """
    one_dim_pen = (compute_energy_penalty(n_sample, b.derivative) for b in basis_component._iterate_over_components())
    out = ndim_tensor_product_basis_penalty(*one_dim_pen)
    if penalize_null_space:
        null_pen = (compute_penalty_null_space(p) for p in out)
        full_rank = (p[None] if ~np.all(p == 0) else jnp.zeros((0, *p.shape)) for p in null_pen)
        out = jnp.concatenate(
            (out, *full_rank),
            axis=0
        )
    return out


def compute_energy_penalty_tensor(
        basis: GAMBSplineEval | GAMMultiplicativeBasis | GAMAdditiveBasis,
        n_sample: int = 10**4,
        penalize_null_space: bool = True,
        apply_identifiability: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
) -> list[jnp.ndarray]:
    """
    Create an energy penalty for each additive component.

    Parameters
    ----------
    basis:
        A BSpline basis or a composition of BSplines.
    n_sample:
        Number of samples for the numerical approximation of the integral.
    penalize_null_space:
        Boolean, if true penalize the null space of every energy penalty component.
    apply_identifiability:
        Callable that may set an identifiability constraint to the penalty tensor. Default
        is no constraint. Other classical options include could be dropping a column,
        or dropping a column and mean center (most common default option in GAM implementations,
        see R mgcv package).

    Returns
    -------
        A list with the penalty tensors for each component.

    """
    return [
        apply_identifiability(
            compute_energy_penalty_tensor_additive_component(bas, n_sample, penalize_null_space=penalize_null_space)
        )
        for bas in basis
    ]



def irregularly_sampled_simps(x, y):
    dx = jnp.diff(x)
    # compute scaling
    h0_over_h1 = dx[:-1:2] / dx[1::2]
    h0_plus_h1 = dx[:-1:2] + dx[1::2]
    glob_scale = h0_plus_h1 / 6
    first_scale = 2 - 1 / h0_over_h1
    second_scale = h0_plus_h1**2 / (dx[:-1:2] * dx[1::2])
    third_scale = 2 - h0_over_h1
    # compute simpson 1/3 formula
    even_tot = jnp.sum(glob_scale * (
            first_scale * y[:len(y)-2:2] + second_scale * y[1:len(y)-1:2] + third_scale * y[2::2]
        )
    )

    def add_correction(out, dx, y):
        len_y = len(y) - 1
        h0, h1 = dx[-2], dx[-1]
        return (
                out + y[len_y] * (2 * h1 ** 2 + 3 * h0 * h1) / (6 * (h0 + h1)) +
                y[len_y - 1] * (h1 ** 2 + 3 * h1 * h0) / (6 * h0) -
                y[len_y - 2] * h1 ** 3 / (6 * h0 * (h0 + h1))
        )
    return jax.lax.cond(len(y) % 2 == 0, add_correction, lambda *x: x[0], even_tot, dx, y)


def regularly_sampled_simps(dx, y):
    # compute scaling
    glob_scale = dx / 3.
    second_scale = 4
    # compute simpson 1/3 formula
    even_tot = jnp.sum(glob_scale * (
            y[:len(y)-2:2] + second_scale * y[1:len(y)-1:2] + y[2::2]
        )
    )

    def add_correction(out, dx, y):
        len_y = len(y) - 1
        return (
                out + y[len_y] * (5 * dx) / 12 +
                y[len_y - 1] * (2 * dx) / 3 -
                y[len_y - 2] * dx / 12
        )
    return jax.lax.cond(len(y) % 2 == 0, add_correction, lambda *x: x[0], even_tot, dx, y)

_vec_irregularly_sampled_simps = jax.vmap(irregularly_sampled_simps, in_axes=(None, 1))
_vec_regularly_sampled_simps = jax.vmap(regularly_sampled_simps, in_axes=(None, 1))


def vmap_simpson_regular(dx, y):
    shape = y.shape[1:]
    return _vec_regularly_sampled_simps(dx, y.reshape(y.shape[0], -1)).reshape(shape)

