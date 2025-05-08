from functools import partial

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nemos.tree_utils import pytree_map_and_reduce
from scipy import sparse

import math

from .basis import GAMBSplineEval
from .basis._basis import GAMAdditiveBasis, GAMMultiplicativeBasis
from .config import config


def symmetric_sqrt(symmetric_matrix):
    if config.DEBUG:
        return _symmetric_sqrt_numpy(symmetric_matrix)
    else:
        return _symmetric_sqrt_jax(symmetric_matrix)


# original PGAM implementation (NeurIPS Balzani)
def _symmetric_sqrt_numpy(symmetric_matrix):
    print("Original implementation...")
    try:
        return np.linalg.cholesky(symmetric_matrix).T
    except np.linalg.LinAlgError:
        eig, U = np.linalg.eigh(symmetric_matrix)
        sort_col = np.argsort(eig)[::-1]
        eig = eig[sort_col]
        U = U[:, sort_col]
        # matrix is sym should be positive
        eig = np.abs(eig)
        i_rem = np.where(eig < 10 ** (-8) * eig.max())[0]
        eig = np.delete(eig, i_rem, 0)
        Bx = np.zeros(U.shape)
        mask = np.arange(U.shape[1])
        mask = mask[np.delete(mask, i_rem, 0)]
        Bx[:, mask] = np.delete(U, i_rem, 1) * np.sqrt(eig)
        Bx = Bx.T
        return Bx

@jax.jit
def _symmetric_sqrt_jax(symmetric_matrix):
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
    rows = (shift_by, *(arr.shape[0] for arr in flat[:-1]))
    cols = (shift_by, *(arr.shape[1] for arr in flat[:-1]))

    def cum_sum(val_iter):
        v_prev = 0
        for v_curr in val_iter:
            yield v_curr + v_prev
            v_prev = v_curr + v_prev
    idx_start_row = jax.tree_util.tree_unflatten(struct, [k for k in cum_sum(rows)])
    idx_start_col = jax.tree_util.tree_unflatten(struct, [k for k in cum_sum(cols)])
    return idx_start_row, idx_start_col


def tree_compute_sqrt_penalty(tree_penalty: Any, reg_strength: Any, shift_by: Optional[int]=0, positive_mon_func: Callable=jnp.exp, apply_identifiability: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x[...,:-1]):
    """
    Compute the square root of penalties in a pytree and apply weighting.

    Parameters
    ----------
    tree_penalty :
        Tree containing penalty matrices.
    reg_strength :
        Regularization strengths tree, same structure as tree_penalty each leave is of shale (m, ) if the
        corresponding leaf of tree_penalty is of shape (m, Ki, Ki).
    shift_by :
        Initial index to shift by, default is 0.
    positive_mon_func :
        Monotonic function to ensure positive weights, default is `jnp.exp`.
    apply_identifiability:
        A function that matches the identifiability constrain at the level of the penalty matrix.
        If for example, we dropped a b-spline element, i.e. dropped a column of the design matrix,
        we should drop the corresponding column of the penalty. Default assumes that we are dropping
        the last column of the design matrix.


    Returns
    -------
    :
        Weighted penalty matrix.
    """
    scaled_pen = jax.tree_util.tree_map(
        lambda pen, reg: compute_weighted_penalty(pen, reg, positive_mon_func=positive_mon_func),
        tree_penalty,
        reg_strength
    )
    sqrt_tree = jax.tree_util.tree_map(lambda x: apply_identifiability(symmetric_sqrt(x)), scaled_pen)
    tree_start_row, tree_start_col = compute_start_block(sqrt_tree, shift_by=shift_by)
    tot_shape = (
        pytree_map_and_reduce(lambda x: x.shape[0], sum, sqrt_tree),
        pytree_map_and_reduce(lambda x: x.shape[1], sum, sqrt_tree)
    )
    return tree_create_block(sqrt_tree, tree_start_row, tree_start_col, tot_shape)


@partial(jax.jit, static_argnums=(1, 2))
def compute_penalty_blocks(
        tree_penalty: Any,
        shift_by: Optional[int]=0,
        apply_identifiability: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x[...,:-1,:-1]
):
    """
    Compute the penalty blocks for a pytree and apply weighting.

    Parameters
    ----------
    tree_penalty:
        Tree containing penalty matrices.
    shift_by:
        Shift blocks by. For GCV compute the penalty is shifted by one block. The reason for that is
        that the design matrix must include the intercept, which is not penalized. No penalization
        corresponds to a 1x1 block of 0s.
    apply_identifiability:
        Function that applies identifiability constraint. Note that here we are not working with
        square roots, i.e. both rows and columns must be dropped when applied an identifiability
        constraint.

    Returns
    -------
        A tree with the individual basis penalties inserted in a block structure of size the overall
        block penalty matrix.

    Notes
    -----
        The output is very sparse, if JAX support for sparse representation improves, consider using
        a compressed representation for sparse matrices.

    """
    scaled_penalties = jax.tree_util.tree_map(apply_identifiability, tree_penalty)
    tree_start_row, tree_start_col = compute_start_block(scaled_penalties, shift_by=shift_by)

    # compute shape of blocks (individual penalty matrices)
    block_shapes = jax.tree_util.tree_map(lambda x: x.shape[1:], scaled_penalties)
    num_pen_per_block = jax.tree_util.tree_map(lambda x: x.shape[0], scaled_penalties)

    # compute size of the full block penalty and allocate the blocks
    size = 1 + sum(jax.tree_util.tree_leaves(block_shapes)[1::2])
    penalty_blocks = jax.tree_util.tree_map(lambda n: jnp.zeros((n, size, size)), num_pen_per_block)
    # function that build the blocks
    func = lambda pen, full, start_row, start_col: pen.at[:, start_row: start_row+size, start_col: start_col+size].set(full)
    return jax.tree_util.tree_map(func, penalty_blocks, scaled_penalties, tree_start_row, tree_start_col)


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
    if config.DEBUG:
        return compute_energy_penalty_numpy(n_samples, basis_derivative)
    else:
        return compute_energy_penalty_jax(n_samples, basis_derivative)


def compute_energy_penalty_jax(n_samples: int, basis_derivative: Callable):
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
    energy_pen = energy_pen + jnp.triu(energy_pen, 1).T
    return energy_pen


def compute_energy_penalty_numpy(n_samples: int, basis_derivative: Callable):
    from scipy.integrate import simpson
    samples = np.linspace(0, 1, n_samples)
    eval_bas = np.asarray(basis_derivative(samples))
    dx = samples[1] - samples[0]
    # Simpson integration of squared basis.
    integr = np.zeros((eval_bas.shape[1], eval_bas.shape[1]))
    for i in range(eval_bas.shape[1]):
        for j in range(i, eval_bas.shape[1]):
            integr[i, j] = simpson(eval_bas[:, i] * eval_bas[:, j], dx=dx)
    integr += np.triu(integr, 1).T
    return integr


def compute_penalty_null_space(penalty):
    if not config.DEBUG:
        return compute_penalty_null_space_jax(penalty)
    else:
        return compute_penalty_null_space_numpy(penalty)


def compute_penalty_null_space_numpy(penalty):
    """
    Compute the null space projection of a penalty matrix.

    Parameters
    ----------
    penalty :
        Penalty matrix of shape (m, K, K).

    Returns
    -------
    :
        Null space projection matrix of shape (K, K).
    """
    # original algorith summed (null-space should be the same)
    penalty = penalty.sum(axis=0)
    eig, U = np.linalg.eigh(penalty)
    zero_idx = np.abs(eig) < np.finfo(float).eps * np.max(eig)
    U = U[:, zero_idx]
    return np.dot(U, U.T)


def compute_penalty_null_space_jax(penalty):
    """
    Compute the null space projection of a penalty matrix.

    Parameters
    ----------
    penalty :
        Penalty matrix of shape (m, K, K).

    Returns
    -------
    :
        Null space projection matrix of shape (K, K).
    """
    penalty = penalty.mean(axis=0)
    eig, U = jnp.linalg.eigh(penalty)
    zero_idx = jnp.abs(eig) < jnp.finfo(float).eps * jnp.max(eig)
    U = U[:, zero_idx]
    return jnp.dot(U, U.T)


def compute_weighted_penalty(penalty_tensor: jnp.ndarray, reg_strength: jnp.ndarray, positive_mon_func=jnp.exp):
    """
    Compute a weighted sum of the penalties.

    Parameters
    ----------
    penalty_tensor :
        Tensor of shape (N, K, K), where K is the number of coefficients and N is the number of penalty matrices.
    reg_strength :
        Regularization strengths of shape (N,).
    positive_mon_func :
        Function that ensures positive weights, default is `jnp.exp`.

    Returns
    -------
    :
        Weighted penalty matrix of shape (K, K).
    """
    pos_reg = positive_mon_func(reg_strength)
    return jnp.sum(penalty_tensor * pos_reg[:, None, None], axis=0)


def create_block_penalty(full_penalty: jnp.ndarray, start_idx_row: int, start_idx_col: int, block_shape: Tuple[int, int]):
    """
    Create a block penalty matrix.

    Parameters
    ----------
    full_penalty :
        Penalty matrix to insert in the block.
    start_idx_row :
        Row start index of the block.
    start_idx_col :
        Column start index of the row block.
    block_shape :
        Shape of the block

    Returns
    -------
    :
        Block penalty matrix of shape (num_weights, num_weights).
    """
    block_rows, block_cols = full_penalty.shape
    block_penalty = jnp.zeros(block_shape).at[
                    start_idx_row: start_idx_row+block_rows, start_idx_col: start_idx_col+block_cols
                    ].set(full_penalty)
    return block_penalty


def tree_create_block(tree_penalty_blocks, start_idx_row, start_idx_col, block_shape: Tuple[int, int]):
    """
    Create a block diagonal penalty matrix from a tree of penalty blocks.

    Parameters
    ----------
    tree_penalty_blocks :
        Tree containing penalty blocks.
    start_idx_row :
        Tree containing row start indices for each block.
    start_idx_col:
        Tree containing column start indices for each block.
    block_shape :
        Shape of the block diag terms.

    Returns
    -------
    :
        Block penalty matrix of shape (block_matrix_n_rows, block_matrix_n_rows).
    """
    tree_penalty_blocks = jax.tree_util.tree_leaves(tree_penalty_blocks)
    start_idx_row = jax.tree_util.tree_leaves(start_idx_row)
    start_idx_col = jax.tree_util.tree_leaves(start_idx_col)

    return sum(
        create_block_penalty(x, y, z, block_shape)
        for x, y, z in zip(tree_penalty_blocks, start_idx_row, start_idx_col)
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
        n_samples: int = 10 ** 4,
        penalize_null_space: bool = True,
) -> jnp.ndarray:
    r"""
    Define a penalty tensor for an additive component.

    Parameters
    ----------
    basis_component:
        Additive component of a basis.
    n_samples:
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
    one_dim_pen = (compute_energy_penalty(n_samples, b.derivative) for b in basis_component._iterate_over_components())
    out = ndim_tensor_product_basis_penalty(*one_dim_pen)
    if penalize_null_space:
        # In GAMs one penalizes the null space of a linear combinations of positive-semidefinite
        # matrices with positive coefficients (a convex cone). The null space of any matrix in the interior
        # of the cone is the intersection of the null space of individual matrix.
        # Example, take two positive-semidef matrices A, B and positive constant a,b>0, then
        # 0 = v.T * (a * A + b * B) v = a * (v.T * A * v) + b * (v.T * B * v) which is true iif
        # (v.T * A * v) = (v.T * B * v) = 0, since A and B are positive semidef. I.e. v is in null(A) intersect
        # null(B). For this reason we can compute the null-space of the sum of the penality matrices and penalize that.
        # In the original code the sum was used, however, the mean is more stable when summing many matrices.
        null_pen = compute_penalty_null_space(out)
        full_rank = null_pen[None] if ~np.all(null_pen == 0) else jnp.zeros((0, *null_pen.shape))
        out = jnp.concatenate(
            (out, full_rank),
            axis=0
        )
    return out


def compute_energy_penalty_tensor(
        basis: GAMBSplineEval | GAMMultiplicativeBasis | GAMAdditiveBasis,
        n_sample: int = 10**4,
        penalize_null_space: bool = True,
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

    Returns
    -------
        A list with the penalty tensors for each component.

    """
    return [
        compute_energy_penalty_tensor_additive_component(bas, n_sample, penalize_null_space=penalize_null_space)
        for bas in basis
    ]

def compute_penalty_agumented_from_basis(
        basis: GAMBSplineEval | GAMMultiplicativeBasis | GAMAdditiveBasis,
        reg_strength: float,
        n_samples: int = 10 ** 4,
        penalize_null_space: bool = True,
        shift_by: Optional[int] = 0,
        positive_mon_func = jnp.exp,
        apply_identifiability: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x[...,:-1],
):
    """
    Compute the block-diagonal penalization matrix.

    Parameters
    ----------
    basis:
        A possibly composite basis.
    reg_strength:
        The regularization strength.
    n_samples:
        Number of samples for computing the numerical integral of the basis energy.
    penalize_null_space:
        Boolean, if true penalize the null space of every energy penalty component.
    shift_by:
        Shift columns by this integer.
    positive_mon_func:
        Non-linearity applied to the regularization strengths enforce positivity.
    apply_identifiability:
        A function that matches the identifiability constrain at the level of the penalty matrix.
        If for example, we dropped a b-spline element, i.e. dropped a column of the design matrix,
        we should drop the corresponding column of the penalty. Default assumes that we are dropping
        the last column of the design matrix.


    Returns
    -------
    Block-diagonal penalization matrix.

    """
    penalty_tree = compute_energy_penalty_tensor(basis, n_samples, penalize_null_space=penalize_null_space)
    return tree_compute_sqrt_penalty(
        penalty_tree,
        reg_strength,
        shift_by=shift_by,
        positive_mon_func=positive_mon_func,
        apply_identifiability=apply_identifiability
    )


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

