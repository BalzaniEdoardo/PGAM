from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from numpy.typing import NDArray

from . import penalty_utils

FLOAT_EPS = jnp.finfo(jnp.float32).eps


_vmap_where = jax.vmap(jnp.where, (None, None, 0), out_axes=0)


@partial(
    jax.jit, static_argnames=("positive_mon_func", "apply_identifiability", "gamma")
)
def _compute_gcv_and_states(
    regularization_strength: Any,
    penalty_tree: Any,
    X: NDArray,
    Q: NDArray,
    R: NDArray,
    y: NDArray,
    positive_mon_func: Callable[[jnp.ndarray], jnp.ndarray] = jnp.exp,
    apply_identifiability: Optional[Callable] = None,
    gamma=1.5,
):
    # identifiability constraint drops column by default
    sqrt_penalty = penalty_utils.tree_compute_sqrt_penalty(
        penalty_tree,
        regularization_strength,
        shift_by=0,
        positive_mon_func=positive_mon_func,
        apply_identifiability=apply_identifiability,
    )

    # add a zero corresponding to not-penalizing the intercept
    sqrt_penalty = jnp.hstack((jnp.zeros((sqrt_penalty.shape[0], 1)), sqrt_penalty))

    n_obs = X.shape[0]
    U, s, V_T = jnp.linalg.svd(jnp.vstack((R, sqrt_penalty)), full_matrices=False)

    # remove low val singular values (numerical stability)
    low_vals = s < FLOAT_EPS * s.max()
    s = jnp.where(low_vals, 0, s)
    U = _vmap_where(low_vals, 0, U)
    V_T = _vmap_where(low_vals, 0, V_T.T).T

    U1 = U[: R.shape[0]]

    # make sure it is 2D
    y = y[:n_obs].reshape(n_obs, -1)
    s_inv = jnp.where(low_vals, 0.0, 1.0 / s)
    square_s_inv = s_inv**2

    # compute A.dot(y) and trA without forming the n_obs x n_obs matrix.
    mat_vec = jnp.dot(V_T, jnp.dot(X.T, y[:n_obs]))
    Ay = X.dot(V_T.T.dot((mat_vec.T * square_s_inv).T))
    trA = (U1**2).sum()
    delta = n_obs - gamma * trA
    alpha = jnp.sum(jnp.power(Ay - y, 2))
    gcv = n_obs * alpha / (delta**2)
    return gcv, alpha, delta, n_obs, U1, V_T, Q, trA, Ay, s_inv, square_s_inv


def symm_mult(sym_mat, factor):
    return jnp.squeeze(
        jnp.dot(
            jnp.dot(factor.T, sym_mat, precision=jax.lax.Precision.HIGHEST),
            factor,
            precision=jax.lax.Precision.HIGHEST,
        )
    )


_vmap_symm_mult = jax.vmap(symm_mult, in_axes=(0, None), out_axes=0)
_vmap_trace = jax.vmap(jnp.linalg.trace, in_axes=0, out_axes=0)


@partial(
    jax.jit, static_argnames=("positive_mon_func", "apply_identifiability", "gamma")
)
def _gcv_grad_compute_from_states(
    regularization_strength,
    penalty_tree,
    y,
    gamma,
    alpha,
    delta,
    n_obs,
    U1,
    V_T,
    Q,
    s_inv,
    positive_mon_func,
    apply_identifiability,
):
    # compute useful vector
    y1 = U1.T @ (Q.T @ y)
    UTU = U1.T @ U1

    blocks = penalty_utils.compute_penalty_blocks(
        penalty_tree,
        apply_identifiability=apply_identifiability,
        shift_by=1,
    )

    comp = V_T.T * s_inv
    M = jtu.tree_map(lambda x: _vmap_symm_mult(x, comp), blocks)
    F = jtu.tree_map(lambda x: jnp.dot(x, UTU, precision=jax.lax.Precision.HIGHEST), M)
    lams = jtu.tree_map(positive_mon_func, regularization_strength)
    alpha_grad = jtu.tree_map(
        lambda x, y: y * _vmap_symm_mult(x, y1),
        jtu.tree_map(lambda x, y: 2 * x - y - jnp.transpose(y, (0, 2, 1)), M, F),
        lams,
    )
    delta_grad = jtu.tree_map(
        lambda x, y: jnp.squeeze(gamma * x * _vmap_trace(y)), lams, F
    )
    gcv_grad = jtu.tree_map(
        lambda x, y: (n_obs / delta**2) * x - 2 * n_obs * alpha / delta**3 * y,
        alpha_grad,
        delta_grad,
    )
    return gcv_grad


def gcv_compute_factory(
    positive_mon_func, apply_identifiability_columns, apply_identifiability, gamma
):

    @jax.custom_vjp
    def _gcv_compute(
        regularization_strength: Any,
        penalty_tree: Any,
        X: NDArray,
        Q: NDArray,
        R: NDArray,
        y: NDArray,
    ):
        """
        Compute the Generalized Cross-validation score.

        Parameters
        ----------
        regularization_strength:
            Pytree containing the current penalization strengths
        penalty_tree:
            Pytree with the same struct as regularization_strength, containing the penalization matrices.
        X:
            Predictors
        Q:
            Q matrix of the QR decomposition of `np.vstack((X, penalty))`
        R:
            R matrix of the QR decomposition of `np.vstack((X, penalty))`
        y:
            Neural activity

        Returns
        -------
        :
            The GCV score

        """
        gcv = _compute_gcv_and_states(
            regularization_strength,
            penalty_tree,
            X,
            Q,
            R,
            y,
            positive_mon_func=positive_mon_func,
            apply_identifiability=apply_identifiability_columns,
            gamma=gamma,
        )[0]
        return gcv

    def _gcv_compute_fwd(
        regularization_strength,
        penalty_tree,
        X,
        Q,
        R,
        y,
    ):
        # Compute and return GCV + intermediates for backward
        gcv, alpha, delta, n_obs, U1, V_T, Q_, trA, Ay, s_inv, square_s_inv = (
            _compute_gcv_and_states(
                regularization_strength,
                penalty_tree,
                X,
                Q,
                R,
                y,
                positive_mon_func=positive_mon_func,
                apply_identifiability=apply_identifiability_columns,
                gamma=gamma,
            )
        )

        # Save inputs for backward (must be JAX types only)
        return gcv, (
            gcv,
            alpha,
            delta,
            n_obs,
            U1,
            V_T,
            Q_,
            trA,
            Ay,
            s_inv,
            square_s_inv,
            regularization_strength,
            penalty_tree,
            X,
            Q,
            R,
            y,
        )

    def _gcv_compute_bwd(res, gcv_bar):
        (
            gcv,
            alpha,
            delta,
            n_obs,
            U1,
            V_T,
            Q,
            trA,
            Ay,
            s_inv,
            square_s_inv,
            regularization_strength,
            penalty_tree,
            X,
            Q,
            R,
            y,
        ) = res

        # Compute full gradient PyTree using the pre-computed states.
        gcv_grad = _gcv_grad_compute_from_states(
            regularization_strength,
            penalty_tree,
            y,
            gamma,
            alpha,
            delta,
            n_obs,
            U1,
            V_T,
            Q,
            s_inv,
            positive_mon_func,
            apply_identifiability,
        )
        return (jtu.tree_map(lambda g: gcv_bar * g, gcv_grad),) + (None,) * 5

    _gcv_compute.defvjp(_gcv_compute_fwd, _gcv_compute_bwd)
    return _gcv_compute
