"""Implement the PQL iteration.

Statsmodels terminology:
1. Link, as usual. link(mean) = jnp.dot(X, w)
2. fitted: inverse link
3. variance: the variance function of the observation model
"""
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
from nemos.glm.initialize_parameters import INVERSE_FUNCS
from nemos.tree_utils import pytree_map_and_reduce
from jaxopt import LBFGSB,LBFGS
from nemos.observation_models import PoissonObservations

from . import penalty_utils

FLOAT_EPS = jnp.finfo(float).eps

tree_concat =  lambda tree1, tree2, axis : jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis), tree1, tree2)


def model_constructors_for_weights_and_pseudo_data(variance_func, link_func, fisher_scoring=False):
    """
    Compute the IRLS weights and pseudo-data.


    See Chapter 3, pp. 106–107 of:

    Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.

    Parameters
    ----------
    variance_func:
        The variance function of the exponential family distribution.
    link_func:
        The link function, which maps the mean to the linear combination of the weights.

    Returns
    -------
        The IRLS weights and pseudo-data computing function.

    """

    @jax.jit
    def compute_alpha(y, rate):
        dy = y - rate
        variance_der= jax.vmap(jax.grad(variance_func))
        link_func_der = jax.vmap(jax.grad(link_func))
        link_func_der2 = jax.vmap(jax.grad(jax.grad(link_func)))
        corr = variance_der(rate) / variance_func(
            rate
        ) + link_func_der2(rate) / link_func_der(rate)
        return 1.0 + dy * jnp.clip(corr, FLOAT_EPS, jnp.inf)

    @jax.jit
    def compute_z(y, rate, alpha):
        rate = jnp.asarray(rate)
        lin_pred = link_func(rate)
        link_func_der = jax.vmap(jax.grad(link_func))
        return lin_pred + link_func_der(rate) * (y - rate) / alpha

    @jax.jit
    def weight_compute(rate, alpha):
        link_func_der = jax.vmap(jax.grad(link_func))
        dmu_deta = jnp.clip(1. / link_func_der(rate), FLOAT_EPS, jnp.inf)
        w = alpha * dmu_deta**2 / variance_func(rate)
        return w

    def compute_pseudo_data_and_weights(y, rate):
        alpha = jtu.tree_map(jnp.ones_like, rate) if fisher_scoring else jtu.tree_map(compute_alpha, y, rate)
        z = jtu.tree_map(compute_z, y, rate, alpha)
        w = jtu.tree_map(weight_compute, rate, alpha)
        return z, w

    return compute_pseudo_data_and_weights


def unflatten_coeffs(coeffs_flat, leaf_shapes):
    """Slice flat coeffs into a list of arrays matching leaf_shapes."""
    indices = jnp.cumsum(jnp.array([0] + leaf_shapes))
    slices = [coeffs_flat[indices[i]:indices[i+1]] for i in range(len(leaf_shapes))]
    return slices


def weighted_least_squares(X, y, weights):
    """
    Robust WLS via QR decomposition.

    X: (n, d)
    y: (n,) or (n, k)
    weights: (n,)
    """
    sqrt_w = jnp.sqrt(weights)  # (n,)
    Xw = X * sqrt_w[:, None]  # (n, d)
    yw = y * sqrt_w if y.ndim == 1 else y * sqrt_w[:, None]  # (n,) or (n, k)

    Q, R = jnp.linalg.qr(Xw, mode='reduced')  # Q: (n, d), R: (d, d)
    beta = jnp.linalg.solve(R, Q.T @ yw)  # (d,) or (d, k)
    return beta, Xw, yw


def pql_outer_iteration(
        reg_strength,
        init_pars,
        X,
        y,
        penalty_tree,
        obs_model,
        variance_func,
        inner_func,
        compute_sqrt_penalty,
        fisher_scoring=False,
        max_iter=100,
        tol_optim=10**-10,
        tol_update=10**-6
):
    """

    Parameters
    ----------
    reg_strength
    init_pars
    X:
        Tree of 2d arrays.
    y:
        Array, 1d or 2d.
    penalty_tree:
        List of penalty tensors trees (num_pen, n, n). num_pen is >1 when penalizing null-space or for multi-dim penalties.
    obs_model:
        A nemos observation model.
    variance_func
    inner_func:
        PQL inner loop function
    fisher_scoring
    max_iter

    Returns
    -------

    """
    inv_link_func = obs_model.default_inverse_link_function
    solver = LBFGSB(inner_func, tol=tol_optim)
    # make sure everything is float
    reg_strength = jtu.tree_map(lambda x: jnp.asarray(x.astype(float)), reg_strength)
    y = jnp.asarray(y.astype(float))
    X = jtu.tree_map(lambda x: jnp.asarray(x.astype(float)), X)
    init_pars = jtu.tree_map(lambda x: jnp.asarray(x.astype(float)), init_pars)

    lower_bnd = jtu.tree_map(lambda x: jnp.full(x.shape, -12.), reg_strength)
    upper_bnd = jtu.tree_map(lambda x: jnp.full(x.shape, 25.), reg_strength)
    bounds = (lower_bnd, upper_bnd)

    loss_unp = lambda p: obs_model._negative_log_likelihood(y, inv_link_func(X.dot(p[0]) + p[1]))

    link_func = INVERSE_FUNCS[inv_link_func]
    get_pseudo_data_and_weight = model_constructors_for_weights_and_pseudo_data(
        variance_func,
        link_func,
        fisher_scoring=fisher_scoring
    )
    # use nemos par struct
    coef, intercept = init_pars
    struct = jtu.tree_structure(X)

    n_obs = jtu.tree_leaves(X)[0].shape[0]
    leaf_shapes = [leaf.shape[1] for leaf in jtu.tree_leaves(X)]  # dims of each leaf
    i = 0
    for i in range(max_iter):
        # identifiability constraint drops column by default
        sqrt_penalty = compute_sqrt_penalty(
            penalty_tree,
            reg_strength,
        )

        # add a zero corresponding to not-penalizing the intercept
        sqrt_penalty = jnp.hstack((jnp.zeros((sqrt_penalty.shape[0], 1)), sqrt_penalty)) / jnp.sqrt(n_obs)

        # initialize coefficients by fitting a GLM
        if i == 0:
            pen = jtu.tree_map(lambda x: x[:, 1:].T.dot(x[:, 1:]), sqrt_penalty)
            loss = lambda p: loss_unp(p) + 0.5 * p[0].dot(pen).dot(p[0])
            (coef, intercept), state = LBFGS(loss, tol=10**-8).run(init_pars)

        # compute weights
        rate = pytree_map_and_reduce(lambda x, c, inter: inv_link_func(x.dot(c) + inter), sum, X, coef, intercept)
        pseudo_y, weights = get_pseudo_data_and_weight(y, rate)

        # attach intercept and concatenate
        X_agu = jnp.concatenate([jnp.hstack([jnp.ones((n_obs, 1))] + jtu.tree_leaves(X)), sqrt_penalty], axis=0)
        pseudo_y_agu = jnp.concatenate([pseudo_y, jnp.zeros((sqrt_penalty.shape[0], *pseudo_y.shape[1:]))], axis=0)
        weights_agu = jnp.concatenate([weights, jnp.ones((sqrt_penalty.shape[0], *weights.shape[1:]))], axis=0)

        # Wls coefficients and QR decomposition of weighted X
        coeffs, Xw, yw = weighted_least_squares(X_agu, pseudo_y_agu, weights_agu)
        Q, R = jnp.linalg.qr(Xw[:n_obs], mode='reduced')
        new_coef = jtu.tree_unflatten(struct, unflatten_coeffs(coeffs[1:], leaf_shapes))
        new_intercept = coeffs[0]

        # full optimization for regularizer strength
        new_reg_strength, state = solver.run(
            reg_strength,
            bounds=bounds,
            penalty_tree=penalty_tree,
            X = Xw[:n_obs],
            Q = Q,
            R = R,
            y = yw[:n_obs],
        )

        # convergence check
        delta_reg = pytree_map_and_reduce(lambda x, y: jnp.linalg.norm(x - y), max, (coef, intercept), (new_coef, new_intercept))
        if delta_reg < tol_update:
            break

        # update
        reg_strength = new_reg_strength
        coef, intercept = new_coef, new_intercept

    return (coef, intercept), reg_strength, i