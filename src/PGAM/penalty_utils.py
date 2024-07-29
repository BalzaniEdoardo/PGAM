import timeit

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np


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
    eval_bas = nmo.basis.bspline(samples, knots, order, der=2, outer_ok=False)
    indices = jnp.triu_indices(eval_bas.shape[1])
    square_bas = eval_bas[:, indices[0]] * eval_bas[:, indices[1]]
    dx = samples[1] - samples[0]
    integr = jax.scipy.integrate.trapezoid(square_bas, dx=dx, axis=0)
    energy_pen = jnp.zeros((eval_bas.shape[1], eval_bas.shape[1]))
    energy_pen = energy_pen.at[indices].set(integr)
    energy_pen = energy_pen + energy_pen.T - jnp.diag(energy_pen.diagonal())
    return energy_pen
