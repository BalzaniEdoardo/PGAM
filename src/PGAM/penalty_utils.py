import jax
import jax.numpy as jnp


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


