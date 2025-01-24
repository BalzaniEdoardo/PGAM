import jax
import jax.numpy as jnp
from typing import Any
from numpy.typing import NDArray


def agument_design_matrix(X, penalty):
    return jnp.vstack((X, penalty))


def agument_and_qr(X, penalty):
    return jnp.linalg.qr(X, penalty)


def _gcv_compute(regularization_strength: Any, penalties: Any, X: NDArray, Q: NDArray, R: NDArray, y: NDArray, gamma=1.5):
    """
    Compute the Generalized Cross-validation score.

    Parameters
    ----------
    regularization_strength:
        Pytree containing the current penalization strengths
    penalties:
        Pytree with the same struct as regularization_strength, containing the penalization matrices.
    X:
        Predictors
    Q:
        Q matrix of the QR decomposition of `np.vstack((X, penalty))`
    R:
        R matrix of the QR decomposition of `np.vstack((X, penalty))`
    y:
        Neural activity
    gamma:
        Smoothness enforcing.

    Returns
    -------
    :
        The GCV score

    """
    #penalty_matrix
    pass