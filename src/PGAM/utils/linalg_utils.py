import numpy as np
from numpy.typing import NDArray


def inner1d_sum(A: NDArray, B: NDArray) -> float:
    """
    Compute the sum of the element-wise product of two arrays.

    Parameters
    ----------
    A:
        First array.
    B:
        Second array.

    Returns
    -------
        The sum of the element-wise product
    """
    return (A * B).sum()