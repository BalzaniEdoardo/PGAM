from numpy.typing import NDArray


def inner1d_sum(a: NDArray, b:NDArray) -> float :
    """
    Compute the element-wise product of two array and sum the elements.
    """
    return (a * b).sum()