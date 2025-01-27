from numpy.typing import ArrayLike, NDArray
from typing import Optional
from pynapple import Tsd, TsdFrame, TsdTensor
from ._basis_utils import apply_constraints
import numpy as np


class GAMBasisMixin:

    def __init__(self, identifiability: bool):
        self._identifiability = int(identifiability)
        self._keep_index: Optional[NDArray] = None

    @property
    def identifiability(self):
        return bool(self._identifiability)

    @identifiability.setter
    def identifiability(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"identifiability must be a boolean. {value} provided instead.")
        self._identifiability = value


    def set_identifiability(self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor):
        """Find and store linearly independent columns.
        """
        X = super()._evaluate(sample_pts)
        _, self._keep_index = apply_constraints(X)


class GAMAdditiveBasisMixin:

    def derivative(self, *xi: ArrayLike):
        return np.hstack(
            self.basis1.derivative(*xi[: self.basis1._n_input_dimensionality]),
            self.basis2.derivative(*xi[: self.basis1._n_input_dimensionality])
        )


