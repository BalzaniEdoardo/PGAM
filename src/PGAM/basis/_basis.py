import inspect
from copy import deepcopy

from nemos.basis._basis import AdditiveBasis, MultiplicativeBasis
from typing import Optional
from pynapple import Tsd, TsdFrame, TsdTensor
from collections import OrderedDict
from typing import Tuple
from nemos.basis._basis import MultiplicativeBasis
from numpy.typing import NDArray, ArrayLike
from nemos.typing import FeatureMatrix
from nemos.type_casting import support_pynapple

import numpy as np
from nemos.utils import row_wise_kron

def has_param(bas, method, param_name="apply_identifiability"):
    attr = getattr(bas, method, None)
    if attr is None:
        return False
    sig = inspect.signature(attr)
    return param_name in (
        p.name
        for p in sig.parameters.values()
    )

def _evaluate_on_grid(bas, *n_samples) -> Tuple[Tuple[NDArray], NDArray]:
    sample_tuple = bas._get_samples(*n_samples)
    Xs = np.meshgrid(*sample_tuple, indexing="ij")
    Y = bas.compute_features(*(grid_axis.flatten() for grid_axis in Xs))
    return Xs, Y.reshape(*Xs[0].shape, bas.n_basis_funcs)


class GAMBasisMixin:

    def __add__(self, other):
        return GAMAdditiveBasis(self, other)

    def __mul__(self, other):
        return GAMMultiplicativeBasis(self, other)

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("Exponent should be an integer!")

        if exponent <= 0:
            raise ValueError("Exponent should be a non-negative integer!")

        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        return _evaluate_on_grid(self, *n_samples)

class GAMAtomicBasisMixin(GAMBasisMixin):

    def __init__(self, identifiability: bool):
        self._identifiability = int(identifiability)
        # get the attribute or the func
        self.apply_constraints = lambda x: x[...,:-1]
        # add a basis if the drop column is enabled
        self._n_basis_funcs = self._n_basis_funcs + self._identifiability
        GAMBasisMixin.__init__(self)

    @property
    def identifiability(self):
        return bool(self._identifiability)

    @identifiability.setter
    def identifiability(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"identifiability must be a boolean. {value} provided instead.")
        self._identifiability = int(value)
        if value:
            self.apply_constraints = lambda x: x[...,:-1]
        else:
            self.apply_constraints = lambda x: x

    def _compute_features(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor, apply_identifiability: Optional[bool] = None
    ) -> FeatureMatrix:
        # this gets the compute feature from the inheritance (bspline or similar)
        X = super()._compute_features(sample_pts)
        if apply_identifiability:
            # enforce drop col
            X = X[..., :-1]
        else:
            X = self.apply_constraints(X)
        return X




    @property
    def n_basis_funcs(self) -> int:
        return self._n_basis_funcs - getattr(self, "_identifiability", 0)

    def derivative(self, sample_pts: np.ndarray, der: int = 2, apply_identifiability: Optional[bool] = None):
        """
        Compute the basis derivative and concatenate output on the second axis.

        Parameters
        ----------
        sample_pts:
            Sample points over which computing the derivative.
        der:
            Order of the derivative.

        Returns
        -------
            The derivative at the sample points.
        """
        X = self._derivative(sample_pts, der=der)
        X = X.reshape(X.shape[0], -1)
        if apply_identifiability:
            X = X[...,:-1]
        else:
            X = self.apply_constraints(X)
        return X


class GAMAdditiveBasis(GAMBasisMixin, AdditiveBasis):

    def __init__(self, basis1, basis2):
        AdditiveBasis.__init__(self, basis1, basis2)
        GAMBasisMixin.__init__(self)

    @support_pynapple("numpy")
    def derivative(self, *xi: ArrayLike):
        return np.hstack(
            self.basis1.derivative(*xi[: self.basis1._n_input_dimensionality]),
            self.basis2.derivative(*xi[self.basis1._n_input_dimensionality:])
        )


class GAMMultiplicativeBasis(GAMBasisMixin, MultiplicativeBasis):

    def __init__(self, basis1: GAMBasisMixin, basis2: GAMBasisMixin):
        # copy and reset number of basis and identifiability.
        basis1 = deepcopy(basis1)
        basis1._n_basis_funcs = basis1.n_basis_funcs
        basis1.identifiability = False
        basis2 = deepcopy(basis2)
        basis2._n_basis_funcs = basis2.n_basis_funcs
        basis2.identifiability = False
        MultiplicativeBasis.__init__(self, basis1, basis2)
        GAMBasisMixin.__init__(self)


    def derivative(self, *xi: ArrayLike):
        kron = support_pynapple(conv_type="numpy")(row_wise_kron)

        return kron(
                self.basis1.derivative(*xi[: self.basis1._n_input_dimensionality]),
                self.basis2.derivative(*xi[self.basis1._n_input_dimensionality :]),
                transpose=False,
        )
