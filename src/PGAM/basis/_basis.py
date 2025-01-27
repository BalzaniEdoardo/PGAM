import inspect
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

class GAMBasisMixin:

    def __init__(self, identifiability: bool):
        self._identifiability = int(identifiability)
        # get the attribute or the func
        self.apply_constraints = lambda x: x[...,:-1]

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

    def _get_default_slicing(
            self, split_by_input: bool, start_slice: int
    ) -> Tuple[OrderedDict, int]:
        """Handle default slicing logic."""
        if split_by_input:
            # should we remove this option?
            if self._input_shape_product[0] == 1 or isinstance(
                    self, MultiplicativeBasis
            ):
                split_dict = {
                    self.label: slice(start_slice, start_slice + self.n_output_features)
                }
            else:
                n_basis = self.n_basis_funcs - self._identifiability
                split_dict = {
                    self.label: {
                        f"{i}": slice(
                            start_slice + i * n_basis,
                            start_slice + (i + 1) * n_basis,
                        )
                        for i in range(self._input_shape_product[0])
                    }
                }
        else:
            split_dict = {
                self.label: slice(start_slice, start_slice + self.n_output_features)
            }
        start_slice += self.n_output_features
        return OrderedDict(split_dict), start_slice

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

    @property
    def n_output_features(self) -> int | None:
        """
        Number of features returned by the basis.

        Notes
        -----
        The number of output features can be determined only when the number of inputs
        provided to the basis is known. Therefore, before the first call to ``compute_features``,
        this property will return ``None``. After that call, or after setting the input shape with
        ``set_input_shape``, ``n_output_features`` will be available.
        """
        n_basis = self.n_basis_funcs - self._identifiability
        if self._input_shape_product is not None:
            return n_basis * self._input_shape_product[0]
        return None

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

class GAMAdditiveBasis(AdditiveBasis):

    def __init__(self, basis1, basis2):
        super().__init__(basis1, basis2)

    @support_pynapple("numpy")
    def derivative(self, *xi: ArrayLike):
        return np.hstack(
            self.basis1.derivative(*xi[: self.basis1._n_input_dimensionality]),
            self.basis2.derivative(*xi[self.basis1._n_input_dimensionality:])
        )


class GAMMultiplicativeBasis(MultiplicativeBasis):

    def __init__(self, basis1: GAMBasisMixin, basis2: GAMBasisMixin):
        super().__init__(basis1, basis2)


    def derivative(self, *xi: ArrayLike):
        kron = support_pynapple(conv_type="numpy")(row_wise_kron)

        kwargs1, kwargs2 = dict(), dict()
        if has_param(self.basis1, "_compute_features", "apply_identifiability"):
            kwargs1 = dict(apply_identifiability=False)
        if has_param(self.basis2, "_compute_features", "apply_identifiability"):
            kwargs2 = dict(apply_identifiability=False)

        return kron(
                self.basis1.derivative(*xi[: self.basis1._n_input_dimensionality], **kwargs1),
                self.basis2.derivative(*xi[self.basis1._n_input_dimensionality :], **kwargs2),
                transpose=False,
        )

    def _compute_features(self, *xi: ArrayLike):
        kron = support_pynapple(conv_type="numpy")(row_wise_kron)
        kwargs1, kwargs2 = dict(), dict()
        if has_param(self.basis1, "_compute_features", "apply_identifiability"):
            kwargs1 = dict(apply_identifiability=False)
        if has_param(self.basis2, "_compute_features", "apply_identifiability"):
            kwargs2 = dict(apply_identifiability=False)
        X = kron(
            self.basis1._compute_features(*xi[: self.basis1._n_input_dimensionality], **kwargs1),
            self.basis2._compute_features(*xi[self.basis1._n_input_dimensionality:], **kwargs2),
            transpose=False,
        )
        return X
