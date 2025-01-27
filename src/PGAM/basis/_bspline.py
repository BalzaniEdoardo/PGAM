from nemos.basis import BSplineEval
from nemos.basis._spline_basis import bspline
from nemos.basis._basis import min_max_rescale_samples, check_transform_input
from nemos.type_casting import support_pynapple

import numpy as np
from nemos.typing import FeatureMatrix
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from typing import Tuple, Optional

from pynapple import Tsd, TsdFrame, TsdTensor
from ._basis_utils import apply_constraints

class GAMBSplineEval(BSplineEval):

    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "GAMBSplineEval",
            identifiability: Optional[bool] = True,
    ):
        super().__init__(n_basis_funcs=n_basis_funcs, order=order, bounds=bounds, label=label)
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

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    def derivative(self, sample_pts: np.ndarray, der: int = 2):
        """
        Compute the basis derivative.

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
        sample_pts, _ = min_max_rescale_samples(
            sample_pts, getattr(self, "bounds", None)
        )
        # add knots if not passed
        knot_locs = self._generate_knots(is_cyclic=False)
        shape = sample_pts.shape
        X = bspline(
            sample_pts, knot_locs, order=self.order, der=der, outer_ok=False
        )
        X = X.reshape(*shape, X.shape[1])
        keep_index = getattr(self, "_keep_index", None)
        if keep_index is not None:
            X = X[..., keep_index]
        return X

    def _evaluate(
        self, sample_pts: ArrayLike | Tsd | TsdFrame | TsdTensor
    ) -> FeatureMatrix:
        X = super()._evaluate(sample_pts)

        keep_index = getattr(self, "_keep_index", None)

        if keep_index is not None:
            return X[..., keep_index]
        elif self._identifiability:
            X, self._keep_index = apply_constraints(X)
        return X

    def set_identifiability(self, sample_pts: ArrayLike | Tsd | TsdTensor):
        """Find and store linearly independent columns.
        """
        X = super()._evaluate(sample_pts)
        _, self._keep_index = apply_constraints(X)
