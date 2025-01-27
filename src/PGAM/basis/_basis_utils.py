from nemos.identifiability_constraints import apply_identifiability_constraints

def apply_constraints(X, add_intercept: bool = True):
    """
    Drop linearly depednent columns.

    Parameters
    ----------
    X:
        A design matrix.
    add_intercept:
        True, if one must add a constant column: [1, X].

    Returns
    -------
        The full rank matrix.

    """
    X_identifiable, kept_index = apply_identifiability_constraints(X[..., ::-1], add_intercept=add_intercept)
    kept_index = X.shape[-1] - kept_index[::-1] - 1
    return X_identifiable[:, ::-1], kept_index