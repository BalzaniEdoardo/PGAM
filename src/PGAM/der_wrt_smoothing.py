import inspect
from .deriv_det_Slam import *
from .gam_data_handlers import *
from .newton_optim import *
from scipy.optimize import minimize
from scipy.special import erfinv
from .utils.linalg_utils import inner1d_sum

from enum import Enum

try:
    import fast_summations
except:
    pass
from opt_einsum import contract

class ReturnType(Enum):
    eval = 0
    eval_grad = 1
    eval_grad_hess = 2


class d2variance_family(sm.genmod.families.Family):
    """Wraps a statsmodels Family to add second and third order variance derivatives.

    The penalty-likelihood machinery requires dV/dmu, d2V/dmu2, and d3V/dmu3.
    statsmodels only supplies dV/dmu via variance.deriv.  This class monkey-patches
    the remaining derivatives onto the variance object and optionally runs a finite-
    difference sanity check at construction time.
    """

    def __init__(self, family, run_tests=True):
        self.__class__ = family.__class__

        for name, method in inspect.getmembers(family):
            if name.startswith("__") and name.endswith("__"):
                continue
            self.__setattr__(name, method)
            if name == "variance":
                self.variance.deriv2 = lambda x: variance_deriv2(self, x)
                self.variance.deriv3 = lambda x: variance_deriv3(self, x)

        self.__link = family.link
        self.deriv3 = lambda x: link_deriv3(self, x)
        epsi = 10**-4
        func = lambda x: self.variance.deriv(x)
        func1 = lambda x: self.variance.deriv2(x)
        self.variance.approx_deriv2 = lambda x: (func(x + epsi) - func(x - epsi)) / (
            2 * epsi
        )
        if run_tests:
            if isinstance(self, sm.genmod.families.family.Tweedie):
                x = np.random.uniform(-1, 1, size=20)
            else:
                x = np.random.uniform(0.1, 0.9, size=10)
            approx_der = (func(x + epsi) - func(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der, self.variance.deriv2(x)):
                print(
                    "\n2rd order variance deriv possibly wrong for %s\n"
                    % self.__class__
                )
            else:
                print("2nd order deriv of variance function is ok!")
            approx_der2 = (func1(x + epsi) - func1(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der2, self.variance.deriv3(x)):
                print(
                    "\n3rd order variance deriv possibly wrong for %s\n"
                    % self.__class__
                )
            else:
                print("3rd order deriv of variance function is ok!")


class deriv3_link(sm.genmod.families.links.Link):
    """Wraps a statsmodels Link to add third and fourth order link derivatives.

    The REML Hessian computation requires g'''(mu) and g''''(mu).  This class
    monkey-patches deriv3 and deriv4 onto the link object and runs a finite-
    difference sanity check at construction time.
    """

    def __init__(self, link, run_tests=True):
        self.__class__ = link.__class__

        for name, method in inspect.getmembers(link):
            if name.startswith("__") and name.endswith("__"):
                continue
            self.__setattr__(name, method)

        self.__link = link
        self.deriv3 = lambda x: link_deriv3(self, x)
        self.deriv4 = lambda x: link_deriv4(self, x)

        epsi = 10**-4
        func = lambda x: self.deriv2(x)
        func1 = lambda x: self.deriv3(x)
        self.approx_deriv3 = lambda x: (func(x + epsi) - func(x - epsi)) / (2 * epsi)
        if run_tests:
            x = np.random.uniform(0.1, 0.9, size=10)
            approx_der = (func(x + epsi) - func(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der, self.deriv3(x)):
                print("\n3rd order deriv possibly wrong for %s\n" % self.__class__)
            else:
                print("3rd order deriv of link function is ok!")
            approx_der1 = (func1(x + epsi) - func1(x - epsi)) / (2 * epsi)
            if checkGrad(approx_der1, self.deriv4(x)):
                print("\n4th order deriv possibly wrong for %s\n" % self.__class__)
            else:
                print("4th order deriv of link function is ok!")

    def __call__(self, x):
        return self.__link(x)


def link_deriv3(self, x):
    """Third derivative of the link function g'''(x), dispatched by family type."""
    # probit must come before Logit: in current statsmodels Probit inherits from
    # CDFLink -> Logit, so isinstance(probit_instance, Logit) is True.
    if isinstance(self, sm.genmod.families.links.probit):
        return (
            2
            * np.sqrt(2)
            * (np.pi ** (3 / 2))
            * np.exp(3 * erfinv(-1 + 2 * x) ** 2)
            * (1 + 4 * erfinv(-1 + 2 * x) ** 2)
        )
    elif isinstance(self, sm.genmod.families.links.identity):
        return np.zeros(shape=x.shape)
    elif isinstance(self, sm.genmod.families.links.Log):
        return 2 / x**3
    elif isinstance(self, sm.genmod.families.links.Logit):
        return -((-2 + 6 * x - 6 * x**2) / ((1 - x) ** 3 * x**3))
    elif isinstance(self, sm.genmod.families.links.inverse_power):
        return -6 / x**4
    else:
        raise NotImplementedError("deriv3 not implemented for %s" % self.__class__)


def link_deriv4(self, x):
    """Fourth derivative of the link function g''''(x), dispatched by family type."""
    # probit must come before Logit: in current statsmodels Probit inherits from
    # CDFLink -> Logit, so isinstance(probit_instance, Logit) is True.
    if isinstance(self, sm.genmod.families.links.probit):
        return (
            4
            * np.sqrt(2)
            * (np.pi**2)
            * np.exp(4 * erfinv(2 * x - 1) ** 2)
            * erfinv(2 * x - 1)
            * (12 * erfinv(2 * x - 1) ** 2 + 7)
        )
    elif isinstance(self, sm.genmod.families.links.identity):
        return np.zeros(shape=x.shape)
    elif isinstance(self, sm.genmod.families.links.Log):
        return -6 / x**4
    elif isinstance(self, sm.genmod.families.links.Logit):
        return 6 * (4 * x**3 - 6 * x**2 + 4 * x - 1) / (((1 - x) ** 4) * x**4)
    elif isinstance(self, sm.genmod.families.links.inverse_power):
        return 24 / x**5
    else:
        raise NotImplementedError("deriv3 not implemented for %s" % self.__class__)


def variance_deriv2(family_object, x):
    """Second derivative of the variance function V''(mu), dispatched by family type."""
    if isinstance(family_object, sm.genmod.families.family.Poisson):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Gamma):
        return 2 * np.ones(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Binomial):
        return -2 * np.ones(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Gaussian):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.InverseGaussian):
        return 6 * x
    elif isinstance(family_object, sm.genmod.families.family.Tweedie):
        k = family_object.var_power
        res = k * (k - 1) * np.fabs(x) ** (k - 2)
        return res
    else:
        raise NotImplementedError(
            "deriv3 not implemented for %s" % family_object.__class__
        )


def variance_deriv3(family_object, x):
    """Third derivative of the variance function V'''(mu), dispatched by family type."""
    if isinstance(family_object, sm.genmod.families.family.Poisson):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Gamma):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Binomial):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Gaussian):
        return np.zeros(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.InverseGaussian):
        return 6 * np.ones(shape=x.shape)
    elif isinstance(family_object, sm.genmod.families.family.Tweedie):
        k = family_object.var_power
        res = k * (k - 1) * (k - 2) * np.fabs(x) ** (k - 3)
        ii = np.flatnonzero(x < 0)
        res[ii] = -1 * res[ii]
        return res
    else:
        raise NotImplementedError(
            "deriv3 not implemented for %s" % family_object.__class__
        )


def create_Slam(rho, sm_handler, var_list):
    """Assemble the total penalty matrix S_lambda = sum_j exp(rho_j) * S_j.

    Parameters
    ----------
    rho : ndarray, shape (M,)
        Log smoothing parameters.  lambda_j = exp(rho_j).
    sm_handler : smooths_handler
        Object holding the per-covariate smoothing matrices.
    var_list : list of str
        Variable names whose penalties to include.

    Returns
    -------
    Slam : ndarray, shape (p, p)
        Weighted sum of all penalty matrices.
    """
    S_list = compute_Sjs(sm_handler, var_list)
    S_tens = np.zeros((len(S_list),) + S_list[0].shape)
    S_tens[:, :, :] = S_list
    Slam = np.einsum("i,ikl->kl", np.exp(rho), S_tens)
    return Slam


def compute_Sall(sm_handler, var_list):
    """Return a flat list of all raw penalty matrices S_j (unweighted, one per lambda).

    Parameters
    ----------
    sm_handler : smooths_handler
    var_list : list of str

    Returns
    -------
    Sall : list of ndarray
        One matrix per smoothing parameter, in the order they appear in var_list.
    """
    Sall = []
    for var_name in var_list:
        S_list = sm_handler[var_name].S_list
        for S in S_list:
            Sall += [S]

    return Sall


def penalty_ll(rho, beta, sm_handler, var_list, phi_est):
    """Log-prior (penalty) contribution to the penalised log-likelihood.

    Returns  -0.5 / phi * beta^T S_lambda beta,  where S_lambda = sum_j lambda_j S_j.

    Parameters
    ----------
    rho : ndarray, shape (M,)
    beta : ndarray, shape (p,)
    sm_handler : smooths_handler
    var_list : list of str
    phi_est : float
        Scale parameter phi.
    """
    Slam = create_Slam(rho, sm_handler, var_list)
    penalty = -0.5 / phi_est * np.dot(np.dot(beta, Slam), beta)
    return penalty


def penalty_ll_Slam(Slam, beta, phi_est):
    """Penalty contribution when the assembled S_lambda matrix is already available.

    Returns  -0.5 / phi * beta^T Slam beta.
    """
    penalty = -0.5 / phi_est * np.dot(np.dot(beta, Slam), beta)
    return penalty


def dbeta_penalty_ll(rho, beta, sm_handler, var_list, phi_est):
    """Gradient of the penalty log-likelihood wrt beta: -S_lambda beta / phi."""
    Slam = create_Slam(rho, sm_handler, var_list)
    grad = -1 / phi_est * np.dot(Slam, beta)
    return grad


def dbeta_penalty_ll_Slam(Slam, beta, phi_est):
    """Gradient of the penalty log-likelihood wrt beta, given pre-assembled Slam."""
    grad = -1 / phi_est * np.dot(Slam, beta)
    return grad


def d2beta_penalty_ll(rho, beta, sm_handler, var_list, phi_est):
    """Hessian of the penalty log-likelihood wrt beta: -S_lambda / phi."""
    Slam = create_Slam(rho, sm_handler, var_list)
    grad = -1 / phi_est * Slam
    return grad


def d2beta_penalty_ll_Slam(Slam, beta, phi_est):
    """Hessian of the penalty log-likelihood wrt beta, given pre-assembled Slam."""
    grad = -1 / phi_est * Slam
    return grad


def unpenalized_ll(beta, y, X, family, phi_est, omega=1):
    """Unpenalised log-likelihood l(beta; y) evaluated at beta.

    Parameters
    ----------
    beta : ndarray, shape (p,)
    y : ndarray, shape (n,)
    X : ndarray, shape (n, p)
    family : d2variance_family
    phi_est : float
    omega : float or ndarray
        Prior weights.
    """
    mu = family.link.inverse(np.dot(X, beta))
    ll = family.loglike(y, mu, var_weights=omega, scale=phi_est)
    return ll


def dbeta_unpenalized_ll(beta, y, X, family, phi_est):
    """Score vector dl/dbeta of the unpenalised log-likelihood.

    Returns (y - mu) / (g'(mu) * V(mu) * phi), contracted with X.
    """
    mu = family.link.inverse(np.dot(X, beta))
    vector = (y - mu) / (family.link.deriv(mu) * family.variance(mu))
    grad_unp_ll = 1.0 / phi_est * np.dot(vector, X)
    return grad_unp_ll


def d2beta_unpenalized_ll(beta, y, X, family, phi_est):
    """Expected (working) Hessian H = d2l/dbeta2 of the unpenalised log-likelihood.

    Uses the PIRLS weight  w = alpha * (dmu/deta)^2 / V(mu)  so the result is
    -X^T diag(w) X / phi.  Negative weights below 1e-15 in absolute value are
    clipped to zero; larger negative values raise ValueError.
    """
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, beta))
    dy = y - mu
    corr = family.variance.deriv(mu) / family.variance(mu) + family.link.deriv2(
        mu
    ) / family.link.deriv(mu)
    alpha = 1 + dy * np.clip(corr, FLOAT_EPS, np.inf)
    mu = family.link.inverse(np.dot(X, beta))
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta**2 / family.variance(mu)
    if any(np.abs(w[w < 0]) > 10**-15):
        raise ValueError("w takes negative values")
    else:
        w = np.clip(w, 0, np.inf)
    WX = (np.sqrt(w) * X.T).T
    hess = -1 / phi_est * np.dot(WX.T, WX)
    return hess


def alpha_mu(y, mu, family):
    """PIRLS correction factor alpha(mu) = 1 + (y - mu) * c(mu).

    c(mu) = V'(mu)/V(mu) + g''(mu)/g'(mu) captures the departure from canonical
    link / unit variance.  For canonical families alpha = 1 (c = 0).
    """
    FLOAT_EPS = np.finfo(float).eps
    dy = y - mu
    corr = family.variance.deriv(mu) / family.variance(mu) + family.link.deriv2(
        mu
    ) / family.link.deriv(mu)
    if not any(np.abs(corr[corr < 0]) > 10**-15):
        corr = np.clip(corr, FLOAT_EPS, np.inf)
    alpha = 1 + dy * corr
    return alpha


def alpha_deriv(y, mu, family):
    """First derivative of alpha wrt mu: dalpha/dmu.

    Used when computing the gradient of the PIRLS weights w wrt mu, which in turn
    feeds into dH/drho.
    """
    FLOAT_EPS = np.finfo(float).eps
    term1 = -(
        family.variance.deriv(mu) / family.variance(mu)
        + family.link.deriv2(mu) / family.link.deriv(mu)
    )
    dy = y - mu
    add1 = (
        family.variance.deriv2(mu) * family.variance(mu)
        - family.variance.deriv(mu) ** 2
    ) / family.variance(mu) ** 2
    add2 = (
        family.link.deriv3(mu) * family.link.deriv(mu) - family.link.deriv2(mu) ** 2
    ) / np.clip(family.link.deriv(mu) ** 2, FLOAT_EPS, np.inf)
    return term1 + dy * (add1 + add2)


def d3beta_unpenalized_ll(beta, y, X, family, phi_est):
    """Third-order derivative of the unpenalised log-likelihood wrt beta.

    Returns a rank-3 tensor of shape (p, p, p).  Expensive: O(n p^3).
    Falls back from fast_summations C extension to plain einsum if unavailable.
    """
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, beta))
    dalpha_dmu = alpha_deriv(y, mu, family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    temp = -1 / phi_est * contract("i,i,im,ir,il->mrl", dalpha_dmu, dmu_deta, X, X, X)
    try:
        d3beta_ll = (
            -1
            / phi_est
            * fast_summations.d3beta_unpenalized_ll_summation(X, dalpha_dmu, dmu_deta)
        )
    except:
        d3beta_ll = (
            -1
            / phi_est
            * np.einsum("i,i,im,ir,il -> mrl", dalpha_dmu, dmu_deta, X, X, X)
        )
    return d3beta_ll


def ll_MLE_rho(
    rho,
    y,
    X,
    family,
    sm_handler,
    var_list,
    phi_est,
    conv_criteria="deviance",
    max_iter=10**3,
    tol=1e-10,
    returnMLE=False,
):
    """Run PIRLS to convergence at fixed rho and return the penalised score or beta.

    Parameters
    ----------
    rho : ndarray, shape (M,)
    y, X, family, sm_handler, var_list, phi_est : standard GAM inputs.
    conv_criteria : 'deviance' or 'gcv'
    max_iter, tol : PIRLS stopping criteria.
    returnMLE : bool
        If True, return beta_hat instead of the penalised score gradient.

    Returns
    -------
    beta_hat or penalised_score_gradient : ndarray
    """
    mu = family.starting_mu(y)
    converged = False
    old_conv_score = -100
    n_obs = y.shape[0]
    iteration = 0
    smooth_pen = np.exp(rho)
    sm_handler.set_smooth_penalties(smooth_pen, var_list)
    pen_matrix = sm_handler.get_penalty_agumented(var_list)
    Xagu = np.vstack((X, pen_matrix))
    yagu = np.zeros(Xagu.shape[0])
    wagu = np.ones(Xagu.shape[0])

    while not converged:
        FLOAT_EPS = np.finfo(float).eps
        alpha = alpha_mu(y, mu, family)
        dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
        w = alpha * dmu_deta**2 / family.variance(mu)
        lin_pred = family.predict(mu)
        z = lin_pred + family.link.deriv(mu) * (y - mu) / alpha
        yagu[:n_obs] = z
        wagu[:n_obs] = w
        model = sm.WLS(yagu, Xagu, wagu)
        fit_OLS = model.fit()
        lin_pred = np.dot(X[:n_obs, :], fit_OLS.params)
        mu = family.fitted(lin_pred)

        conv_score = convergence_score(
            y, model, family, eta=lin_pred, criteria=conv_criteria
        )
        converged = abs(conv_score - old_conv_score) < tol * conv_score
        old_conv_score = conv_score
        if iteration >= max_iter:
            break
        iteration += 1
    if returnMLE:
        return fit_OLS.params
    ll_beta_hat = dbeta_unpenalized_ll(
        fit_OLS.params, y, X, family, phi_est
    ) + dbeta_penalty_ll(rho, fit_OLS.params, sm_handler, var_list, phi_est)
    return ll_beta_hat


def convergence_score(y, model, family, criteria="gcv", eta=None):
    """Dispatch to the chosen PIRLS convergence criterion."""
    if criteria == "gcv":
        return compute_gcv_convergence(y, model)
    if criteria == "deviance":
        return compute_deviance(y, eta, family)


def compute_gcv_convergence(y, model):
    """GCV score used as a PIRLS convergence criterion."""
    res = sm.OLS(model.wendog, model.wexog).fit()
    n_obs = y.shape[0]
    hat_diag = res.get_influence().hat_matrix_diag[:n_obs]
    trA = hat_diag.sum()
    rsd = model.wendog[:n_obs] - res.fittedvalues[:n_obs]
    rss = np.sum(np.power(rsd, 2))
    sig_hat = rss / (n_obs - trA)
    gcv = sig_hat * n_obs / (n_obs - trA)
    return gcv


def compute_deviance(y, eta, family):
    """Deviance used as a PIRLS convergence criterion."""
    mu = family.link.inverse(eta)
    return family.deviance(y, mu)


def mle_gradient_bassed_optim(
    rho,
    sm_handler,
    var_list,
    y,
    X,
    family,
    phi_est=1,
    method="Newton-CG",
    num_random_init=1,
    beta_zero=None,
    tol=10**-8,
    max_iter=1000,
):
    """Fit beta_hat at fixed rho via gradient-based optimisation of the penalised likelihood.

    Minimises  -l(beta) - penalty(beta; rho)  using scipy.optimize.minimize.
    With method='Newton-CG' the exact Hessian is passed; other methods use gradient only.

    Parameters
    ----------
    rho : ndarray, shape (M,)
    sm_handler, var_list, y, X, family, phi_est : standard GAM inputs.
    method : str
        scipy optimiser name.  'Newton-CG' is most reliable.
    num_random_init : int
        Number of random restarts when beta_zero is None.
    beta_zero : ndarray or None
        Starting point; if None, num_random_init random N(0, 0.1) starts are tried.
    tol, max_iter : stopping criteria.

    Returns
    -------
    beta_hat : ndarray, shape (p,)
    res : OptimizeResult
    beta_zero : ndarray
    """
    Slam = create_Slam(rho, sm_handler, var_list)
    func = lambda beta: -1 * (
        unpenalized_ll(beta, y, X, family, phi_est, omega=1)
        + penalty_ll_Slam(Slam, beta, phi_est)
    )
    grad_func = lambda beta: -1 * (
        dbeta_unpenalized_ll(beta, y, X, family, phi_est)
        + dbeta_penalty_ll_Slam(Slam, beta, phi_est)
    )
    if method == "Newton-CG":
        hess_func = lambda beta: -1 * (
            d2beta_penalty_ll_Slam(Slam, beta, phi_est)
            + d2beta_unpenalized_ll(beta, y, X, family, phi_est)
        )
    else:
        hess_func = None
    if beta_zero is None:
        curr_min = np.inf
        for kk in range(num_random_init):
            beta0 = np.random.normal(0, 0.1, X.shape[1])
            tmp = minimize(
                func, beta0, method=method, jac=grad_func, hess=hess_func, tol=tol
            )
            if tmp.fun < curr_min:
                res = tmp
                curr_min = tmp.fun
                beta_zero = beta0.copy()
    res = minimize(
        func,
        beta_zero,
        method=method,
        jac=grad_func,
        hess=hess_func,
        tol=tol,
        options={"maxiter": max_iter, "disp": False},
    )

    return res.x, res, beta_zero


def Vbeta_rho(
    rho,
    b_hat,
    y,
    X,
    family,
    sm_handler,
    var_list,
    phi_est,
    inverse=False,
    compute_grad=False,
    return_logdet=False,
):
    """Bayesian posterior covariance V_beta = -(H + S_lambda)^{-1}, or its inverse.

    Computed via QR decomposition of sqrt(W) X followed by SVD of [R; B], where B
    is the Cholesky square root of S_lambda.  This avoids forming X^T W X explicitly
    and is numerically stable when p is large relative to n.

    Parameters
    ----------
    rho : ndarray, shape (M,)
    b_hat : ndarray, shape (p,)
        MAP estimate of beta at which to evaluate V_beta.
    y, X, family, sm_handler, var_list, phi_est : standard GAM inputs.
    inverse : bool
        If False (default), returns -(H + S_lambda)  (i.e. the negative Hessian
        of the penalised log-likelihood, which equals V_beta^{-1} up to sign).
        If True, returns -(H + S_lambda)^{-1} = V_beta.
    compute_grad : bool
        If True, re-optimise beta_hat before computing (expensive; for debugging).
    return_logdet : bool
        If True, also return log|H + S_lambda| computed for free from the SVD
        singular values (avoids a second O(p³) decomposition in the caller).

    Returns
    -------
    sum_hes_inv : ndarray, shape (p, p)
        Either -(H + S_lambda) or V_beta depending on `inverse`.
    log_det : float, optional
        log|H + S_lambda|.  Only returned when return_logdet=True.
    """
    if compute_grad:
        b_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method="Newton-CG",
            num_random_init=10,
        )[0]
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, b_hat))
    alpha = alpha_mu(y, mu, family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta**2 / family.variance(mu)
    if any(np.abs(w[w < 0]) > 10**-15):
        raise ValueError("w takes negative values")
    else:
        w = np.clip(w, 0, np.inf)
    WX = (np.sqrt(w) * X.T).T

    # mode='r' skips forming Q (DORGQR), saving ~2/3 of the QR time; Q is never used
    R = np.linalg.qr(WX, mode="r")
    sm_handler.set_smooth_penalties(np.exp(rho), var_list)
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    _, s, V_T = linalg.svd(np.vstack((R, B)))

    i_rem = np.where(s < 10 ** (-8) * s.max())[0]
    s = np.delete(s, i_rem, 0)
    V_T = np.delete(V_T, i_rem, 0)

    di = np.diag_indices(s.shape[0])
    D = np.zeros((s.shape[0], s.shape[0]))
    if inverse:
        D[di] = phi_est / (s) ** 2
    else:
        D[di] = (s) ** 2 / phi_est

    V_T = np.asarray(V_T)
    sum_hes_inv = -(V_T.T @ D @ V_T)

    if return_logdet:
        log_det = 2.0 * np.sum(np.log(s)) - len(s) * np.log(phi_est)
        return sum_hes_inv, log_det
    return sum_hes_inv

def Vbeta_rho_all(
    rho,
    b_hat,
    y,
    X,
    family,
    sm_handler,
    var_list,
    phi_est,
    return_logdet=False,
):
    """Bayesian posterior covariance V_beta = -(H + S_lambda)^{-1}, or its inverse.

    Computed via QR decomposition of sqrt(W) X followed by SVD of [R; B], where B
    is the Cholesky square root of S_lambda.  This avoids forming X^T W X explicitly
    and is numerically stable when p is large relative to n.

    Parameters
    ----------
    rho : ndarray, shape (M,)
    b_hat : ndarray, shape (p,)
        MAP estimate of beta at which to evaluate V_beta.
    y, X, family, sm_handler, var_list, phi_est : standard GAM inputs.
    return_logdet : bool
        If True, also return log|H + S_lambda| computed for free from the SVD
        singular values (avoids a second O(p³) decomposition in the caller).

    Returns
    -------
    sum_hes_inv : ndarray, shape (p, p)
        Either (H + S_lambda) or V_beta depending on `inverse`.
    log_det : float, optional
        log|H + S_lambda|.  Only returned when return_logdet=True.
    """
    FLOAT_EPS = np.finfo(float).eps
    mu = family.link.inverse(np.dot(X, b_hat))
    alpha = alpha_mu(y, mu, family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta**2 / family.variance(mu)
    if any(np.abs(w[w < 0]) > 10**-15):
        raise ValueError("w takes negative values")
    else:
        w = np.clip(w, 0, np.inf)
    WX = (np.sqrt(w) * X.T).T

    # mode='r' skips forming Q (DORGQR), saving ~2/3 of the QR time; Q is never used
    R = np.linalg.qr(WX, mode="r")
    sm_handler.set_smooth_penalties(np.exp(rho), var_list)
    B = sm_handler.get_penalty_agumented(var_list)
    B = np.array(B, dtype=np.float64)
    _, s, V_T = linalg.svd(np.vstack((R, B)))

    i_rem = np.where(s < 10 ** (-8) * s.max())[0]
    s = np.delete(s, i_rem, 0)
    V_T = np.delete(V_T, i_rem, 0)

    d = s ** 2 / phi_est
    dinv = 1 / d

    V_T = np.asarray(V_T)
    sum_hes_inv = d * V_T.T @ V_T
    sum_hes = dinv * V_T.T @ V_T


    if return_logdet:
        # log|H+S| = 2·Σlog(s) − r·log(phi),  free from the SVD already done above
        log_det = 2.0 * np.sum(np.log(s)) - len(s) * np.log(phi_est)
        return sum_hes_inv, sum_hes, log_det
    return sum_hes_inv, sum_hes


def dbeta_hat(
    rho,
    b_hat,
    S_all,
    sm_handler,
    var_list,
    y,
    X,
    family,
    phi_est=1,
    compute_gradient=False,
    method="Newton-CG",
):
    """Gradient of the MAP estimate beta_hat wrt log-smoothing parameters rho.

    From implicit differentiation of the score equations
        (H + S_lambda) beta_hat = H_score,
    one obtains
        J = d beta_hat / d rho_k = -(H + S_lambda)^{-1} (d S_lambda / d rho_k) beta_hat
                                 = V_beta * lambda_k * S_k * beta_hat / phi

    Parameters
    ----------
    rho : ndarray, shape (M,)
    b_hat : ndarray, shape (p,)
        MAP estimate (reused unless compute_gradient=True).
    S_all : list of ndarray
        Flat list of raw penalty matrices, one per smoothing parameter.
    sm_handler, var_list, y, X, family, phi_est : standard GAM inputs.
    compute_gradient : bool
        If True, re-optimise beta_hat first (expensive).

    Returns
    -------
    true_grad : ndarray, shape (M, p)
        J[k, :] = d beta_hat / d rho_k.
    """
    if compute_gradient:
        beta = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method=method,
            num_random_init=10,
            tol=10**-12,
        )[0]
    else:
        beta = b_hat
    sum_hes_inv = Vbeta_rho(
        rho, beta, y, X, family, sm_handler, var_list, phi_est, inverse=True
    )

    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    Slam_tensor = (Slam_tensor.T * np.exp(rho)).T / phi_est
    P1 = np.einsum("kij,j->ki", Slam_tensor, beta)
    true_grad = np.einsum("li,ki->kl", sum_hes_inv, P1)
    return true_grad


def d2beta_hat(rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est=1,
               dH_drho=None, neg_sum_inv=None, grad_beta_precomp=None):
    r"""Second derivative of beta_hat wrt rho: d^2 beta_hat / (d rho_h d rho_k).

    Differentiates the expression for J = d beta_hat / d rho a second time.
    The formula in the book (Wood 2017) contains a sign error; the correct
    expression is implemented here (see project Overleaf notes).

    Parameters
    ----------
    dH_drho : ndarray, shape (M, p, p), optional
        Pre-computed d H / d rho.  If provided, the grad_H_drho call is skipped.
    neg_sum_inv : ndarray, shape (p, p), optional
        Pre-computed -(H + S_lambda)^{-1} = V_beta.  If provided, the Vbeta_rho
        (QR + SVD) call is skipped.
    grad_beta_precomp : ndarray, shape (M, p), optional
        Pre-computed d beta_hat / d rho.  If provided, both the Vbeta_rho call
        and the P1/grad_beta einsum are skipped.

    Returns
    -------
    hes_beta : ndarray, shape (M, M, p)
        hes_beta[h, k, :] = d^2 beta_hat / (d rho_h d rho_k).
    """
    # dSlam/drho_k = exp(rho_k) * S_k / phi  (always needed for grad_neg_sum)
    dSlam_drho = (np.array(S_all).T * np.exp(rho)).T / phi_est

    # --- V_beta = -(H+S)^{-1}: dominant cost, skipped when pre-supplied ---
    if neg_sum_inv is None:
        neg_sum_inv = np.array(
            Vbeta_rho(
                rho, b_hat, y, X, family, sm_handler, var_list, phi_est,
                inverse=True, compute_grad=False,
            )
        )

    # --- grad_beta[k] = neg_sum_inv @ (dSlam_drho[k] @ b_hat) ---
    if grad_beta_precomp is not None:
        grad_beta = grad_beta_precomp
    else:
        P1 = np.einsum("kij,j->ki", dSlam_drho, b_hat)   # (M, p)
        grad_beta = np.einsum("li,ki->kl", neg_sum_inv, P1)  # (M, p)

    # --- dH/drho: skipped when pre-supplied ---
    if dH_drho is None:
        dH_drho = grad_H_drho(
            rho, b_hat, y, X, sm_handler, var_list, family, S_all, phi_est,
            dB=grad_beta,
        )

    grad_neg_sum = -(dH_drho + dSlam_drho)

    if np.isfortran(neg_sum_inv):
        neg_sum_inv = np.array(neg_sum_inv, order="C")
    if np.isfortran(grad_neg_sum):
        grad_neg_sum = np.array(grad_neg_sum, order="C")
    if np.isfortran(b_hat):
        b_hat = np.array(b_hat, order="C")
    if np.isfortran(grad_beta):
        grad_beta = np.array(grad_beta, order="C")

    T = np.tensordot(grad_beta, dSlam_drho, axes=(1, 2))  # (M, M, p)

    # step 1: right-multiply neg_sum_inv into grad_neg_sum[h]
    tmp2 = np.einsum("ij,hjl->hil", neg_sum_inv, grad_neg_sum)  # (M, p, p)

    # step 2: contract tmp2 with grad_beta.
    # Note: tmp_p[k,l] = (-neg_sum_inv @ dSlam_drho[k] @ b_hat)[l] = -grad_beta[k,l],
    # so tmp_p = -grad_beta and the two intermediate tensors are not needed.
    add1 = -np.einsum("hil,kl->hki", tmp2, grad_beta)
    add2 = T @ neg_sum_inv.T

    di1, di2 = np.diag_indices(rho.shape[0])
    hes_beta = add1 + add2

    # Diagonal correction: einsum("ij,hjl,l->hi", neg_sum_inv, dSlam_drho, b_hat)
    # = neg_sum_inv @ P1[h] = grad_beta[h], so just add grad_beta on the diagonal.
    hes_beta[di1, di2] += grad_beta

    return hes_beta


def w_mu(mu, y, family):
    """PIRLS weights w(mu) = alpha(mu) * (dmu/deta)^2 / V(mu)."""
    FLOAT_EPS = np.finfo(float).eps
    alpha = alpha_mu(y, mu, family)
    dmu_deta = np.clip(1 / family.link.deriv(mu), FLOAT_EPS, np.inf)
    w = alpha * dmu_deta**2 / family.variance(mu)
    return w


def w_deriv(mu, y, family):
    """First derivative of the PIRLS weights wrt mu: dw/dmu."""
    FLOAT_EPS = np.finfo(float).eps
    alpha_prime = alpha_deriv(y, mu, family)
    g_prime = family.link.deriv(mu)
    g_2prime = family.link.deriv2(mu)
    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    alpha = alpha_mu(y, mu, family)
    w_prime = (
        alpha_prime * g_prime * V - alpha * (2 * g_2prime * V + V_prime * g_prime)
    ) / (np.clip(g_prime**3, FLOAT_EPS, np.inf) * V**2)
    return w_prime


def w_2deriv(mu, y, family):
    """Second derivative of the PIRLS weights wrt mu: d2w/dmu2."""
    FLOAT_EPS = np.finfo(float).eps
    alpha = alpha_mu(y, mu, family)
    alpha_prime = alpha_deriv(y, mu, family)
    alpha_2prime = alpha_deriv2(y, mu, family)
    g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)
    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)

    NUM = alpha_prime * g_prime * V - alpha * (2 * g_2prime * V + V_prime * g_prime)
    NUM_prime = (
        alpha_2prime * g_prime * V
        + alpha_prime * g_2prime * V
        + alpha_prime * g_prime * V_prime
        - alpha_prime * (2 * g_2prime * V + g_prime * V_prime)
        - alpha
        * (
            2 * g_3prime * V
            + 2 * g_2prime * V_prime
            + V_2prime * g_prime
            + V_prime * g_2prime
        )
    )

    tmp_DEN_prime = 3 * g_2prime * V + 2 * g_prime * V_prime
    w_2prime = (NUM_prime * (g_prime * V) - NUM * tmp_DEN_prime) / (
        g_prime**4 * V**3
    )
    return w_2prime


def dw_dbeta(beta, y, family, X):
    """Gradient of PIRLS weights wrt beta: dw/dbeta, shape (n, p).

    dw/dbeta_l = (dw/dmu)(dmu/deta) * X_l  for each observation.
    """
    mu = family.link.inverse(np.dot(X, beta))
    w_prime = w_deriv(mu, y, family)
    g_prime = np.clip(family.link.deriv(mu), np.finfo(float).eps, np.inf)
    grad_w = (w_prime / g_prime) * X.T
    return grad_w


def w_rho(
    rho,
    beta,
    y,
    X,
    sm_handler,
    var_list,
    family,
    S_all,
    phi_est,
    compute_grad=False,
    method="Newton-CG",
):
    """PIRLS weights evaluated at beta_hat(rho).  Primarily a convenience wrapper."""
    if compute_grad:
        beta_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method=method,
            num_random_init=10,
            tol=10**-12,
        )[0]
    else:
        beta_hat = beta
    mu = family.link.inverse(np.dot(X, beta_hat))
    w = w_mu(mu, y, family)
    return w


def dw_drho(
    rho,
    beta,
    y,
    X,
    sm_handler,
    var_list,
    family,
    S_all,
    phi_est,
    compute_grad=False,
    method="Newton-CG",
):
    """Gradient of PIRLS weights wrt rho: dw/drho, shape (M, n).

    Uses the chain rule  dw/drho_h = sum_l (dw/dbeta_l)(dbeta_l/drho_h).
    """
    if compute_grad:
        np.random.seed(4)
        beta_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method=method,
            num_random_init=10,
            tol=10**-14,
        )[0]
    else:
        beta_hat = beta
    mu = family.link.inverse(np.dot(X, beta_hat))
    w_prime = w_deriv(mu, y, family)
    dB = dbeta_hat(
        rho, beta_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est
    )
    g_prime = family.link.deriv(mu)
    dw = np.zeros((rho.shape[0], X.shape[0]))
    for h in range(rho.shape[0]):
        for k in range(X.shape[0]):
            dw[h, k] = np.dot(dB[h, :], X[k, :]) * w_prime[k] / g_prime[k]

    return dw

def grad_H_chunked_fused(X, w_prime, dB, phi_est, block_size=256):
    n, p = X.shape
    M = dB.shape[0]

    result = np.empty((M, p, p))

    for start in range(0, M, block_size):
        end = min(start + block_size, M)
        B = end - start

        # compute hv block on the fly:
        # hv[:, r] = sum_l dB[r,l] * w_prime[l,k]
        block_dB = dB[start:end]              # (B, M)
        block_hv = (block_dB @ w_prime).T     # (n, B)

        V = (block_hv[:, :, None] * X[:, None, :])  # (n, B, p)
        V = V.reshape(n, B * p)

        chunk = X.T @ V
        chunk = chunk.reshape(p, B, p).transpose(1, 0, 2)

        result[start:end] = chunk

    return result / phi_est

def grad_H_drho(
    rho, beta, y, X, sm_handler, var_list, family, S_all, phi_est, compute_grad=False,
    dB=None,
):
    """Gradient of the unpenalised Hessian H wrt rho: dH/drho, shape (M, p, p).

    H = X^T diag(w) X / phi.  Differentiating through w gives
        dH[k]/drho_h = X^T diag(dw/drho_h) X / phi.

    Parameters
    ----------
    dB : ndarray, shape (M, p), optional
        Pre-computed d beta_hat / d rho.  If provided, the internal dbeta_hat
        call (and its Vbeta_rho / QR+SVD) is skipped.
    """
    if compute_grad:
        beta_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method="Newton-CG",
            num_random_init=10,
        )[0]
    else:
        beta_hat = beta
    w_prime = dw_dbeta(beta_hat, y, family, X)
    if dB is None:
        dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    if np.isfortran(X):
        X = np.array(X, order="C")
    if np.isfortran(dB):
        dB = np.array(dB, order="C")
    if np.isfortran(w_prime):
        w_prime = np.array(w_prime, order="C")
    grad_H = grad_H_chunked_fused(X, w_prime, dB, phi_est, block_size=256)
    return grad_H


def reml_objective(
    rho,
    y,
    X,
    sm_handler,
    var_list,
    family,
    S_all,
    phi_est,
    return_type="eval_grad_hess",
    omega=1,
    null_dim=None,
    return_intermediates=False,
):
    """Laplace-REML objective and its derivatives wrt the log-smoothing parameters rho.

    Delegates to the scalable standalone functions (laplace_appr_REML,
    grad_laplace_appr_REML, hess_laplace_appr_REML) after a single beta
    optimisation, so beta_hat is never recomputed more than once per call.

    Parameters
    ----------
    rho : ndarray, shape (M,)
        Log smoothing parameters.
    y, X, sm_handler, var_list, family, S_all, phi-_est : standard GAM inputs.
    return_type : {"eval", "eval_grad", "eval_gradhess"}
        Controls which quantities are computed and returned:
        - "eval"            -> REML value only
        - "eval-grad"       -> (REML, grad_REML)
        - "eval-grad-hess"  -> (REML, grad_REML, hess_REML)
    omega : float
        Prior observation weights.
    fix_beta : bool or ndarray
        If an ndarray, use it as beta_hat instead of re-optimising.

    Returns
    -------
    REML : float
        Laplace-approximated REML value (Wood 2017 eq. 6.18).
    grad_REML : ndarray, shape (M,)  -- only when return_type != "eval"
        Gradient of REML wrt rho.
    hess_REML : ndarray, shape (M, M)  -- only when return_type == "eval-grad-hess"
        Hessian of REML wrt rho.  -hess_REML^{-1} is V_rho (eq. 6.30).
    """
    return_type = ReturnType(getattr(ReturnType, return_type))
    b_hat = mle_gradient_bassed_optim(
        rho,
        sm_handler,
        var_list,
        y,
        X,
        family,
        phi_est=phi_est,
        method="BFGS",
        num_random_init=1,
        tol=1e-10,
    )[0]

    ll_penalty = penalty_ll(rho, b_hat, sm_handler, var_list, phi_est)
    ll_unpen = unpenalized_ll(b_hat, y, X, family, phi_est, omega=omega)

    Slam_trans, S_transf = transform_Slam(S_all, rho)

    # Compute Sinv and log|S_λ|+ together in the transformed space.
    # Sinv is reused for gradient add2 and Hessian add2 — avoids two extra
    # Cholesky factorizations that logDet_Slam / hes_logDet_Slam would do.
    lams = np.exp(rho)
    Slam_t = np.einsum("ijk,i->jk", S_transf, lams)
    try:
        Pinv_t = np.diag(1 / np.sqrt(np.abs(np.diag(Slam_t))))
        P_t    = np.diag(np.sqrt(np.abs(np.diag(Slam_t))))
        L_t    = np.linalg.cholesky(np.einsum("ij,jh,hk->ik", Pinv_t, Slam_t, Pinv_t))
        Linv_t = np.linalg.pinv(L_t)
        Sinv   = np.einsum("ij,kj,kh,hl->il", Pinv_t, Linv_t, Linv_t, Pinv_t)
        log_det_Slam_val = (
            2 * np.sum(np.log(np.diag(L_t))) + 2 * np.sum(np.log(np.diag(P_t)))
        )
    except np.linalg.LinAlgError:
        Slam_t = np.triu(Slam_t) + np.triu(Slam_t, 1).T
        d_t, U_t = np.linalg.eigh(Slam_t)
        idx = d_t > np.finfo(float).eps
        Utmp = U_t[:, idx] * (1 / np.sqrt(d_t[idx]))
        Sinv = np.dot(Utmp, Utmp.T)
        log_det_Slam_val = np.sum(np.log(d_t[idx]))

    log_det_Slam = -0.5 * log_det_Slam_val / phi_est

    # log|H+S| comes free from the SVD already done inside Vbeta_rho
    Vb, Vb_inv, log_det_H_plus_S = Vbeta_rho_all(
        rho,
        b_hat,
        y,
        X,
        family,
        sm_handler,
        var_list,
        phi_est,
        return_logdet=True,
    )
    log_det_sum = -0.5 * log_det_H_plus_S

    if null_dim is None:
        M = b_hat.shape[0] - np.linalg.matrix_rank(Slam_trans)
    else:
        M = null_dim

    REML = (
            ll_unpen + ll_penalty - log_det_Slam + log_det_sum + 0.5 * M * np.log(np.pi * 2)
    )
    if return_type == ReturnType.eval:
        return REML

    S_raw    = np.array(S_all)                        # (M, p, p) — computed once
    S_tensor = S_raw * lams[:, None, None]             # (M, p, p)
    dSlam_drho = S_tensor / phi_est                   # (M, p, p) = exp(rho_k)*S_k/phi

    # neg_sum_inv = -(H+S)^{-1} from the already-computed Vbeta_rho_all — no extra QR+SVD
    neg_sum_inv = -Vb_inv

    # dB = d beta_hat / d rho — inline using neg_sum_inv, no dbeta_hat / Vbeta_rho call
    P1 = np.einsum("kij,j->ki", dSlam_drho, b_hat)    # (M, p)
    dB = np.einsum("li,ki->kl", neg_sum_inv, P1)       # (M, p)

    # ── add1 (gradient): -0.5 * beta^T (lam_r * S_r) beta / phi ──────────────
    add1 = -0.5 * np.einsum("i,rij,j->r", b_hat, S_tensor, b_hat) / phi_est

    # ── add2 (gradient): +0.5 * d log|S_lambda|+ / drho / phi ────────────────
    add2 = (
        0.5
        * np.array([lams[j] * inner1d_sum(Sinv, S_transf[j].T) for j in range(len(rho))])
        / phi_est
    )

    # ── add3 (gradient): -0.5 * tr(Vb_inv @ dVb[r]) ─────────────────────────
    # tr(Vb_inv @ dH[r]) = (dB @ q)[r] / phi;  tr(Vb_inv @ lam_r*S_r) = lam_r*tr_VS[r]
    mu = family.link.inverse(np.dot(X, b_hat))
    h_mu = small_h_mu(mu, y, family)  # (n,)

    VX = Vb_inv @ X.T                                 # (p, n) — hoisted, reused below
    diag_XVX = np.sum(X * VX.T, axis=1)              # (n,)
    q = X.T @ (h_mu * diag_XVX)                      # (p,) — hoisted, reused below
    tr_dH = dB @ q / phi_est                          # (M,)
    tr_VS = np.einsum("ij,hji->h", Vb_inv, S_raw) / phi_est  # (M,) — hoisted, reused below

    add3 = -0.5 * (tr_dH + lams * tr_VS)
    grad_REML = add1 + add2 + add3
    if return_type == ReturnType.eval_grad:
        if return_intermediates:
            return REML, grad_REML, dict(b_hat=b_hat, Vb=Vb, Vb_inv=Vb_inv, dB=dB)
        return REML, grad_REML

    # ── Hessian ───────────────────────────────────────────────────────────────

    # ── add1 (Hessian) ────────────────────────────────────────────────────────
    di = np.diag_indices(len(rho))
    add1 = np.einsum("hj,ji,ki->hk", dB, Vb, dB, optimize="optimal")
    add1[di] -= (
        0.5 * np.einsum("i,hij,j->h", b_hat, S_tensor, b_hat, optimize="optimal")
        / phi_est
    )

    # ── add2 (Hessian): reuse Sinv ────────────────────────────────────────────
    Sinv_Sj = [np.einsum("ij,jk->ik", Sinv, S_transf[j]) for j in range(len(rho))]
    hes_det = np.zeros((len(rho), len(rho)))
    for _i in range(len(rho)):
        for _j in range(len(rho)):
            hes_det[_i, _j] = -lams[_i] * lams[_j] * inner1d_sum(Sinv_Sj[_i], Sinv_Sj[_j].T)
            if _i == _j:
                hes_det[_i, _j] += lams[_i] * inner1d_sum(Sinv, S_transf[_i].T)
    add2 = 0.5 * hes_det / phi_est

    # ── add3 (Hessian) — streaming trace ──────────────────────────────────────
    # dH/drho: one grad_H_chunked_fused, passing dB to skip the internal dbeta_hat
    dH_drho = grad_H_drho(
        rho, b_hat, y, X, sm_handler, var_list, family, S_all, phi_est, dB=dB
    )
    dVb = dH_drho + dSlam_drho  # replaces dVb_drho call

    tmp = np.einsum("ij,hjk->hik", Vb_inv, dVb, optimize=True)  # (M, p, p)
    tr_AhAr = np.einsum("hij,rji->hr", tmp, tmp)  # (M, M)

    h_prime = deriv_small_h(mu, y, family)  # (n,)
    g_prime = family.link.deriv(mu)         # (n,)
    tilde_h = h_prime / g_prime             # (n,)

    # d2B: pass pre-computed quantities to skip all internal Vbeta_rho / grad_H calls
    d2B = d2beta_hat(
        rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est,
        dH_drho=dH_drho, neg_sum_inv=neg_sum_inv, grad_beta_precomp=dB,
    )

    # Term 1: reuse A = X @ dB.T, diag_XVX already computed above
    A = X @ dB.T                             # (n, M)
    w1 = tilde_h * diag_XVX                 # (n,)
    tr_d2H_t1 = (A * w1[:, None]).T @ A     # (M, M)

    # Term 2: reuse q and diag_XVX from gradient section
    tr_d2H_t2 = np.einsum("l,hrl->hr", q, d2B)  # (M, M)

    tr_d2H = (tr_d2H_t1 + tr_d2H_t2) / phi_est  # (M, M)

    # Diagonal penalty: reuse tr_VS from gradient section
    tr_d2Vb = tr_d2H.copy()
    tr_d2Vb[di] += lams * tr_VS

    add3 = 0.5 * (tr_AhAr - tr_d2Vb)

    hess_REML = add1 + add2 + add3
    return REML, grad_REML, hess_REML


def hes_H_drho(
    rho, beta_hat, y, X, S_all, sm_handler, var_list, family, phi_est
):
    """Second derivative of the unpenalised Hessian H wrt rho: d2H/drho, shape (M, M, p, p).

    Uses the chain rule applied twice through the PIRLS weights w(beta_hat(rho)).
    """
    mu = family.link.inverse(np.dot(X, beta_hat))
    h_prime = deriv_small_h(mu, y, family)
    h = small_h_mu(mu, y, family)
    g_prime = family.link.deriv(mu)
    g_prime_inv = np.array(1 / g_prime, order="C")
    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    d2B = d2beta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    return _hes_H_blas(X, dB, d2B, h, h_prime, g_prime_inv, phi_est)


def _hes_H_blas(X, dB, d2B, h, h_prime, g_prime_inv, phi_est):
    """BLAS-explicit computation of d2H/drho, shape (M, M, p, p)."""
    n, p = X.shape
    M = dB.shape[0]

    # ---- Term 1 ---------------------------------------------------------
    # part1[h,r,i,j] = Σ_k X[k,i]*X[k,j] * (h'/g')[k] * (X dB[h])_k * (X dB[r])_k
    A = X @ dB.T                                           # (n, M)
    tilde_h = h_prime * g_prime_inv                        # (n,)
    E = (A[:, :, None] * X[:, None, :]).reshape(n, M * p)  # (n, M*p)
    part1 = (
        (E.T @ (tilde_h[:, None] * E))   # (M*p, M*p)
        .reshape(M, p, M, p)
        .transpose(0, 2, 1, 3)           # (M, M, p, p)
    )

    # ---- Term 2 ---------------------------------------------------------
    # part2[h,r,i,j] = Σ_k X[k,i]*X[k,j] * h[k] * (X d2B[h,r])_k
    v = np.tensordot(X, d2B, axes=([1], [2]))  # (n, M, M)
    hv = h[:, None, None] * v                  # (n, M, M)
    part2 = np.empty((M, M, p, p))
    for h_idx in range(M):
        for r_idx in range(M):
            WX = hv[:, h_idx, r_idx, None] * X  # (n, p)
            part2[h_idx, r_idx] = WX.T @ X       # one BLAS dgemm

    return (part1 + part2) / phi_est

def compute_T_matrices(rho, X, y, family, beta_hat, var_list, phi_est, S_all, sm_handler):
    """Compute d2w/drho (second derivative of PIRLS weights wrt rho), shape (M, M, n).

    This is an alternative implementation to hes_w_wrt_rho, kept for reference.
    S_all and sm_handler are required to evaluate dbeta_hat and d2beta_hat.

    Parameters
    ----------
    rho : ndarray, shape (M,)
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,)
    family : d2variance_family
    beta_hat : ndarray, shape (p,)
    var_list : list of str
    phi_est : float
    S_all : list of ndarray
        Raw penalty matrices.
    sm_handler : smooths_handler

    Returns
    -------
    d2w_drho : ndarray, shape (M, M, n)
    """
    FLOAT_EPS = np.finfo(float).eps
    dw_dB = dw_dbeta(beta_hat, y, family, X)
    dB = dbeta_hat(
        rho, beta_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est
    )
    dw_drho_vals = np.einsum("li,rl->ri", dw_dB, dB, optimize="optimal")

    mu = family.link.inverse(np.dot(X, beta_hat))
    w = w_mu(mu, y, family)

    Tj = dw_drho_vals / np.abs(w)
    d2B = d2beta_hat(
        rho, beta_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est
    )
    w_2prime = w_2deriv(mu, y, family)
    w_prime = w_deriv(mu, y, family)

    g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_2prime = family.link.deriv2(mu)

    den = np.clip(g_prime, FLOAT_EPS, np.inf)
    frac1 = (w_2prime * g_prime - w_prime * g_2prime) / den**3
    frac2 = w_prime / den

    d2w_drho = np.zeros((rho.shape[0], rho.shape[0], X.shape[0]))
    for h in range(rho.shape[0]):
        XdBh = X * dB[h, :]
        for r in range(rho.shape[0]):
            XdBr = X * dB[r, :]
            for k in range(X.shape[0]):
                d2w_drho[h, r, k] = (
                    np.dot(XdBh[k, :], XdBr[k, :]) * frac1[k]
                    + np.dot(X[k, :], d2B[h, r, :]) * frac2[k]
                )
    return d2w_drho


def hes_w_wrt_rho(
    rho, beta, y, X, sm_handler, var_list, family, S_all, phi_est, compute_grad=False
):
    """Second derivative of PIRLS weights wrt rho: d2w/drho, shape (M, M, n).

    Used in the Hessian of the REML objective.
    """
    FLOAT_EPS = np.finfo(float).eps
    if compute_grad:
        beta_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method="Newton-CG",
            num_random_init=10,
        )[0]
    else:
        beta_hat = beta

    dB = dbeta_hat(
        rho, beta_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est
    )
    d2B = d2beta_hat(
        rho, beta_hat, S_all, sm_handler, var_list, y, X, family, phi_est=phi_est
    )

    mu = family.link.inverse(np.dot(X, beta_hat))
    w_prime = w_deriv(mu, y, family)
    w_2prime = w_2deriv(mu, y, family)
    g_prime = family.link.deriv(mu)
    g_2prime = np.clip(family.link.deriv2(mu), FLOAT_EPS, np.inf)
    add2 = np.zeros((rho.shape[0], rho.shape[0], X.shape[0]))
    add1 = np.zeros((rho.shape[0], rho.shape[0], X.shape[0]))
    for r in range(rho.shape[0]):
        for h in range(rho.shape[0]):
            for k in range(X.shape[0]):
                dBrX = np.dot(dB[r, :], X[k, :])
                dBhX = np.dot(dB[h, :], X[k, :])
                add2[r, h, k] = np.dot(X[k, :], d2B[r, h, :]) * (
                    w_prime[k] / g_prime[k]
                )
                add1[r, h, k] = (
                    dBrX
                    * dBhX
                    * (w_2prime[k] * g_prime[k] - w_prime[k] * g_2prime[k])
                    / (g_prime[k] ** 3)
                )

    d2w = add1 + add2
    return d2w


def det_H_rho(X, y, family, w, beta_hat, rho, S_all, sm_handler, var_list):
    """Placeholder: intended to compute det(H) as a function of rho.

    Not yet implemented.  The variable assignments below were left from an
    incomplete draft; the function currently does nothing.

    Parameters
    ----------
    X, y, family, w, beta_hat : standard GAM quantities.
    rho : ndarray, shape (M,)
    S_all : list of ndarray
    sm_handler : smooths_handler
    var_list : list of str
    """
    w_prime = dw_dbeta(beta_hat, y, family, X)
    dB = dbeta_hat(rho, beta_hat, S_all, sm_handler, var_list, y, X, family)
    pass


def H_rho(rho, beta, y, X, family, phi_est, sm_handler, var_list, comp_gradient=True):
    """Unpenalised Hessian H = X^T diag(w) X / phi evaluated at beta(rho).

    Parameters
    ----------
    rho : ndarray, shape (M,)
    beta : ndarray, shape (p,)
    y, X, family, phi_est : standard GAM inputs.
    sm_handler : smooths_handler
    var_list : list of str
    comp_gradient : bool
        If True, re-optimise beta at rho before computing H (for testing only).

    Returns
    -------
    H : ndarray, shape (p, p)
    """
    if comp_gradient:
        beta = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method="Newton-CG",
            num_random_init=10,
        )[0]
    mu = family.link.inverse(np.dot(X, beta))
    w = w_mu(mu, y, family)
    H = np.dot(X.T * w, X) / (phi_est)
    return H


def R_rho(rho, b_hat, y, X, family, sm_handler, var_list, phi_est):
    """Cholesky factor R of V_beta = R^T R.

    Returns
    -------
    R : ndarray, shape (p, p)  (upper triangular)
    """
    Vb = -Vbeta_rho(
        rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=False
    )
    R = np.linalg.cholesky(Vb)
    return R


def dVb_drho(
    rho, beta, S_all, y, X, family, sm_handler, var_list, phi_est, compute_grad=False,
    dB=None,
):
    """Gradient of V_beta wrt rho: dV_beta/drho, shape (M, p, p).

    dV_beta/drho_k = dH/drho_k + dS_lambda/drho_k,
    where dS_lambda/drho_k = lambda_k * S_k / phi.

    Parameters
    ----------
    dB : ndarray, shape (M, p), optional
        Pre-computed d beta_hat / d rho.  Passed through to grad_H_drho to
        skip the internal dbeta_hat (and its Vbeta_rho) call.
    """
    if compute_grad:
        b_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method="Newton-CG",
            num_random_init=10,
        )[0]
    else:
        b_hat = beta
    dH = grad_H_drho(rho, b_hat, y, X, sm_handler, var_list, family, S_all, phi_est, dB=dB)
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    dS = (np.exp(rho) * Slam_tensor.T).T / phi_est
    D = dH + dS
    return D


def d2Vb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est):
    """Second derivative of V_beta wrt rho: d2V_beta/drho2, shape (M, M, p, p).

    d2V_beta/(drho_h drho_k) = d2H/(drho_h drho_k) + delta_{hk} * lambda_k * S_k / phi.
    """
    d2Vb = hes_H_drho(rho, b_hat, y, X, S_all, sm_handler, var_list, family, phi_est)
    Slam_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    Slam_tensor[:, :, :] = S_all
    dS = (np.exp(rho) * Slam_tensor.T).T / phi_est
    di1, di2 = np.diag_indices(rho.shape[0])
    d2Vb[di1, di2] = d2Vb[di1, di2] + dS
    return d2Vb


def test_dVb_drho(
    rho, S_all, y, X, family, sm_handler, var_list, phi_est, inverse=False
):
    """Compute V_beta at rho, re-optimising beta.  Used only for numerical testing."""
    beta = mle_gradient_bassed_optim(
        rho,
        sm_handler,
        var_list,
        y,
        X,
        family,
        phi_est=phi_est,
        method="Newton-CG",
        num_random_init=10,
    )[0]
    Vb = -Vbeta_rho(
        rho, beta, y, X, family, sm_handler, var_list, phi_est, inverse=inverse
    )
    return Vb


def grad_cholesky(grad_D_rho, R):
    """Derivative of the Cholesky factor R wrt a scalar parameter.

    Given dD/d(param) and R such that D = R^T R, returns dR/d(param) using
    the upper-triangular recurrence from Appendix B.7 of Wood (2017).

    Parameters
    ----------
    grad_D_rho : ndarray, shape (p, p)
        Symmetric matrix dD/d(param).
    R : ndarray, shape (p, p)
        Upper-triangular Cholesky factor.

    Returns
    -------
    grad_chol : ndarray, shape (p, p)  (upper triangular)
    """
    grad_chol = np.zeros(grad_D_rho.shape)
    for i in range(R.shape[0]):
        Bii = (
            grad_D_rho[i, i]
            - np.dot(grad_chol[:i, i].flatten(), R[:i, i].flatten())
            - np.dot(R[:i, i].flatten(), grad_chol[:i, i].flatten())
        )
        grad_chol[i, i] = 0.5 * Bii / R[i, i]
        for j in range(i + 1, R.shape[0]):
            Bij = (
                grad_D_rho[i, j]
                - np.dot(grad_chol[:i, i].flatten(), R[:i, j].flatten())
                - np.dot(R[:i, i].flatten(), grad_chol[:i, j].flatten())
            )
            grad_chol[i, j] = (Bij - R[i, j] * grad_chol[i, i]) / R[i, i]
    return grad_chol


def grad_chol_Vb_rho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est,
                     Vb_inv=None, dVb=None):
    """Gradient of the Cholesky factor of V_beta wrt rho, shape (M, p, p).

    Needed for the V'' correction term in eq. 6.31 of Wood (2017).

    Parameters
    ----------
    Vb_inv : ndarray, shape (p, p), optional
        Pre-computed (H + S_lambda)^{-1}.  If provided, the Vbeta_rho call is skipped.
    dVb : ndarray, shape (M, p, p), optional
        Pre-computed dH/drho + dSlam/drho.  If provided, the dVb_drho call is skipped.
    """
    if Vb_inv is None:
        Vb_inv = -Vbeta_rho(
            rho, b_hat, y, X, family, sm_handler, var_list, phi_est, inverse=True
        )
    if dVb is None:
        dVb = dVb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
    R = np.array(np.linalg.cholesky(Vb_inv)).T
    dVb_xform = -np.einsum("ij,hjk,kl->hil", Vb_inv, dVb, Vb_inv, optimize="optimal")
    grad_chol_Vb = np.zeros(dVb_xform.shape)
    for j in range(rho.shape[0]):
        grad_chol_Vb[j] = grad_cholesky(dVb_xform[j], R)

    return grad_chol_Vb


def cholesky_Vb_rho(rho, S_all, y, X, family, sm_handler, var_list, phi_est):
    """Cholesky factor of V_beta, re-optimising beta at rho.  For testing."""
    Vb = test_dVb_drho(
        rho, S_all, y, X, family, sm_handler, var_list, phi_est, inverse=True
    )

    R = np.array(np.linalg.cholesky(Vb))
    R = R.T
    return R


def ftest_for_chol_Ax(x):
    """Symmetric 3x3 test matrix A(x) for validating grad_cholesky."""
    A = np.zeros((3, 3))
    A[0, 0] = 1 + x**2 + x**4
    A[0, 1] = 2 * x + x**3
    A[0, 2] = 3 * x**2
    A[1, 1] = 1 + 2 * x**2
    A[1, 2] = x**3 + 2 * x
    A[2, 2] = 1 + x**2 + x**4
    A = A + np.tril(A.T, 1)
    return A


def ftest_for_chol_dA_dx(x):
    """Analytical derivative dA/dx of the test matrix from ftest_for_chol_Ax."""
    A = np.zeros((3, 3))
    A[0, 0] = 2 * x + 4 * x**3
    A[0, 1] = 2 + 3 * x**2
    A[0, 2] = 6 * x
    A[1, 1] = 4 * x
    A[1, 2] = 3 * x**2 + 2
    A[2, 2] = 2 * x + 4 * x**3
    A = A + np.tril(A.T, 1)
    return A


def cholA(x):
    """Cholesky of the test matrix, shaped for use with numerical Jacobian checkers."""
    A = ftest_for_chol_Ax(x)
    A = A.reshape(3, 3)
    B = np.linalg.cholesky(A).T
    return B.reshape((1,) + B.shape)


def gradcholA(x, B):
    """Analytical Cholesky derivative for the test matrix."""
    dA = ftest_for_chol_dA_dx(x)
    dB = grad_cholesky(dA, B[0])
    return dB.reshape((1,) + dB.shape)


def alpha_deriv2(y, mu, family):
    """Second derivative of alpha wrt mu: d2alpha/dmu2."""
    FLOAT_EPS = np.finfo(float).eps
    dy = y - mu
    add1 = (
        family.variance.deriv2(mu) * family.variance(mu)
        - family.variance.deriv(mu) ** 2
    ) / family.variance(mu) ** 2
    add2 = (
        family.link.deriv3(mu) * family.link.deriv(mu) - family.link.deriv2(mu) ** 2
    ) / np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf) ** 2
    term1 = -2 * (add1 + add2)

    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)
    V_3prime = family.variance.deriv3(mu)

    g_prime = np.clip(family.link.deriv(mu), FLOAT_EPS, np.inf)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)
    g_4prime = family.link.deriv4(mu)

    add3 = (V_3prime * V**2 - 3 * V_2prime * V_prime * V + 2 * V_prime**3) / (
        V**3
    )
    add4 = (
        g_4prime * g_prime**2 - 3 * g_3prime * g_2prime * g_prime + 2 * g_2prime**3
    ) / (g_prime**3)
    alpha_2prime = term1 + dy * (add3 + add4)
    return alpha_2prime


def small_h_mu(mu, y, family):
    """Auxiliary scalar h(mu) = (dw/dmu) / g'(mu) used in d2H/drho computations."""
    g_prime = family.link.deriv(mu)
    small_h = w_deriv(mu, y, family) / g_prime
    return small_h


def deriv_small_h(mu, y, family):
    """Derivative of h(mu) wrt mu: dh/dmu.  Used in the Hessian of H wrt rho."""
    V = family.variance(mu)
    V_prime = family.variance.deriv(mu)
    V_2prime = family.variance.deriv2(mu)

    g_prime = family.link.deriv(mu)
    g_2prime = family.link.deriv2(mu)
    g_3prime = family.link.deriv3(mu)

    alpha = alpha_mu(y, mu, family)
    alpha_prime = alpha_deriv(y, mu, family)
    alpha_2prime = alpha_deriv2(y, mu, family)

    term1 = (
        (
            alpha_2prime * g_prime * V
            - alpha_prime * g_2prime * V
            - 2 * alpha * g_3prime * V
            - 3 * alpha * g_2prime * V_prime
            - alpha * V_2prime * g_prime
        )
        * g_prime
        * V
    )
    term2 = -(
        alpha_prime * g_prime * V - alpha * (2 * g_2prime * V + V_prime * g_prime)
    ) * (4 * g_2prime * V + 2 * g_prime * V_prime)
    small_h_prime = (term1 + term2) / (g_prime**5 * V**3)
    return small_h_prime


def laplace_appr_REML(
    rho,
    beta,
    S_all,
    y,
    X,
    family,
    phi_est,
    sm_handler,
    var_list,
    omega=1,
    compute_grad=False,
    fixRand=False,
    method="Newton-CG",
    tol=10**-12,
    num_random_init=1,
    null_dim=None,
):
    """Laplace approximation of the REML (marginal likelihood) at rho.

    Evaluates:
        REML = l(beta_hat) + penalty(beta_hat; rho)
               + 0.5 * log|S_lambda|+
               - 0.5 * log|H + S_lambda|
               + (M/2) * log(2*pi)       [Wood 2017 eq. 6.18, M = null-space dim]

    where M is the dimension of the penalty null space and |.|+ denotes the
    product of non-zero eigenvalues (pseudo-determinant).

    This is the objective that grad_laplace_appr_REML and hess_laplace_appr_REML
    differentiate.  The Hessian of the negative REML wrt rho is V_rho^{-1}
    (eq. 6.30 of Wood 2017).

    Parameters
    ----------
    compute_grad : bool
        If True, re-optimise beta_hat at rho before evaluating.
    fixRand : bool
        Fix numpy random seed before re-optimisation (reproducibility).
    """
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        b_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method=method,
            num_random_init=num_random_init,
            tol=tol,
        )[0]
    else:
        b_hat = beta

    ll_penalty = penalty_ll(rho, b_hat, sm_handler, var_list, phi_est)
    ll_unpen = unpenalized_ll(b_hat, y, X, family, phi_est, omega=omega)

    Slam_trans, S_transf = transform_Slam(S_all, rho)

    log_det_Slam = (
        -0.5 * logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all) / phi_est
    )

    # log|H+S| comes free from the SVD already done inside Vbeta_rho
    _, log_det_H_plus_S = Vbeta_rho(
        rho,
        b_hat,
        y,
        X,
        family,
        sm_handler,
        var_list,
        phi_est,
        inverse=False,
        compute_grad=False,
        return_logdet=True,
    )
    log_det_sum = -0.5 * log_det_H_plus_S

    if null_dim is None:
        M = b_hat.shape[0] - np.linalg.matrix_rank(Slam_trans)
    else:
        M = null_dim
    reml_approx = (
        ll_unpen + ll_penalty - log_det_Slam + log_det_sum + 0.5 * M * np.log(np.pi * 2)
    )
    return reml_approx


def grad_laplace_appr_REML_dense(
    rho,
    beta,
    S_all,
    y,
    X,
    family,
    phi_est,
    sm_handler,
    var_list,
    omega=1,
    compute_grad=False,
    fixRand=False,
    method="Newton-CG",
    num_random_init=1,
    tol=10**-12,
):
    """Gradient of the Laplace-approximated REML wrt rho, shape (M,).

    Three additive terms:
        add1 = -0.5 * beta_hat^T (d S_lambda / d rho) beta_hat / phi
        add2 = +0.5 * d log|S_lambda|+ / d rho / phi
        add3 = -0.5 * tr( V_beta^{-1} (d V_beta / d rho) )

    where V_beta = -(H + S_lambda)^{-1} and d V_beta / d rho = dH/drho + dS/drho.
    """
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        b_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method=method,
            num_random_init=num_random_init,
            tol=tol,
        )[0]
    else:
        b_hat = beta

    lams = np.exp(rho)
    S_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    S_tensor[:, :, :] = S_all
    S_tensor = (S_tensor.T * lams).T

    add1 = -0.5 * np.einsum("i,rij,j->r", b_hat, S_tensor, b_hat) / phi_est

    Slam_trans, S_transf = transform_Slam(S_all, rho)

    add2 = (
        0.5
        * grad_logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all)
        / phi_est
    )

    Vb_inv = -Vbeta_rho(
        rho,
        b_hat,
        y,
        X,
        family,
        sm_handler,
        var_list,
        phi_est,
        inverse=True,
        compute_grad=False,
    )
    dVb = dVb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)

    add3 = np.zeros(add1.shape)
    for j in range(rho.shape[0]):
        add3[j] = -0.5 * inner1d_sum(Vb_inv, dVb[j].T)

    return add1 + add2 + add3


def hess_laplace_appr_REML_dense(
    rho,
    beta,
    S_all,
    y,
    X,
    family,
    phi_est,
    sm_handler,
    var_list,
    compute_grad=False,
    fixRand=False,
    method="Newton-CG",
    num_random_init=1,
    tol=10**-12,
):
    """Hessian of the Laplace-approximated REML wrt rho, shape (M, M).

    Inverting the negative of this matrix gives V_rho (eq. 6.30 of Wood 2017),
    the smoothing-parameter uncertainty needed for the corrected AIC (eq. 6.32)
    and the corrected covariance V'_beta (eq. 6.31).

    Three additive terms:
        add1[h,k] = dB[h]^T V_beta dB[k]  - 0.5 * delta_{hk} * beta^T dSlam[h] beta / phi
        add2      = +0.5 * d2 log|S_lambda|+ / drho2 / phi
        add3[h,k] = 0.5 * tr( (V_beta^{-1} dVb[h])^2 - V_beta^{-1} d2Vb[h,k] )
    """
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        b_hat = mle_gradient_bassed_optim(
            rho,
            sm_handler,
            var_list,
            y,
            X,
            family,
            phi_est=phi_est,
            method=method,
            num_random_init=num_random_init,
            tol=tol,
        )[0]
    else:
        b_hat = beta

    Vb = -Vbeta_rho(
        rho,
        b_hat,
        y,
        X,
        family,
        sm_handler,
        var_list,
        phi_est,
        inverse=False,
        compute_grad=False,
    )
    dB = dbeta_hat(rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est)

    lams = np.exp(rho)
    S_tensor = np.zeros((len(S_all),) + S_all[0].shape)
    S_tensor[:, :, :] = S_all
    S_tensor = (S_tensor.T * lams).T

    add1 = np.einsum("hj,ji,ki->hk", dB, Vb, dB, optimize="optimal")
    di1, di2 = np.diag_indices(rho.shape[0])
    add1[di1, di2] = (
        add1[di1, di2]
        - 0.5
        * np.einsum("i,hij,j->h", b_hat, S_tensor, b_hat, optimize="optimal")
        / phi_est
    )

    Slam_trans, S_transf = transform_Slam(S_all, rho)

    add2 = 0.5 * hes_logDet_Slam(rho, S_transf) / phi_est

    Vb_inv = -Vbeta_rho(
        rho,
        b_hat,
        y,
        X,
        family,
        sm_handler,
        var_list,
        phi_est,
        inverse=True,
        compute_grad=False,
    )
    dVb = dVb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
    d2Vb = d2Vb_drho(rho, b_hat, S_all, y, X, family, sm_handler, var_list, phi_est)
    try:
        add3 = 0.5 * fast_summations.trace_log_det_H_summation_1(Vb_inv, dVb, d2Vb)
    except:
        tmp = np.einsum("ij,hjk->hik", Vb_inv, dVb, optimize=True)
        add3 = 0.5 * (
            np.einsum("hij,rji->hr", tmp, tmp) - np.einsum("ij,hrji->hr", Vb_inv, d2Vb)
        )
    return add1 + add2 + add3


def grad_laplace_appr_REML(
    rho,
    beta,
    S_all,
    y,
    X,
    family,
    phi_est,
    sm_handler,
    var_list,
    omega=1,
    compute_grad=False,
    fixRand=False,
    method="Newton-CG",
    num_random_init=1,
    tol=1e-12,
):
    """Gradient of the Laplace REML wrt rho, shape (M,).

    Avoids materialising the (M, p, p) dVb tensor by computing
    tr(Vb_inv @ dVb[r]) directly via diag(X Vb_inv X^T).

    Memory : O(p·n)   instead of O(M·p²)  (no dVb tensor)
    Compute: O(n·p² + M·p)  instead of O(n·M·p²)

    Agrees with grad_laplace_appr_REML_dense on small problems
    (validated by TestGradLaplaceREMLScalable).
    """
    if compute_grad:
        if fixRand:
            np.random.seed(4)
        b_hat = mle_gradient_bassed_optim(
            rho, sm_handler, var_list, y, X, family,
            phi_est=phi_est, method=method,
            num_random_init=num_random_init, tol=tol,
        )[0]
    else:
        b_hat = beta

    lams = np.exp(rho)
    S_raw    = np.array(S_all)                                     # (M, p, p)
    S_tensor = S_raw * lams[:, None, None]                         # (M, p, p)

    # ── add1: -0.5 * beta^T (lam_r * S_r) beta / phi ─────────────────────────
    add1 = -0.5 * np.einsum("i,rij,j->r", b_hat, S_tensor, b_hat) / phi_est

    # ── add2: +0.5 * d log|S_lambda|+ / drho / phi ────────────────────────────
    Slam_trans, S_transf = transform_Slam(S_all, rho)
    add2 = (
        0.5
        * grad_logDet_Slam(rho, S_transf, compute_grad=False, S_all=S_all)
        / phi_est
    )

    # ── add3: -0.5 * tr(Vb_inv @ dVb[r]) — streaming, no dVb materialised ────
    # dVb[r] = dH[r] + lam_r * S_r / phi
    # tr(Vb_inv @ dH[r])     = (dB @ q)[r] / phi
    # tr(Vb_inv @ lam_r*S_r) = lam_r * tr(Vb_inv @ S_r) / phi
    # where q[l] = Σ_k h_mu[k] * diag_XVX[k] * X[k,l]

    Vb_inv = -np.array(Vbeta_rho(
        rho, b_hat, y, X, family, sm_handler, var_list, phi_est,
        inverse=True, compute_grad=False,
    ))

    mu    = family.link.inverse(np.dot(X, b_hat))
    h_mu  = small_h_mu(mu, y, family)                             # (n,)

    dB = dbeta_hat(rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est)

    VX       = Vb_inv @ X.T                                       # (p, n)
    diag_XVX = np.sum(X * VX.T, axis=1)                          # (n,)

    q     = X.T @ (h_mu * diag_XVX)                              # (p,)
    tr_dH = dB @ q / phi_est                                      # (M,)

    tr_VS = np.einsum("ij,hji->h", Vb_inv, S_raw) / phi_est      # (M,)

    add3 = -0.5 * (tr_dH + lams * tr_VS)

    return add1 + add2 + add3


def hess_laplace_appr_REML(
    rho,
    beta,
    S_all,
    y,
    X,
    family,
    phi_est,
    sm_handler,
    var_list,
    compute_grad=False,
    fixRand=False,
    method="Newton-CG",
    num_random_init=1,
    tol=1e-12,
    return_intermediates=False,
    intermediates=None,
):
    """Hessian of the Laplace REML wrt rho, shape (M, M).

    Avoids materialising the (M, M, p, p) second-derivative tensor d2Vb by
    computing the required trace contractions directly.

    Memory : O(M·p²)   instead of O(M²·p²)
    Compute: O(n·p² + n·M·p + M²·p²)  instead of O(n·M²·p²)

    Agrees with hess_laplace_appr_REML_dense on small problems
    (validated by TestHessLaplaceREMLScalable).

    Parameters
    ----------
    return_intermediates : bool
        If True, return ``(hess, intermediates)`` where ``intermediates`` is a
        dict with keys ``Vb``, ``Vb_inv``, ``dB``, ``dVb``.  These are the
        quantities already computed internally; passing them to downstream
        callers (e.g. ``compute_AIC``) avoids 3 redundant Vbeta_rho calls.
    intermediates : dict or None
        Pre-computed dict with keys ``b_hat``, ``Vb``, ``Vb_inv``, ``dB``
        (as returned by ``reml_objective(..., return_intermediates=True)``).
        When provided, ``mle_gradient_bassed_optim`` and ``Vbeta_rho_all`` are
        skipped.  Used by ``RemlProblem`` to eliminate the redundant recomputation
        that occurs when ``trust-constr`` calls ``fun`` then ``hess`` at the same rho.
    """
    M = len(rho)
    lams = np.exp(rho)
    S_raw      = np.array(S_all)                      # (M, p, p)
    S_tensor   = S_raw * lams[:, None, None]           # (M, p, p)
    dSlam_drho = S_tensor / phi_est                   # (M, p, p)

    if intermediates is not None:
        # Reuse quantities already computed by reml_objective at this rho —
        # skips mle_gradient_bassed_optim and Vbeta_rho_all.
        b_hat  = intermediates["b_hat"]
        Vb     = intermediates["Vb"]
        Vb_inv = intermediates["Vb_inv"]
        dB     = intermediates["dB"]
    else:
        if compute_grad:
            if fixRand:
                np.random.seed(4)
            b_hat = mle_gradient_bassed_optim(
                rho, sm_handler, var_list, y, X, family,
                phi_est=phi_est, method=method,
                num_random_init=num_random_init, tol=tol,
            )[0]
        else:
            b_hat = beta

        # One QR+SVD gives both Vb = (H+S) and Vb_inv = (H+S)^{-1}.
        Vb, Vb_inv = Vbeta_rho_all(
            rho, b_hat, y, X, family, sm_handler, var_list, phi_est
        )
        neg_sum_inv = -Vb_inv
        P1 = np.einsum("kij,j->ki", dSlam_drho, b_hat)   # (M, p)
        dB = np.einsum("li,ki->kl", neg_sum_inv, P1)      # (M, p)

    neg_sum_inv = -Vb_inv

    # ── add1 ──────────────────────────────────────────────────────────────────
    add1 = np.einsum("hj,ji,ki->hk", dB, Vb, dB, optimize="optimal")
    di = np.diag_indices(M)
    add1[di] -= (
        0.5 * np.einsum("i,hij,j->h", b_hat, S_tensor, b_hat, optimize="optimal")
        / phi_est
    )

    # ── add2 ──────────────────────────────────────────────────────────────────
    Slam_trans, S_transf = transform_Slam(S_all, rho)
    add2 = 0.5 * hes_logDet_Slam(rho, S_transf) / phi_est

    # ── add3 — streaming trace, no (M, M, p, p) materialisation ──────────────
    # add3[h,r] = 0.5 * ( tr(A[h]@A[r]) - tr(Vb_inv @ d2Vb[h,r]) )
    # where A[h] = Vb_inv @ dVb[h]   and   d2Vb = d2H + δ_{hr}*lam_h*S_h/phi

    # dH/drho: one grad_H_chunked_fused call; pass dB to skip internal dbeta_hat
    dH_drho = grad_H_drho(
        rho, b_hat, y, X, sm_handler, var_list, family, S_all, phi_est, dB=dB
    )
    # dVb = dH + dS — no dVb_drho call needed
    dVb = dH_drho + dSlam_drho

    # First part: tr( (Vb_inv dVb[h]) (Vb_inv dVb[r]) ) — O(M·p² + M²·p²)
    tmp = np.einsum("ij,hjk->hik", Vb_inv, dVb, optimize=True)    # (M, p, p)
    tmp_flat  = tmp.reshape(tmp.shape[0], -1)
    tmpT_flat = tmp.transpose(0, 2, 1).reshape(tmp.shape[0], -1)
    tr_AhAr   = tmp_flat @ tmpT_flat.T

    # Second part: tr(Vb_inv @ d2Vb[h,r]) via diag(X Vb_inv X^T)
    mu       = family.link.inverse(np.dot(X, b_hat))
    h_mu     = small_h_mu(mu, y, family)                           # (n,)
    h_prime  = deriv_small_h(mu, y, family)                        # (n,)
    g_prime  = family.link.deriv(mu)                               # (n,)
    tilde_h  = h_prime / g_prime                                   # (n,)

    VX        = Vb_inv @ X.T                                       # (p, n)
    diag_XVX  = np.sum(X * VX.T, axis=1)                          # (n,)

    # d2B: pass all pre-computed quantities — skips Vbeta_rho, grad_beta, dH_drho
    d2B = d2beta_hat(                                              # (M, M, p)
        rho, b_hat, S_all, sm_handler, var_list, y, X, family, phi_est,
        dH_drho=dH_drho, neg_sum_inv=neg_sum_inv, grad_beta_precomp=dB,
    )

    # Term 1 of tr(Vb_inv @ d2H): Σ_k tilde_h[k]*diag_XVX[k]*A[k,h]*A[k,r]
    A  = X @ dB.T                                                   # (n, M)
    w1 = tilde_h * diag_XVX                                        # (n,)
    tr_d2H_t1 = (A * w1[:, None]).T @ A                            # (M, M)

    # Term 2 of tr(Vb_inv @ d2H): Σ_{k,l} h[k]*diag_XVX[k]*X[k,l]*d2B[h,r,l]
    q         = X.T @ (h_mu * diag_XVX)                           # (p,)
    tr_d2H_t2 = np.einsum("l,hrl->hr", q, d2B)                    # (M, M)

    tr_d2H = (tr_d2H_t1 + tr_d2H_t2) / phi_est                    # (M, M)

    # Penalty diagonal: δ_{hr} * lam_h/phi * tr(Vb_inv @ S_h)
    tr_VS  = np.einsum("ij,hji->h", Vb_inv, S_raw) / phi_est      # (M,)
    tr_d2Vb       = tr_d2H.copy()
    tr_d2Vb[di]  += lams * tr_VS

    add3 = 0.5 * (tr_AhAr - tr_d2Vb)

    hess = add1 + add2 + add3
    if return_intermediates:
        return hess, dict(Vb=Vb, Vb_inv=Vb_inv, dB=dB, dVb=dVb)
    return hess


class RemlProblem:
    """Pairs reml_objective and hess_laplace_appr_REML with a last-call cache.

    scipy.optimize.minimize(method='trust-constr') evaluates fun(rho) → (f, grad)
    and hess(rho) at the same rho in separate calls.  Without caching this doubles
    mle_gradient_bassed_optim + Vbeta_rho_all.  The cache detects when hess is called
    at the same rho as the preceding eval_grad and injects pre-computed
    (b_hat, Vb, Vb_inv, dB) so those steps are skipped.

    Usage
    -----
    prob = RemlProblem(y, X, sm_handler, var_list, family, S_all, phi_est)
    result = minimize(prob.eval_grad, rho0, jac=True, hess=prob.hess,
                      method="trust-constr", bounds=bounds)
    """

    def __init__(self, y, X, sm_handler, var_list, family, S_all, phi_est,
                 null_dim=None, omega=1):
        self._y        = y
        self._X        = X
        self._sm       = sm_handler
        self._var_list = var_list
        self._family   = family
        self._S_all    = S_all
        self._phi      = phi_est
        self._null_dim = null_dim
        self._omega    = omega
        self._cached_rho = None
        self._cached_itm = None  # dict: b_hat, Vb, Vb_inv, dB

    def eval_grad(self, rho):
        """Return (REML, grad_REML) and populate the intermediate cache."""
        val, grad, itm = reml_objective(
            rho, self._y, self._X, self._sm, self._var_list,
            self._family, self._S_all, self._phi,
            return_type="eval_grad",
            omega=self._omega, null_dim=self._null_dim,
            return_intermediates=True,
        )
        self._cached_rho = rho.copy()
        self._cached_itm = itm
        return val, grad

    def hess(self, rho):
        """Return the REML Hessian, reusing cached intermediates when possible."""
        if (self._cached_rho is not None
                and np.array_equal(rho, self._cached_rho)):
            return hess_laplace_appr_REML(
                rho, None, self._S_all, self._y, self._X,
                self._family, self._phi, self._sm, self._var_list,
                intermediates=self._cached_itm,
            )
        return hess_laplace_appr_REML(
            rho, None, self._S_all, self._y, self._X,
            self._family, self._phi, self._sm, self._var_list,
            compute_grad=True,
        )


def balance_diag_func(rho, s, d):
    """Objective function for balancing diagonal initialisation of smoothing parameters."""
    lam = np.exp(rho)
    lam_s = (lam * s.T).T
    vec = np.zeros(s.shape[0])
    idx_mat = s > np.finfo(float).eps
    for j in range(s.shape[0]):
        idx = idx_mat[j, :]
        vec[j] = np.mean(d[idx] / (d[idx] + lam_s[j, idx])) - 0.4
    return np.sum(vec**2)


def grad_balance_diag_func(rho, s, d):
    """Gradient of balance_diag_func wrt rho."""
    lam = np.exp(rho)
    lam_s = (lam * s.T).T
    grad = np.zeros(lam.shape[0])
    idx_mat = s > np.finfo(float).eps
    for j in range(lam.shape[0]):
        idx = idx_mat[j, :]
        n = np.sum(idx)
        grad[j] = (
            2
            / n**2
            * np.sum(d[idx] / (d[idx] + lam_s[j, idx]) - 0.4)
            * (np.sum(-d[idx] * s[j, idx] * lam[j] / (d[idx] + lam_s[j, idx]) ** 2))
        )
    return grad
