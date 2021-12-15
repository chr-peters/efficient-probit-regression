import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gennorm


class PGeneralizedProbitModel:
    def __init__(self, p, X: np.ndarray, y: np.ndarray, w: np.ndarray = None):
        if not set(y.astype(int)) == {-1, 1}:
            raise ValueError("Elements of y must be 1 or -1 and can't be all the same!")

        if not X.shape[0] == y.shape[0]:
            raise ValueError(
                f"Shapes don't fit! X shape of {X.shape}"
                f" incompatible to y shape of {y.shape}!"
            )

        if w is None:
            w = np.ones(shape=y.shape)

        self.X = X
        self.y = y
        self.w = w
        self.p = p
        self._params = None

    def negative_log_likelihood(self, params: np.ndarray):
        """Returns negative Likelihood. (revison)"""
        self._check_params(params)

        return np.sum(self.w * _g(-self.y * np.dot(self.X, params), p=self.p))

    def gradient(self, params: np.ndarray):
        self._check_params(params)

        Z = -self.y[:, np.newaxis] * self.X
        grad_vec = self.w * _g_grad(np.dot(Z, params), p=self.p)
        grad_vec = grad_vec[:, np.newaxis]
        return np.sum(Z * grad_vec, axis=0)

    def fit(self):
        """The function fit() fits a model (revision)"""
        def fun(params):
            return self.negative_log_likelihood(params) / self.X.shape[0]

        def jac(params):
            return self.gradient(params) / self.X.shape[0]

        x0 = np.zeros(self.X.shape[1])
        results = minimize(fun=fun, x0=x0, jac=jac, method="BFGS")

        if not results.success:
            # TODO: Find a test that doesn't lead to convergence
            warnings.warn(f"The solver didn't converge! Message: {results.message}")

        self._params = results.x

    def get_params(self):
        """The function get_params() retuns the estimated parameters (revision)"""
        if self._params is None:
            raise AttributeError("Model must be fitted to get params!")
        return self._params

    def _check_params(self, params: np.ndarray):
        if not params.shape[0] == self.X.shape[1]:
            raise ValueError(f"Parameter vector has invalid shape of {params.shape}")


class ProbitModel(PGeneralizedProbitModel):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray = None):
        super().__init__(p=2, X=X, y=y, w=w)


def p_gen_norm_pdf(x, p):
    """Returns the densitiy at a point x for p>= 1."""
    return gennorm.pdf(x, beta=p, scale=np.power(p, 1 / p))


def p_gen_norm_cdf(x, p):
    """Returns the cumulative densitiy until a point x for p>= 1."""
    return gennorm.cdf(x, beta=p, scale=np.power(p, 1 / p))


def _g_orig(z, p):
    return -np.log(p_gen_norm_cdf(-z, p))


def _g_replacement(z, p):
    """Replaces g if z > _CUTOFF_P[p] for numerical stability."""
    return 1 / p * np.power(z, p)


def _g_grad_orig(z, p):
    return p_gen_norm_pdf(z, p) / p_gen_norm_cdf(-z, p)


def _g_grad_replacement(z, p):
    """Replaces g_grad if z > _CUTOFF_P[p] for numerical stability."""
    return np.power(z, p - 1)


# this is the value where _g and _g_grad use the lower tails instead of the
# exact implementation for numerical reasons
def _CUTOFF_P(p):
    if p <= 2:
        return 35
    elif p <= 3:
        return 12
    elif p <= 4:
        return 7
    else:
        return 5


def _G_DIFF_P(p):
    return _g_orig(_CUTOFF_P(p), p) - _g_replacement(_CUTOFF_P(p), p)


def _G_GRAD_DIFF_P(p):
    return _g_grad_orig(_CUTOFF_P(p), p) - _g_grad_replacement(_CUTOFF_P(p), p)


def _g(z: np.ndarray, p):
    results = np.empty(z.shape)
    greater = z > _CUTOFF_P(p)
    results[greater] = _G_DIFF_P(p) + _g_replacement(z[greater], p)
    results[~greater] = _g_orig(z[~greater], p)
    return results


def _g_grad(z: np.ndarray, p):
    results = np.empty(z.shape)
    greater = z > _CUTOFF_P(p)
    results[greater] = _G_GRAD_DIFF_P(p) + _g_grad_replacement(z[greater], p)
    results[~greater] = _g_grad_orig(z[~greater], p)
    return results


class PGeneralizedProbitSGD:
    """
    Stochastic Gradient descent for probit regression.
    Adapts the learning rate in each iteration using inverse scaling.

    :param p: The order of the probit model.
    :param initial_learning_rate: The initial learning rate.
    :param power_t: Inverse scaling is used to adapt
        the learning rate in each iteration.
        The update formula is
        learning_rate = initial_learning_rate / power(cur_iteration, power_t)
    """

    def __init__(
        self, p: int, initial_learning_rate: float = 0.1, power_t: float = 0.5
    ):
        self.p = p
        self.initial_learning_rate = initial_learning_rate
        self.power_t = power_t
        self.cur_iteration = 0

        self._params = None

    def get_params(self):

        """The function get_params() returns the estimated parameters. (revision)"""
        if self._params is None:
            raise AttributeError("Model must be fitted to get params!")
        return self._params

    def new_sample(self, x: np.ndarray, y: int):
        """
        Performs one step of SGD on a new sample x, y.
        """
        self.cur_iteration += 1
        x = x.flatten()
        d = x.shape[0]
        if self._params is None:
            self._params = np.zeros(d)

        z = -y * x
        grad = z * _g_grad(np.dot(z, self._params), p=self.p)
        cur_learning_rate = self.initial_learning_rate / np.power(
            self.cur_iteration, self.power_t
        )

        self._params -= cur_learning_rate * grad


class ProbitSGD(PGeneralizedProbitSGD):
    def __init__(self, initial_learning_rate: float = 0.1, power_t: float = 0.5):
        super().__init__(
            p=2, initial_learning_rate=initial_learning_rate, power_t=power_t
        )
