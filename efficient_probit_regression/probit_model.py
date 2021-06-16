import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class ProbitModel:
    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray = None):
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
        self._params = None

    def negative_log_likelihood(self, params: np.ndarray):
        self._check_params(params)

        return np.sum(self.w * _g(-self.y * np.dot(self.X, params)))

    def gradient(self, params: np.ndarray):
        self._check_params(params)

        Z = -self.y[:, np.newaxis] * self.X
        grad_vec = self.w * _g_grad(np.dot(Z, params))
        grad_vec = grad_vec[:, np.newaxis]
        return np.sum(Z * grad_vec, axis=0)

    def fit(self):
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
        if self._params is None:
            raise AttributeError("Model must be fitted to get params!")
        return self._params

    def _check_params(self, params: np.ndarray):
        if not params.shape[0] == self.X.shape[1]:
            raise ValueError(f"Parameter vector has invalid shape of {params.shape}")


def _g_orig(z):
    return -np.log(norm.cdf(-z))


def _g_replacement(z):
    """Replaces g if z > _CUTOFF for numerical stability."""
    return 0.5 * z ** 2


def _g_grad_orig(z):
    return norm.pdf(z) / norm.cdf(-z)


# this is the value where _g and _g_grad use the lower tails instead of the
# exact implementation for numerical reasons
_CUTOFF = 35
_G_DIFF = _g_orig(_CUTOFF) - _g_replacement(_CUTOFF)
_G_GRAD_DIFF = _g_grad_orig(_CUTOFF) - _CUTOFF


def _g(z: np.ndarray):
    results = np.empty(z.shape)
    greater = z > _CUTOFF
    results[greater] = _G_DIFF + _g_replacement(z[greater])
    results[~greater] = _g_orig(z[~greater])
    return results


def _g_grad(z: np.ndarray):
    results = np.empty(z.shape)
    greater = z > _CUTOFF
    results[greater] = _G_GRAD_DIFF + z[greater]
    results[~greater] = _g_grad_orig(z[~greater])
    return results
