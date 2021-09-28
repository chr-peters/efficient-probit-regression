import numpy as np
from scipy.sparse import diags


def _calculate_lev_score_exact(X):
    Xt = X.T
    XXinv = np.linalg.pinv(Xt.dot(X))
    lev = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        xi = X[i : i + 1, :]
        lev[i] = (xi.dot(XXinv)).dot(xi.T)
    return lev


def _calculate_lewis_weights_exact(X, T=20):
    n = X.shape[0]
    w = np.ones(n)

    for i in range(T):
        Wp = diags(np.power(w, -0.5))
        # Q = qr(Wp.dot(X))
        # s = _calculate_sensitivities_leverage(Q)
        s = _calculate_lev_score_exact(Wp.dot(X))
        w_nxt = np.power(w * s, 0.5)
        # print("|w_t - w_t+1|/|w_t| = ", np.linalg.norm(w - w_nxt) / np.linalg.norm(w))
        w = w_nxt

    return np.array(w + 1.0 / n, dtype=float)


def lewis_sampling(
    X: np.ndarray, y: np.ndarray, sample_size: int, precomputed_weights=None
):
    """
    Returns X_reduced, y_reduced, probabilities
    """
    if precomputed_weights is None:
        s = _calculate_lewis_weights_exact(X)
    else:
        s = precomputed_weights

    # calculate probabilities
    p = s / np.sum(s)

    # draw the sample
    rng = np.random.default_rng()
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False, p=p)

    return X[sample_indices], y[sample_indices], p[sample_indices]
