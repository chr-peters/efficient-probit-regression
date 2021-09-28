import numba
import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from scipy.stats import expon, multivariate_normal, norm, truncnorm
from tqdm import tqdm

_rng = np.random.default_rng()


def _check_sample(X, y, sample_size):
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Incompatible shapes of X and y: {X.shape[0]} != {y.shape[0]}"
        )

    if sample_size > X.shape[0]:
        raise ValueError("Sample size can't be greater than total number of samples!")

    if sample_size <= 0:
        raise ValueError("Sample size must be greater than zero!")


def uniform_sampling(X: np.ndarray, y: np.ndarray, sample_size: int):
    """
    Draw a uniform sample of X and y without replacement.

    Returns
    -------
    X, y : Sample
    """
    _check_sample(X, y, sample_size)

    sample_indices = _rng.choice(X.shape[0], size=sample_size, replace=False)

    return X[sample_indices], y[sample_indices]


def fast_QR(X, k=1, add_exponential=False, p=2):
    """
    Returns Q of a fast QR decomposition of X.
    """
    n, d = X.shape
    f = np.random.randint(d ** 2, size=n)
    g = np.random.randint(2, size=n) * 2 - 1

    # init the sketch
    X_ = np.zeros((d ** 2, d))
    if not add_exponential:
        for i in range(n):
            X_[f[i]] += g[i] * X[i]
    else:
        for i in range(n):
            lamb = expon.rvs()
            X_[f[i]] += g[i] / np.power(lamb, 1 / p) * X[i]

    R_ = np.linalg.qr(X_, mode="r")
    R_inv = np.linalg.inv(R_)

    n, d = R_inv.shape
    g = np.random.normal(loc=0, scale=1 / np.sqrt(k), size=(d, k))
    r = np.dot(R_inv, g)
    Q_ = np.dot(X, r)

    return Q_


def compute_leverage_scores(
    X: np.ndarray, p=2, fast_approx=False, add_exponential=False
):
    if not len(X.shape) == 2:
        raise ValueError("X must be 2D!")

    if not fast_approx:
        Q, *_ = np.linalg.qr(X)
    else:
        Q = fast_QR(X, add_exponential=add_exponential, p=p)

    leverage_scores = np.power(np.linalg.norm(Q, axis=1, ord=p), p)

    return leverage_scores


@numba.jit(nopython=True)
def _check_norm_change(Q, x):
    Q = np.ascontiguousarray(Q)
    x = np.ascontiguousarray(x)
    return np.abs(np.linalg.norm(Q.T @ x) - np.linalg.norm(x)) < 1e-6


@numba.jit(nopython=True)
def _fast_inv_update(M_inv, outer, x):
    M_inv = np.ascontiguousarray(M_inv)
    outer = np.ascontiguousarray(outer)
    x = np.ascontiguousarray(x)
    scalar = 1 + np.dot(x, M_inv @ x)
    M_inv -= 1 / scalar * (M_inv @ outer @ M_inv)


def _compute_leverage_scores_online_pinv(X: np.ndarray):
    n = X.shape[0]
    d = X.shape[1]

    M = np.zeros(shape=(d, d))
    M_inv = np.zeros(shape=(d, d))
    Q = np.zeros(shape=(d, d))

    X = X.astype(float)

    leverage_scores = []

    for i in range(n):
        cur_row = X[i]
        outer = np.outer(cur_row, cur_row)
        M += outer
        if _check_norm_change(Q, cur_row):
            _fast_inv_update(M_inv, outer, cur_row)
        else:
            M_inv = np.linalg.pinv(M)
            Q = sp.linalg.orth(M)
            r = Q.shape[1]
            if r < d:
                Q = np.concatenate((Q, np.zeros((d, d - r))), axis=1)

        cur_leverage_score = np.dot(cur_row, M_inv @ cur_row)
        cur_leverage_score = np.minimum(cur_leverage_score, 1)
        cur_leverage_score = np.maximum(cur_leverage_score, 0)
        leverage_scores.append(cur_leverage_score)

    return np.array(leverage_scores)


def _compute_leverage_scores_online_solve(X: np.ndarray):
    n = X.shape[0]
    d = X.shape[1]

    ATA = np.zeros(shape=(d, d))

    leverage_scores = []

    for i in range(n):
        cur_row = X[i]
        ATA += np.outer(cur_row, cur_row)
        try:
            cur_leverage_score = np.dot(cur_row, np.linalg.solve(ATA, cur_row))
            if cur_leverage_score < 0:
                cur_leverage_score = np.dot(
                    cur_row, np.linalg.lstsq(ATA, cur_row, rcond=None)[0]
                )
        except np.linalg.LinAlgError:
            cur_leverage_score = np.dot(
                cur_row, np.linalg.lstsq(ATA, cur_row, rcond=None)[0]
            )
        cur_leverage_score = np.minimum(cur_leverage_score, 1)
        leverage_scores.append(cur_leverage_score)

    return np.array(leverage_scores)


def compute_leverage_scores_online(X: np.ndarray, method="pinv"):
    if method == "pinv":
        return _compute_leverage_scores_online_pinv(X)
    elif method == "solve":
        return _compute_leverage_scores_online_solve(X)
    else:
        raise ValueError("Method must be either pinv or solve!")


def _round_up(x: np.ndarray) -> np.ndarray:
    """
    Rounds each element in x up to the nearest power of two.
    """
    if not np.all(x >= 0):
        raise ValueError("All elements of x must be greater than zero!")

    greater_zero = x > 0

    results = x.copy()
    results[greater_zero] = np.power(2, np.ceil(np.log2(x[greater_zero])))

    return results


def leverage_score_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    augmented: bool = False,
    online: bool = False,
    round_up: bool = False,
    precomputed_scores: np.ndarray = None,
    p=2,
    fast_approx=False,
    add_exponential=False,
):
    """
    Draw a leverage score weighted sample of X and y without replacement.

    Parameters
    ----------
    X : np.ndarray
        Data Matrix
    y : np.ndarray
        Label vector
    sample_size : int
        Sample size
    augmented : bool
        Wether to add the additive 1 / |W| term
    online : bool
        Compute online leverage scores in one pass over the data
    round_up : bool
        Round the leverage scores up to the nearest power of two
    precomputed_scores : np.ndarray
        To avoid recomputing the leverage scores every time,
        pass the precomputed scores here.

    Returns
    -------
    X, y : Sample
    w : New sample weights
    """
    _check_sample(X, y, sample_size)

    if precomputed_scores is None:
        if online:
            leverage_scores = compute_leverage_scores_online(X)
        else:
            leverage_scores = compute_leverage_scores(
                X, p=p, fast_approx=fast_approx, add_exponential=add_exponential
            )
    else:
        leverage_scores = precomputed_scores

    if augmented:
        leverage_scores = leverage_scores + 1 / X.shape[0]

    if round_up:
        leverage_scores = _round_up(leverage_scores)

    p = leverage_scores / np.sum(leverage_scores)

    w = 1 / (p * sample_size)

    sample_indices = _rng.choice(
        X.shape[0],
        size=sample_size,
        replace=False,
        p=p,
    )

    return X[sample_indices], y[sample_indices], w[sample_indices]


@numba.jit(nopython=True)
def _fast_leverage_score(row, A):
    return np.dot(
        np.ascontiguousarray(row), np.ascontiguousarray(np.linalg.solve(A, row))
    )


def online_ridge_leverage_score_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    augmentation_constant: float = None,
    lambda_ridge: float = 1e-6,
):
    """
    Sample X and y proportional to the online ridge leverage scores.
    """
    n, d = X.shape

    ATA_ridge = lambda_ridge * np.eye(d)

    sampler = ReservoirSampler(sample_size=sample_size, d=d)

    # the remaining samples
    for i in range(n):
        cur_row = X[i]
        cur_label = y[i]

        cur_ridge_leverage_score = _fast_leverage_score(cur_row, ATA_ridge)
        cur_weight = np.minimum(cur_ridge_leverage_score, 1)

        if augmentation_constant is not None:
            cur_weight += augmentation_constant

        sampler.insert_record(row=cur_row, label=cur_label, weight=cur_weight)

        if sampler.was_last_record_sampled():
            ATA_ridge += cur_row[:, np.newaxis] @ cur_row[np.newaxis, :]

    X_sample, y_sample = sampler.get_sample()
    return X_sample, y_sample, np.ones(y_sample.shape)


class ReservoirSampler:
    """
    Implementation of a reservoir sampler as described in
    "A general purpose unequal probability sampling plan" by M. T. Chao,
    adapted here for row sampling of datasets consisting of a data matrix X
    and a label vector y.

    Parameters
    ----------
    sample_size : int
        Numer of rows in the resulting sample.

    d : int
        Second dimension of the sample.
        The whole sample will have a dimension of sample_size x d.
    """

    def __init__(self, sample_size: int, d: int):
        self.sample_size = sample_size
        self.d = d
        self._sample_X = np.empty(shape=(sample_size, d))
        self._sample_y = np.empty(shape=(sample_size,))
        self._row_counter = 0
        self._weight_sum = 0
        self._last_record_sampled = False

    def get_sample(self):
        """
        Returns the sample of X and the sample of y.
        """
        if self._row_counter < self.sample_size:
            return (
                self._sample_X[: self._row_counter],
                self._sample_y[: self._row_counter],
            )
        return self._sample_X, self._sample_y

    def insert_record(self, row: np.ndarray, label: float, weight: float):
        """
        Insert a data record consisting of a row and a label.
        The record will be sampled with a probability that is proportional to
        the given weight.
        """
        self._weight_sum += weight

        if self._row_counter < self.sample_size:
            self._sample_X[self._row_counter] = row
            self._sample_y[self._row_counter] = label
            self._row_counter += 1
            self._last_record_sampled = True
            return

        p = self.sample_size * weight / self._weight_sum
        if _rng.random() < p:
            random_index = _rng.choice(self.sample_size)
            self._sample_X[random_index] = row
            self._sample_y[random_index] = label
            self._row_counter += 1
            self._last_record_sampled = True
            return

        self._last_record_sampled = False

    def was_last_record_sampled(self):
        return self._last_record_sampled


def truncated_normal_rejection(
    a: np.ndarray, b: np.ndarray, mean: np.ndarray, std: np.ndarray, size
):
    if not type(mean) == np.ndarray:
        mean = np.full(shape=size, fill_value=mean)
    if not type(std) == np.ndarray:
        std = np.full(shape=size, fill_value=std)

    sample = norm.rvs(loc=mean, scale=std, size=size)
    not_in_sample = (sample < a) | (sample > b)
    while not np.all(~not_in_sample):
        sample[not_in_sample] = norm.rvs(
            loc=mean[not_in_sample],
            scale=std[not_in_sample],
            size=np.sum(not_in_sample),
        )
        not_in_sample = (sample < a) | (sample > b)
    return sample


def truncated_normal(
    a: np.ndarray,
    b: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    size,
    random_state=None,
):
    """
    Use rejection sampling if the interval [a, b] covers at least a probability
    mass of 5%.
    Otherwise use the implementation given in scipy.stats.truncnorm.

    The parameters a and b specify the actual interval where the
    probability mass is located, mean and std specify the
    original normal distribution.
    """
    if not type(mean) == np.ndarray:
        mean = np.full(shape=size, fill_value=mean)
    if not type(std) == np.ndarray:
        std = np.full(shape=size, fill_value=std)
    if not type(a) == np.ndarray:
        a = np.full(shape=size, fill_value=a)
    if not type(b) == np.ndarray:
        b = np.full(shape=size, fill_value=b)

    a_cdf = norm.cdf(a, loc=mean, scale=std)
    b_cdf = norm.cdf(b, loc=mean, scale=std)
    rejection_ok = (b_cdf - a_cdf) > 0.05
    size_rejection = np.sum(rejection_ok)
    sample = np.zeros(size)
    sample[rejection_ok] = truncated_normal_rejection(
        a[rejection_ok],
        b[rejection_ok],
        mean[rejection_ok],
        std[rejection_ok],
        size=size_rejection,
    )

    a_scipy = (a - mean) / std
    b_scipy = (b - mean) / std
    sample[~rejection_ok] = truncnorm.rvs(
        a=a_scipy[~rejection_ok],
        b=b_scipy[~rejection_ok],
        loc=mean[~rejection_ok],
        scale=std[~rejection_ok],
        size=size - size_rejection,
        random_state=random_state,
    )

    return sample


def gibbs_sampler_probit(
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    num_samples,
    num_chains,
    burn_in=100,
    probabilities=None,
):
    n, d = X.shape

    if probabilities is None:
        probabilities = np.full(n, 1 / n)

    factor_squared = 1 / (probabilities * n)

    prior_cov_inv = np.linalg.inv(prior_cov)
    B = np.linalg.inv(
        prior_cov_inv + X.T @ np.multiply(X, factor_squared[:, np.newaxis])
    )

    beta_start = np.zeros(d)  # TODO: set this to the MLE

    def progress_if_not_parallel(input):
        if num_chains == 1:
            return tqdm(input)
        else:
            return input

    def simulate_chain():
        beta = beta_start
        samples = []
        for i in progress_if_not_parallel(range(num_samples + burn_in)):
            a = np.where(y == -1, -np.inf, 0)
            b = np.where(y == -1, 0, np.inf)

            # sample latent variables
            latent_mean = X @ beta
            latent = truncated_normal(
                a,
                b,
                mean=latent_mean,
                std=1,
                size=n,
            )

            beta_mean = B @ (
                prior_cov_inv @ prior_mean + X.T @ (latent * factor_squared)
            )
            beta = multivariate_normal.rvs(size=1, mean=beta_mean, cov=B)

            samples.append(beta)

        return np.array(samples[burn_in:])

    if num_chains == 1:
        samples = simulate_chain()
    else:
        sample_chunks = Parallel(n_jobs=num_chains)(
            delayed(simulate_chain)() for i in range(num_chains)
        )
        samples = np.vstack(sample_chunks)

    return samples
