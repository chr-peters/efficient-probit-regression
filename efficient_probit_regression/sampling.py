import numpy as np

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


def compute_leverage_scores(X: np.ndarray):
    if not len(X.shape) == 2:
        raise ValueError("X must be 2D!")

    Q, *_ = np.linalg.qr(X)
    leverage_scores = np.linalg.norm(Q, axis=1) ** 2

    return leverage_scores


def compute_leverage_scores_online(X: np.ndarray):
    if not len(X.shape) == 2:
        raise ValueError("X must be 2D!")

    n = X.shape[0]
    d = X.shape[1]
    ATA = np.zeros(shape=(d, d))
    leverage_scores = []
    for i in range(n):
        cur_column_vec = X[i][np.newaxis, :]
        ATA += cur_column_vec.T @ cur_column_vec
        try:
            cur_leverage_score = np.dot(X[i], np.linalg.solve(ATA, X[i]))
        except np.linalg.LinAlgError:
            # singular matrix, use least squares
            cur_leverage_score = np.dot(X[i], np.linalg.lstsq(ATA, X[i], rcond=None)[0])
        leverage_scores.append(cur_leverage_score)

    return np.array(leverage_scores)


def leverage_score_sampling(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    augmented: bool = False,
    online: bool = False,
    precomputed_scores: np.ndarray = None,
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
            leverage_scores = compute_leverage_scores(X)
    else:
        leverage_scores = precomputed_scores

    if augmented:
        leverage_scores = leverage_scores + 1 / X.shape[0]

    p = leverage_scores / np.sum(leverage_scores)

    # w = 1 / (p * sample_size)
    w = np.ones(y.shape)

    sample_indices = _rng.choice(
        X.shape[0],
        size=sample_size,
        replace=False,
        p=p,
    )

    return X[sample_indices], y[sample_indices], w[sample_indices]
