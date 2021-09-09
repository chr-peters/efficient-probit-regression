import numpy as np
from scipy.stats import multivariate_normal

from efficient_probit_regression.metrics import mmd


def test_mmd():
    seed = 1

    # equal distributions
    sample_1 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )
    sample_2 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )

    mmd_value = mmd(sample_1, sample_2)

    assert mmd_value <= 0.0001

    # different means
    sample_1 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )
    sample_2 = multivariate_normal.rvs(
        mean=np.array([2, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )

    mmd_value = mmd(sample_1, sample_2)

    assert mmd_value > 0.1

    # different covariances
    sample_1 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )
    sample_2 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=2 * np.eye(3), size=1000, random_state=seed
    )

    mmd_value = mmd(sample_1, sample_2)

    assert mmd_value > 0.1
