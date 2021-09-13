import numpy as np
import pytest
from scipy.stats import multivariate_normal

from efficient_probit_regression.metrics import gaussian_kernel, mmd, polynomial_kernel


@pytest.mark.parametrize("kernel_function", [gaussian_kernel, polynomial_kernel])
def test_mmd(kernel_function):
    seed = 1

    # equal distributions
    sample_1 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )
    sample_2 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )

    mmd_value = mmd(sample_1, sample_2, kernel_function=kernel_function)
    print(mmd_value)

    assert mmd_value <= 0.0001

    # different means
    sample_1 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )
    sample_2 = multivariate_normal.rvs(
        mean=np.array([2, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )

    mmd_value = mmd(sample_1, sample_2, kernel_function=kernel_function)
    print(mmd_value)

    assert mmd_value > 0.1

    # different covariances
    sample_1 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=np.eye(3), size=1000, random_state=seed
    )
    sample_2 = multivariate_normal.rvs(
        mean=np.array([1, 2, 3]), cov=2 * np.eye(3), size=1000, random_state=seed
    )

    mmd_value = mmd(sample_1, sample_2, kernel_function=kernel_function)
    print(mmd_value)

    assert mmd_value > 0.1
