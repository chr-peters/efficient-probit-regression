import numba
import numpy as np


@numba.jit(nopython=True)
def gaussian_kernel(x_1: np.ndarray, x_2: np.ndarray):
    return np.exp(-0.5 * np.linalg.norm(x_1 - x_2) ** 2)


@numba.jit(nopython=True)
def polynomial_kernel(x_1: np.ndarray, x_2: np.ndarray):
    return np.power(1 + np.dot(x_1, x_2), 2)


@numba.jit(nopython=True)
def mmd(sample_x: np.ndarray, sample_y: np.ndarray, kernel_function=gaussian_kernel):
    """
    Computes an estimate of the maximum mean discrepancy between sample_1 and sample_2.

    See https://arxiv.org/abs/0805.2368 for more info.
    """
    m = sample_x.shape[0]
    n = sample_y.shape[0]

    sum_x = 0
    for i in range(m):
        for j in range(m):
            sum_x += kernel_function(sample_x[i], sample_x[j])

    sum_y = 0
    for i in range(n):
        for j in range(n):
            sum_y += kernel_function(sample_y[i], sample_y[j])

    sum_x_y = 0
    for i in range(m):
        for j in range(n):
            sum_x_y += kernel_function(sample_x[i], sample_y[j])

    return np.sqrt(1 / m ** 2 * sum_x + 1 / n ** 2 * sum_y - 2 / (m * n) * sum_x_y)
