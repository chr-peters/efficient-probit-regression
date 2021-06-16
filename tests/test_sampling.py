import numpy as np
import pytest
from numpy.testing import assert_allclose

from efficient_probit_regression import leverage_score_sampling, uniform_sampling
from efficient_probit_regression.sampling import compute_leverage_scores


def test_uniform_sampling_invalid_shapes():
    X = np.zeros((5, 3))
    y = np.zeros(4)

    with pytest.raises(ValueError):
        uniform_sampling(X, y, sample_size=3)


def test_uniform_sampling_invalid_sample_size():
    X = np.zeros((5, 3))
    y = np.zeros(5)

    with pytest.raises(ValueError):
        uniform_sampling(X, y, sample_size=6)

    with pytest.raises(ValueError):
        uniform_sampling(X, y, sample_size=0)


def test_uniform_sampling():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 4, 5])

    X_sampled, y_sampled = uniform_sampling(X, y, sample_size=3)

    assert X_sampled.shape == (3, 3)
    assert y_sampled.shape == (3,)

    # check if all but one row is in the sample
    rows_in_sample = np.full(4, fill_value=False, dtype=np.bool_)
    for i in range(len(rows_in_sample)):
        cur_row = X[i]
        cur_y = y[i]
        for j in range(3):
            if np.array_equal(X_sampled[j], cur_row) and np.array_equal(
                y_sampled[j], cur_y
            ):
                rows_in_sample[i] = True
                continue

    assert np.sum(rows_in_sample) == 3


def test_leverage_score_sampling():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 4, 5])

    X_sampled, y_sampled, weights = leverage_score_sampling(X, y, sample_size=3)

    assert X_sampled.shape == (3, 3)
    assert y_sampled.shape == (3,)

    # check if all but one row is in the sample
    rows_in_sample = np.full(4, fill_value=False, dtype=np.bool_)
    for i in range(len(rows_in_sample)):
        cur_row = X[i]
        cur_y = y[i]
        for j in range(3):
            if np.array_equal(X_sampled[j], cur_row) and np.array_equal(
                y_sampled[j], cur_y
            ):
                rows_in_sample[i] = True
                continue

    assert np.sum(rows_in_sample) == 3


def test_leverage_score_sampling_augmented():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 4, 5])

    X_sampled, y_sampled, weights = leverage_score_sampling(
        X, y, sample_size=3, augmented=True
    )

    assert X_sampled.shape == (3, 3)
    assert y_sampled.shape == (3,)

    # check if all but one row is in the sample
    rows_in_sample = np.full(4, fill_value=False, dtype=np.bool_)
    for i in range(len(rows_in_sample)):
        cur_row = X[i]
        cur_y = y[i]
        for j in range(3):
            if np.array_equal(X_sampled[j], cur_row) and np.array_equal(
                y_sampled[j], cur_y
            ):
                rows_in_sample[i] = True
                continue

    assert np.sum(rows_in_sample) == 3


def test_compute_leverage_scores_invalid():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        compute_leverage_scores(X)


def test_compute_leverage_scores():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    true_leverage_scores = np.array(
        [
            0.9977175621816176,
            0.7787312305078449,
            0.36405804147365234,
            0.8594931658368866,
        ]
    )

    leverage_scores = compute_leverage_scores(X)

    assert leverage_scores.shape == (4,)
    assert_allclose(leverage_scores, true_leverage_scores)


def test_leverage_scores_indifferent_of_labeling():
    """
    In theory, the leverage scores also depend on the labeling vector y.
    This test confirms, that y doesn't matter and it suffices to compute the
    leverage scores only for X.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, -1, 1, -1])

    # multiply the rows of X by labels in y
    Z = y[:, np.newaxis] * X

    leverage_scores_X = compute_leverage_scores(X)
    leverage_scores_Z = compute_leverage_scores(Z)

    assert_allclose(leverage_scores_X, leverage_scores_Z)
