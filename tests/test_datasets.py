import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from efficient_probit_regression.datasets import BaseDataset

X = np.array([[1, 0], [0.1, 1], [-0.1, 1], [-1, 0], [0, -1]])
X_intercept = np.array([[1, 0, 1], [0.1, 1, 1], [-0.1, 1, 1], [-1, 0, 1], [0, -1, 1]])
y = np.array([1, -1, -1, 1, -1])
beta_opt = np.array([0.0, -0.4307273])
beta_opt_intercept = np.array([0.0, -0.39382684, -0.19691342])


class ExampleDataset(BaseDataset):
    def __init__(self, add_intercept, use_caching, cache_dir=None):
        super().__init__(
            add_intercept=add_intercept, use_caching=use_caching, cache_dir=cache_dir
        )

    def get_name(self):
        return "example_name"

    def load_X_y(self):
        return X, y


def test_base_dataset_no_caching_no_intercept():
    """
    Use a small toy example to test the abstract dataset base class.
    """
    dataset = ExampleDataset(add_intercept=False, use_caching=False)
    assert dataset.get_name() == "example_name"
    assert_array_equal(dataset.get_X(), X)
    assert_array_equal(dataset.get_y(), y)
    assert_array_almost_equal(dataset.get_beta_opt(p=2), beta_opt, decimal=4)
    assert dataset.get_n() == 5
    assert dataset.get_d() == 2


def test_base_dataset_no_caching_with_intercept():
    dataset = ExampleDataset(add_intercept=True, use_caching=False)
    assert_array_equal(dataset.get_X(), X_intercept)
    assert_array_equal(dataset.get_y(), y)
    assert_array_almost_equal(dataset.get_beta_opt(p=2), beta_opt_intercept, decimal=4)
    assert dataset.get_n() == 5
    assert dataset.get_d() == 3


def test_base_dataset_caching(tmp_path):
    """
    Use a small toy example to test the abstract dataset base class.
    """

    # run the tests twice to simulate a cache hit
    for i in range(2):
        dataset = ExampleDataset(
            add_intercept=True, use_caching=True, cache_dir=tmp_path
        )
        assert dataset.get_name() == "example_name"
        assert_array_equal(dataset.get_X(), X_intercept)
        assert_array_equal(dataset.get_y(), y)
        assert_array_almost_equal(
            dataset.get_beta_opt(p=2), beta_opt_intercept, decimal=4
        )
        assert dataset.get_n() == 5
        assert dataset.get_d() == 3
