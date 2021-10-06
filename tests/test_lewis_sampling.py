from efficient_probit_regression.datasets import Iris
from efficient_probit_regression.lewis_sampling import lewis_sampling


def test_lewis_iris():
    dataset = Iris()
    X, y = dataset.get_X(), dataset.get_y()

    sample_size = 50
    X_reduced, y_reduced, p = lewis_sampling(X, y, sample_size)

    assert X_reduced.shape == (50, 5)
    assert y_reduced.shape == (50,)

    X_reduced, y_reduced, p = lewis_sampling(X, y, sample_size, fast_approx=True)

    assert X_reduced.shape == (50, 5)
    assert y_reduced.shape == (50,)
