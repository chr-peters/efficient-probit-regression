"""
This tests the probit model using the statsmodels implementation as a reference.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris, make_classification
from statsmodels.discrete.discrete_model import Probit as Probit_statsmodels

from efficient_probit_regression import ProbitModel

X = np.array([[1, 2], [3, 4]])
y = np.array([1, -1])


@pytest.fixture
def iris():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_valid_init():
    ProbitModel(X, y)


def test_invalid_shapes():
    y = np.array([1, -1, 1])

    with pytest.raises(ValueError):
        ProbitModel(X, y)


def test_invalid_y_labels():
    y = np.array([1, 0])

    with pytest.raises(ValueError):
        ProbitModel(X, y)


def test_invalid_param_shape_in_likelihood():
    model = ProbitModel(X, y)
    params = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        model.negative_log_likelihood(params)


def test_negative_log_likelihood(iris):
    X, y = iris
    y_0_1 = np.where(y == 1, 1, 0)
    y_m1_1 = np.where(y == 1, 1, -1)

    model_statsmodels = Probit_statsmodels(y_0_1, X)
    results = model_statsmodels.fit()
    params_opt = results.params

    model = ProbitModel(X=X, y=y_m1_1)

    assert_allclose(
        -model_statsmodels.loglike(params_opt),
        model.negative_log_likelihood(params_opt),
    )


def test_gradient_shape(iris):
    X, y = iris
    y_m1_1 = np.where(y == 1, 1, -1)
    model = ProbitModel(X=X, y=y_m1_1)

    params = np.zeros(X.shape[1])

    grad = model.gradient(params)

    assert grad.shape == params.shape


def test_gradient(iris):
    X, y = iris
    y_0_1 = np.where(y == 1, 1, 0)
    y_m1_1 = np.where(y == 1, 1, -1)

    model_statsmodels = Probit_statsmodels(y_0_1, X)
    model = ProbitModel(X=X, y=y_m1_1)

    params = np.zeros(X.shape[1])

    assert_allclose(
        -model_statsmodels.score(params),
        model.gradient(params),
    )


def test_invalid_param_shape_in_gradient():
    model = ProbitModel(X, y)
    params = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        model.gradient(params)


def test_fit_iris(iris):
    X, y = iris
    y_0_1 = np.where(y == 1, 1, 0)
    y_m1_1 = np.where(y == 1, 1, -1)

    model_statsmodels = Probit_statsmodels(y_0_1, X)
    statsmodels_result = model_statsmodels.fit()

    model = ProbitModel(X=X, y=y_m1_1)
    model.fit()

    assert_allclose(statsmodels_result.params, model.get_params(), rtol=1e-5)


def test_fit_iris_scaled_weights(iris):
    X, y = iris
    y_0_1 = np.where(y == 1, 1, 0)
    y_m1_1 = np.where(y == 1, 1, -1)

    model_statsmodels = Probit_statsmodels(y_0_1, X)
    statsmodels_result = model_statsmodels.fit()

    model = ProbitModel(X=X, y=y_m1_1, w=3 * np.ones(shape=y.shape))
    model.fit()

    assert_allclose(statsmodels_result.params, model.get_params(), rtol=1e-5)


def test_params_not_fitted():
    model = ProbitModel(X, y)

    with pytest.raises(AttributeError):
        model.get_params()


def test_big_outliers():
    X = np.array([[1000, 1], [900, 1]])
    y = np.array([1, -1])

    model = ProbitModel(X, y)

    with pytest.warns(None) as record:
        model.fit()

    # assert that no warnings were issued
    assert len(record) == 0

    # assert that the predicton results are correct
    preds_raw = np.dot(X, model.get_params())
    assert preds_raw[0] > 0
    assert preds_raw[1] < 0


def test_heavy_hitters_convergence():
    """
    This test asserts that the solver still converges without a warning
    even though there are some heavy hitters in the dataset.
    """
    X, y = make_classification(
        n_samples=1000, n_features=10, n_clusters_per_class=1, random_state=1
    )
    y = np.where(y == 0, -1, 1)

    # add the heavy hitters
    rng = np.random.default_rng(seed=1)
    num_heavy_hitters = 10
    random_rows = rng.choice(X.shape[0], size=num_heavy_hitters, replace=False)
    X[random_rows] += rng.normal(
        loc=0, scale=1000, size=(num_heavy_hitters, X.shape[1])
    )

    model = ProbitModel(X, y)

    with pytest.warns(None) as record:
        model.fit()

    assert len(record) == 0
