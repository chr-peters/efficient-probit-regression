import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from efficient_probit_regression.datasets import BaseDataset
from efficient_probit_regression.experiments import (
    LeverageScoreSamplingExperiment,
    UniformSamplingExperiment,
)


class ExampleDataset(BaseDataset):
    def __init__(self):
        super().__init__(use_caching=False)

    def get_name(self):
        return "example_name"

    def load_X_y(self):
        X = np.array([[1, 0], [0.1, 1], [-0.1, 1], [-1, 0], [0, -1]])
        y = np.array([1, -1, -1, 1, -1])
        return X, y


def test_uniform_sampling_experiment(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = UniformSamplingExperiment(
        dataset=dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=2,
        num_runs=3,
    )
    experiment.run()

    df = pd.read_csv(results_filename)

    run_unique, run_counts = np.unique(df["run"], return_counts=True)
    assert_array_equal(run_unique, [1, 2, 3])
    assert_array_equal(run_counts, [3, 3, 3])

    assert np.all(df["ratio"][~df["ratio"].isna()] >= 1)

    assert np.sum(df["sampling_time_s"].isna()) == 0
    assert np.sum(df["total_time_s"].isna()) == 0

    assert np.all(df["sampling_time_s"] > 0)
    assert np.all(df["total_time_s"] > 0)


def test_uniform_sampling_reduction(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = UniformSamplingExperiment(
        dataset=dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=1,
        num_runs=1,
    )

    for cur_config in experiment.get_config_grid():
        cur_X, cur_y, cur_weights = experiment.get_reduced_X_y_weights(cur_config)
        assert_array_equal(cur_weights, np.ones(cur_config["size"]))
        assert cur_X.shape[0] == cur_config["size"]
        assert cur_X.shape[1] == dataset.get_d()
        assert cur_y.shape[0] == cur_config["size"]


def test_leverage_score_sampling_experiment(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = LeverageScoreSamplingExperiment(
        dataset=dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=2,
        num_runs=3,
    )
    experiment.run()

    df = pd.read_csv(results_filename)

    run_unique, run_counts = np.unique(df["run"], return_counts=True)
    assert_array_equal(run_unique, [1, 2, 3])
    assert_array_equal(run_counts, [3, 3, 3])

    assert np.all(df["ratio"][~df["ratio"].isna()] >= 1)

    assert np.sum(df["sampling_time_s"].isna()) == 0
    assert np.sum(df["total_time_s"].isna()) == 0

    assert np.all(df["sampling_time_s"] > 0)
    assert np.all(df["total_time_s"] > 0)


def test_leverage_score_sampling_experiment_parallel(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = LeverageScoreSamplingExperiment(
        dataset=dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=2,
        num_runs=3,
    )
    experiment.run(parallel=True)

    df = pd.read_csv(results_filename)

    run_unique, run_counts = np.unique(df["run"], return_counts=True)
    assert_array_equal(run_unique, [1, 2, 3])
    assert_array_equal(run_counts, [3, 3, 3])

    assert np.all(df["ratio"][~df["ratio"].isna()] >= 1)

    assert np.sum(df["sampling_time_s"].isna()) == 0
    assert np.sum(df["total_time_s"].isna()) == 0

    assert np.all(df["sampling_time_s"] > 0)
    assert np.all(df["total_time_s"] > 0)


def test_leverage_score_sampling_reduction(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = LeverageScoreSamplingExperiment(
        dataset=dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=1,
        num_runs=1,
        only_compute_once=False,
    )

    for cur_config in experiment.get_config_grid():
        cur_X, cur_y, cur_weights = experiment.get_reduced_X_y_weights(cur_config)
        assert cur_X.shape[0] == cur_config["size"]
        assert cur_y.shape[0] == cur_config["size"]
        assert cur_X.shape[1] == dataset.get_d()
        assert cur_weights.shape[0] == cur_config["size"]
