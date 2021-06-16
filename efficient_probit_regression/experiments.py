import abc
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import settings
from .datasets import BaseDataset
from .probit_model import ProbitModel
from .sampling import compute_leverage_scores, leverage_score_sampling, uniform_sampling

_logger = settings.get_logger()

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: BaseDataset,
        results_filename,
    ):
        self.num_runs = num_runs
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.dataset = dataset
        self.results_filename = results_filename

    @abc.abstractmethod
    def get_reduced_X_y_weights(self, config):
        """
        Abstract method that each experiment overrides to return the reduced
        matrix X, label vector y and weights that correspond to an experimental config.

        Parameters:
        -----------
        config : dict
            The current experimental config.

        Returns:
        --------
        X : np.ndarray
            Reduced matrix X.
        y : np.ndarray
            Reduced label vector y.
        w : np.ndarray
            New weight vector.
        """
        pass

    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        grid = []
        for size in np.arange(
            start=self.min_size,
            stop=self.max_size + self.step_size,
            step=self.step_size,
        ):
            for run in range(1, self.num_runs + 1):
                grid.append({"run": run, "size": size})

        return grid

    def optimize(self, X, y, w):
        """
        Optimize the Probit regression problem given by X, y and w.

        Returns:
        --------
        beta_opt : np.ndarray
            The optimal parameters.
        """
        try:
            model = ProbitModel(X=X, y=y, w=w)
            model.fit()
            beta_opt = model.get_params()
        except ValueError:
            # this is executet if y only contains 1 or -1 label
            beta_opt = None
        return beta_opt

    def run(self, parallel=False, n_jobs=4):
        """
        Run the experiment.
        """
        X, y = self.dataset.get_X(), self.dataset.get_y()
        model = ProbitModel(X=X, y=y)
        beta_opt = self.dataset.get_beta_opt()

        def objective_function(beta):
            return model.negative_log_likelihood(beta)

        f_opt = objective_function(beta_opt)

        _logger.info("Running experiments...")

        def job_function(cur_config):
            _logger.info(f"Current experimental config: {cur_config}")

            start_time = perf_counter()

            X_reduced, y_reduced, weights = self.get_reduced_X_y_weights(cur_config)
            sampling_time = perf_counter() - start_time

            cur_beta_opt = self.optimize(X=X_reduced, y=y_reduced, w=weights)
            total_time = perf_counter() - start_time

            if cur_beta_opt is not None:
                cur_ratio = objective_function(cur_beta_opt) / f_opt
            else:
                cur_ratio = None
            return {
                **cur_config,
                "ratio": cur_ratio,
                "sampling_time_s": sampling_time,
                "total_time_s": total_time,
            }

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(job_function)(cur_config)
                for cur_config in self.get_config_grid()
            )
        else:
            results = [
                job_function(cur_config) for cur_config in self.get_config_grid()
            ]

        _logger.info(f"Writing results to {self.results_filename}")

        df = pd.DataFrame(results)
        df.to_csv(self.results_filename, index=False)

        _logger.info("Done.")


class UniformSamplingExperiment(BaseExperiment):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: BaseDataset,
        results_filename,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
        )

    def get_reduced_X_y_weights(self, config):
        X, y = self.dataset.get_X(), self.dataset.get_y()
        size = config["size"]

        X_reduced, y_reduced = uniform_sampling(X=X, y=y, sample_size=size)

        weights = np.ones(size)

        return X_reduced, y_reduced, weights


class LeverageScoreSamplingExperiment(BaseExperiment):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: BaseDataset,
        results_filename,
        only_compute_once=True,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
        )
        self.only_compute_once = only_compute_once

    def run(self, **kwargs):
        if self.only_compute_once:
            _logger.info("Computing leverage scores upfront...")
            self._leverage_scores = compute_leverage_scores(self.dataset.get_X())
            _logger.info("Done.")

        super().run(**kwargs)

    def get_reduced_X_y_weights(self, config):
        X, y = self.dataset.get_X(), self.dataset.get_y()
        size = config["size"]

        if self.only_compute_once:
            precomputed_scores = self._leverage_scores
        else:
            precomputed_scores = None

        X_reduced, y_reduced, weights = leverage_score_sampling(
            X=X,
            y=y,
            sample_size=size,
            augmented=True,
            precomputed_scores=precomputed_scores,
        )

        return X_reduced, y_reduced, weights
