import abc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import settings
from .datasets import BaseDataset
from .probit_model import ProbitModel, ProbitSGD
from .sampling import (
    compute_leverage_scores,
    compute_leverage_scores_online,
    gibbs_sampler_probit,
    leverage_score_sampling,
    online_ridge_leverage_score_sampling,
    uniform_sampling,
)

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


class SGDExperiment(BaseExperiment):
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
        """
        In SGD, no reduction is performed.
        """
        X, y = self.dataset.get_X(), self.dataset.get_y()

        return X, y, np.ones(y.shape)

    def optimize(self, X, y, w):
        """
        Applies SGD in one pass over the data.

        Returns:
        --------
        params : np.ndarray
            The parameters found by SGD.
        """
        n = X.shape[0]
        sgd = ProbitSGD()
        for i in _rng.permutation(n):  # pass over the data in random order
            cur_sample = X[i]
            cur_label = y[i]
            sgd.new_sample(x=cur_sample, y=cur_label)

        return sgd.get_params()


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
        online=False,
        round_up=True,
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
        self.online = online
        self.round_up = round_up

    def run(self, **kwargs):
        if self.only_compute_once:
            if self.online:
                _logger.info("Computing online leverage scores upfront...")
                self._leverage_scores = compute_leverage_scores_online(
                    self.dataset.get_X()
                )
            else:
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
            online=self.online,
            precomputed_scores=precomputed_scores,
            round_up=self.round_up,
        )

        return X_reduced, y_reduced, weights


class OnlineRidgeLeverageScoreSamplingExperiment(BaseExperiment):
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

        X_reduced, y_reduced, weights = online_ridge_leverage_score_sampling(
            X=X,
            y=y,
            sample_size=size,
            augmentation_constant=1 / X.shape[0],
        )

        return X_reduced, y_reduced, weights


class BaseExperimentBayes(abc.ABC):
    def __init__(
        self,
        dataset: BaseDataset,
        num_runs: int,
        min_size: int,
        max_size: int,
        step_size: int,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        samples_per_chain: int,
        num_chains: int,
    ):
        self.dataset = dataset
        self.num_runs = num_runs
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.samples_per_chain = samples_per_chain
        self.num_chains = num_chains

    @abc.abstractmethod
    def get_reduced_X_y_probabilities(self, size):
        pass

    @abc.abstractmethod
    def get_method_name(self):
        """
        Returns the name of the method, like "uniform", "leverage", or "leverage_online".
        """  # noqa
        pass

    def _get_latest_run(self, dir: Path):
        """
        This makes sure that no files are overwritten in dir, by getting the
        latest run number.
        """
        file_list = dir.glob(
            f"{self.dataset.get_name()}_sample_{self.get_method_name()}_run_*.csv"
        )
        max_run = 0
        for cur_file in file_list:
            cur_run = int(cur_file.stem.split("_")[-1])
            if cur_run > max_run:
                max_run = cur_run

        return max_run

    def run(self, results_dir=settings.RESULTS_DIR_BAYES):
        if not results_dir.exists():
            results_dir.mkdir()

        # make sure that no old data is overwritten
        latest_run = self._get_latest_run(dir=results_dir)

        for cur_run in range(latest_run + 1, latest_run + self.num_runs + 1):
            # create a sample for each size
            samples = []
            for cur_size in range(
                self.min_size, self.max_size + self.step_size, self.step_size
            ):
                _logger.info(
                    f"METHOD: {self.get_method_name()} - RUN: {cur_run} - SIZE: {cur_size}"  # noqa
                )
                _logger.info("Reducing the data...")
                (
                    X_reduced,
                    y_reduced,
                    probabilities,
                ) = self.get_reduced_X_y_probabilities(size=cur_size)

                _logger.info(
                    f"Done. Running the Gibbs sampler with NUM_CHAINS: {self.num_chains} "  # noqa
                    f"and SAMPLES_PER_CHAIN: {self.samples_per_chain}"
                )

                cur_sample = gibbs_sampler_probit(
                    X=X_reduced,
                    y=y_reduced,
                    prior_mean=self.prior_mean,
                    prior_cov=self.prior_cov,
                    num_samples=self.samples_per_chain,
                    num_chains=self.num_chains,
                    burn_in=100,
                    probabilities=probabilities,
                )

                _logger.info("Done.")
                samples.append({"size": cur_size, "sample": cur_sample})

            # concatenate the samples to a dataframe
            df_list = []
            for cur_sample in samples:
                cur_df = pd.DataFrame(
                    cur_sample["sample"],
                    columns=[f"beta_{i}" for i in range(self.dataset.get_d())],
                )
                cur_df["size"] = cur_sample["size"]
                cur_df["run"] = cur_run
                df_list.append(cur_df)

            df = pd.concat(df_list, ignore_index=True)
            df.to_csv(
                results_dir
                / f"{self.dataset.get_name()}_sample_{self.get_method_name()}_run_{cur_run}.csv",  # noqa
                index=False,
            )


class UniformSamplingExperimentBayes(BaseExperimentBayes):
    def get_method_name(self):
        return "uniform"

    def get_reduced_X_y_probabilities(self, size):
        X_reduced, y_reduced = uniform_sampling(
            X=self.dataset.get_X(), y=self.dataset.get_y(), sample_size=size
        )

        return X_reduced, y_reduced, np.full(size, 1 / self.dataset.get_n())


class LeverageScoreSamplingExperimentBayes(BaseExperimentBayes):
    precomputed_scores = None

    def get_method_name(self):
        return "leverage"

    def get_reduced_X_y_probabilities(self, size):
        if self.precomputed_scores is None:
            self.precomputed_scores = compute_leverage_scores(self.dataset.get_X())

        X_reduced, y_reduced, weights = leverage_score_sampling(
            X=self.dataset.get_X(),
            y=self.dataset.get_y(),
            sample_size=size,
            augmented=True,
            online=False,
            round_up=True,
            precomputed_scores=self.precomputed_scores,
        )

        probabilities = 1 / (weights * size)

        return X_reduced, y_reduced, probabilities


class OnlineLeverageScoreSamplingExperimentBayes(BaseExperimentBayes):
    precomputed_scores = None

    def get_method_name(self):
        return "leverage_online"

    def get_reduced_X_y_probabilities(self, size):
        if self.precomputed_scores is None:
            self.precomputed_scores = compute_leverage_scores_online(
                self.dataset.get_X()
            )

        X_reduced, y_reduced, weights = leverage_score_sampling(
            X=self.dataset.get_X(),
            y=self.dataset.get_y(),
            sample_size=size,
            augmented=True,
            online=True,
            round_up=True,
            precomputed_scores=self.precomputed_scores,
        )

        probabilities = 1 / (weights * size)

        return X_reduced, y_reduced, probabilities
