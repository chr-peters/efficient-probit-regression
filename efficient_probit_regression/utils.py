from . import settings
from .datasets import BaseDataset
from .experiments import LeverageScoreSamplingExperiment, UniformSamplingExperiment

_logger = settings.get_logger()


def run_experiments(dataset: BaseDataset, min_size, max_size, step_size, num_runs):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    _logger.info("Starting uniform sampling experiment")
    experiment_uniform = UniformSamplingExperiment(
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_uniform.csv",
    )
    experiment_uniform.run(parallel=True)

    _logger.info("Starting leverage score sampling experiment")
    experiment_leverage = LeverageScoreSamplingExperiment(
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_leverage.csv",
    )
    experiment_leverage.run(parallel=True)

    _logger.info("Starting online leverage score sampling experiment")
    experiment_leverage = LeverageScoreSamplingExperiment(
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        dataset=dataset,
        results_filename=settings.RESULTS_DIR
        / f"{dataset.get_name()}_leverage_online.csv",
        online=True,
    )
    experiment_leverage.run(parallel=True)
