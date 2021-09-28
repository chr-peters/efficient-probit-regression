from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Covertype
from efficient_probit_regression.experiments import UniformSamplingExperiment

_logger = settings.get_logger()

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 11

P = 1

dataset = Covertype()

_logger.info("Starting uniform sampling experiment")
experiment = UniformSamplingExperiment(
    p=P,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.get_results_dir_p(P)
    / f"{dataset.get_name()}_uniform_p_{P}.csv",
)
experiment.run(parallel=True)
