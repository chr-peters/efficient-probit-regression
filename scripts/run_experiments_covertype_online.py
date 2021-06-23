from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Covertype
from efficient_probit_regression.experiments import LeverageScoreSamplingExperiment

_logger = settings.get_logger()

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

dataset = Covertype()

_logger.info("Starting online leverage score sampling experiment")
experiment_leverage_online = LeverageScoreSamplingExperiment(
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_leverage_online.csv",
    only_compute_once=True,
    online=True,
)
experiment_leverage_online.run(parallel=True)
