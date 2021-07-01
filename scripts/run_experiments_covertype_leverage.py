from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Covertype
from efficient_probit_regression.experiments import LeverageScoreSamplingExperiment

_logger = settings.get_logger()

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

dataset = Covertype()

_logger.info("Starting leverage score sampling experiment with rounding and rescaling")
experiment = LeverageScoreSamplingExperiment(
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.RESULTS_DIR
    / f"{dataset.get_name()}_leverage_rounded_scaled.csv",
    only_compute_once=True,
    online=True,
    round_up=True,
    rescale=True,
)
experiment.run(parallel=True)
