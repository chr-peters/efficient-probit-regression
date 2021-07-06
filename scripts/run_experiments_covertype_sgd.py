from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Covertype
from efficient_probit_regression.experiments import SGDExperiment

_logger = settings.get_logger()

MIN_SIZE = 1
MAX_SIZE = 1
STEP_SIZE = 1
NUM_RUNS = 21

dataset = Covertype()

_logger.info("Starting SGD experiment")
experiment = SGDExperiment(
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sgd.csv",
)
experiment.run(parallel=True)
