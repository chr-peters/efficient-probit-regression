from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Covertype, KDDCup, Webspam  # noqa
from efficient_probit_regression.experiments import LeverageScoreSamplingExperiment

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

# MIN_SIZE = 1000
# MAX_SIZE = 30000
# STEP_SIZE = 1000
# NUM_RUNS = 21

P = 1

dataset = Covertype()
# dataset = KDDCup()
# dataset = Webspam()

experiment = LeverageScoreSamplingExperiment(
    p=P,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.get_results_dir_p(P)
    / f"{dataset.get_name()}_leverage_p_{P}.csv",
    only_compute_once=False,
    online=False,
    round_up=True,
    fast_approx=True,
)
experiment.run(parallel=True)
