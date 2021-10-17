from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Covertype, KDDCup, Webspam  # noqa
from efficient_probit_regression.experiments import UniformSamplingExperiment

logger = settings.get_logger()

num_runs = 30
P = 2

dataset = Covertype()
# dataset = KDDCup()
# dataset = Webspam()

if dataset.get_name() in ["covertype", "webspam"]:
    min_size = 500
    max_size = 15000
    step_size = 500
elif dataset.get_name() == "kddcup":
    min_size = 1000
    max_size = 30000
    step_size = 1000
else:
    raise ValueError(
        "Unknown dataset! Can't determine min_size, max_size and step_size!"
    )

logger.info(
    f"Settings min_size = {min_size}, max_size = {max_size} and step_size = {step_size}"
)

logger.info("Starting uniform sampling experiment")
experiment = UniformSamplingExperiment(
    p=P,
    min_size=min_size,
    max_size=max_size,
    step_size=step_size,
    num_runs=num_runs,
    dataset=dataset,
    results_filename=settings.get_results_dir_p(P)
    / f"{dataset.get_name()}_uniform_p_{P}.csv",
)
experiment.run(parallel=True)
