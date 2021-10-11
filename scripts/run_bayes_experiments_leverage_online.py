import numpy as np

from efficient_probit_regression.datasets import Covertype
from efficient_probit_regression.experiments import (
    OnlineLeverageScoreSamplingExperimentBayes,
)
from efficient_probit_regression.settings import get_logger

logger = get_logger()

min_size = 500
max_size = 15000
step_size = 500

num_runs = 5
samples_per_chain = 1000
num_chains = 1

dataset = Covertype()

prior_mean = np.zeros(dataset.get_d())
prior_cov = 10 * np.eye(dataset.get_d())

if dataset.get_name() == "covertype":
    burn_in = 100
elif dataset.get_name() == "kddcup":
    burn_in = 2000
else:
    raise ValueError("Unknown dataset! Can't determine burn_in!")

logger.info(f"Setting burn_in = {burn_in}")

experiment = OnlineLeverageScoreSamplingExperimentBayes(
    dataset=dataset,
    num_runs=num_runs,
    min_size=min_size,
    max_size=max_size,
    step_size=step_size,
    prior_mean=prior_mean,
    prior_cov=prior_cov,
    samples_per_chain=samples_per_chain,
    num_chains=num_chains,
    burn_in=burn_in,
)

experiment.run()
