import numpy as np

from efficient_probit_regression.datasets import Covertype
from efficient_probit_regression.experiments import (
    OnlineLeverageScoreSamplingExperimentBayes,
)

min_size = 500
max_size = 15000
step_size = 500

num_runs = 5
samples_per_chain = 1000
num_chains = 1

dataset = Covertype()

prior_mean = np.zeros(dataset.get_d())
prior_cov = 10 * np.eye(dataset.get_d())

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
)

experiment.run()
