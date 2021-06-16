from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# the downloaded datasets will go here
DATA_DIR = BASE_DIR / ".data-cache"

RESULTS_DIR = BASE_DIR / "experimental-results"

PLOTS_DIR = BASE_DIR / "plots"

LOGGER_NAME = "efficient_probit_regression"
