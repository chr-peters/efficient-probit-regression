import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# the downloaded datasets will go here
DATA_DIR = BASE_DIR / ".data-cache"

RESULTS_DIR = BASE_DIR / "experimental-results"

RESULTS_DIR_BAYES = BASE_DIR / "experimental-results" / "bayes"

PLOTS_DIR = BASE_DIR / "plots"


_logger = None


def get_logger():
    global _logger

    if _logger is not None:
        return _logger

    _logger = logging.getLogger("efficient-probit-regression")
    _logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - PID: %(process)d - "
            "PName: %(processName)s - %(levelname)s - %(message)s"
        )
    )
    _logger.addHandler(stream_handler)

    return _logger
