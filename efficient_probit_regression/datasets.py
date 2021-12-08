import abc
import bz2
import io
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_covtype,
    fetch_kddcup99,
    load_iris,
    load_svmlight_file,
    make_blobs,
)
from sklearn.preprocessing import scale

from efficient_probit_regression import settings
from efficient_probit_regression.probit_model import PGeneralizedProbitModel

_logger = settings.get_logger()

_rng = np.random.default_rng()


def add_intercept(X):
    return np.append(X, np.ones(shape=(X.shape[0], 1)), axis=1)


class BaseDataset(abc.ABC):
    def __init__(self, add_intercept=True, use_caching=True, cache_dir=None):
        self.use_caching = use_caching
        if cache_dir is None:
            cache_dir = settings.DATA_DIR
        self.cache_dir = cache_dir

        if use_caching and not self.cache_dir.exists():
            self.cache_dir.mkdir()

        self.X = None
        self.y = None
        self.beta_opt_dir = {}
        self.add_intercept = add_intercept

    @abc.abstractmethod
    def load_X_y(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _load_X_y_cached(self):
        if not self.use_caching:
            _logger.info("Loading X and y...")
            X, y = self.load_X_y()
            if self.add_intercept:
                X = add_intercept(X)
            _logger.info("Done.")
            return X, y

        X_path = self.get_binary_path_X()
        y_path = self.get_binary_path_y()
        if X_path.exists() and y_path.exists():
            _logger.info(
                f"Loading cached versions of X and y found at {X_path} and {y_path}..."
            )
            X = np.load(X_path)
            if self.add_intercept:
                X = add_intercept(X)
            y = np.load(y_path)
            _logger.info("Done.")
            return X, y

        _logger.info("Loading X and y...")
        X, y = self.load_X_y()
        _logger.info("Done.")
        np.save(X_path, X)
        np.save(y_path, y)
        _logger.info(f"Saved X and y at {X_path} and {y_path}.")

        if self.add_intercept:
            X = add_intercept(X)

        return X, y

    def _compute_beta_opt(self, p):
        model = PGeneralizedProbitModel(p=p, X=self.get_X(), y=self.get_y())
        model.fit()
        beta_opt = model.get_params()
        return beta_opt

    def _get_beta_opt_cached(self, p):
        if not self.use_caching:
            _logger.info(f"Computing beta_opt for p={p}...")
            beta_opt = self._compute_beta_opt(p)
            _logger.info("Done.")
            return beta_opt

        beta_opt_path = self.get_binary_path_beta_opt(p)
        if beta_opt_path.exists():
            _logger.info(
                f"Loading cached version of beta_opt for p={p} found at {beta_opt_path}..."  # noqa
            )
            beta_opt = np.load(beta_opt_path)
            _logger.info("Done.")
            return beta_opt

        _logger.info(f"Computing beta_opt for p={p}...")
        beta_opt = self._compute_beta_opt(p)
        _logger.info("Done.")
        np.save(beta_opt_path, beta_opt)
        _logger.info(f"Saved beta_opt for p={p} at {beta_opt_path}.")

        return beta_opt

    def _assert_data_loaded(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_X_y_cached()

    def get_binary_path_X(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_X.npy"

    def get_binary_path_y(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_y.npy"

    def get_binary_path_beta_opt(self, p) -> Path:
        return self.cache_dir / f"{self.get_name()}_beta_opt_p_{p}.npy"

    def get_X(self):
        """The function get_X() returns the data matrix from the data object."""
        self._assert_data_loaded()
        return self.X

    def get_y(self):
        self._assert_data_loaded()
        return self.y

    def get_n(self):
        self._assert_data_loaded()
        return self.X.shape[0]

    def get_d(self):
        self._assert_data_loaded()
        return self.X.shape[1]

    def get_beta_opt(self, p):
        if p not in self.beta_opt_dir.keys():
            self.beta_opt_dir[p] = self._get_beta_opt_cached(p)

        return self.beta_opt_dir[p]


class Covertype(BaseDataset):
    """
    Dataset Homepage:
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """

    features_continuous = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        return "covertype"

    def load_X_y(self):
        _logger.info("Fetching covertype from sklearn...")
        sklearn_result = fetch_covtype(as_frame=True)
        df = sklearn_result.frame

        _logger.info("Preprocessing...")

        # Cover_Type 2 gets the label 1, everything else gets the label -1.
        # This ensures maximum balance.
        y = df["Cover_Type"].apply(lambda x: 1 if x == 2 else -1).to_numpy()
        df = df.drop("Cover_Type", axis="columns")

        # scale the continuous features to mean zearo and variance 1
        # and leave the 0/1 features as is
        X_continuous = df[self.features_continuous].to_numpy()
        X_continuous = scale(X_continuous)

        features_binary = list(set(df.columns) - set(self.features_continuous))
        X_binary = df[features_binary].to_numpy()

        # put binary features and scaled features back together
        X = np.append(X_continuous, X_binary, axis=1)

        return X, y


class KDDCup(BaseDataset):
    """
    Dataset Homepage:
    https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    """

    features_continuous = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    features_discrete = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        return "kddcup"

    def load_X_y(self):
        _logger.info("Fetching kddcup from sklearn...")
        sklearn_result = fetch_kddcup99(as_frame=True, percent10=True)
        df = sklearn_result.frame

        _logger.info("Preprocessing...")

        # convert label "normal." to -1 and everything else to 1
        y = df.labels.apply(lambda x: -1 if x.decode() == "normal." else 1).to_numpy()

        # get all the continuous features
        X_continuous = df[self.features_continuous]

        # the feature num_outbound_cmds has only one value that doesn't
        # change, so drop it
        X_continuous = X_continuous.drop("num_outbound_cmds", axis="columns")

        # convert to numpy array
        X_continuous = X_continuous.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X_continuous)

        return X, y


class Webspam(BaseDataset):
    """
    Dataset Source:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam
    """

    dataset_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.bz2"  # noqa: E501

    def __init__(self, drop_sparse_columns=True, use_caching=True):
        self.drop_sparse_columns = drop_sparse_columns
        super().__init__(use_caching=use_caching)

    def get_name(self):
        if self.drop_sparse_columns:
            return "webspam"
        else:
            return "webspam_with_sparse"

    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def download_dataset(self):
        _logger.info(f"Downloading data from {self.dataset_url}")

        context = ssl._create_unverified_context()
        with urllib.request.urlopen(self.dataset_url, context=context) as f:
            contents = f.read()

        _logger.info("Download completed.")
        _logger.info("Extracting data...")

        file_raw = bz2.open(io.BytesIO(contents))
        X_sparse, y = load_svmlight_file(file_raw)

        # convert scipy Compressed Sparse Row array into numpy array
        X = X_sparse.toarray()

        df = pd.DataFrame(X)
        df["LABEL"] = y

        _logger.info(f"Writing .csv file to {self.get_raw_path()}")

        df.to_csv(self.get_raw_path(), index=False)

    def load_X_y(self):
        if not self.get_raw_path().exists():
            _logger.info(f"Couldn't find dataset at location {self.get_raw_path()}")
            self.download_dataset()

        df = pd.read_csv(self.get_raw_path())

        _logger.info("Preprocessing the data...")

        y = df["LABEL"].to_numpy()
        df = df.drop("LABEL", axis="columns")

        # drop all columns that only have constant values
        # drop all columns that contain only one non-zero entry
        for cur_column_name in df.columns:
            cur_column = df[cur_column_name]
            cur_column_sum = cur_column.astype(bool).sum()
            unique_values = cur_column.unique()
            if len(unique_values) <= 1:
                df = df.drop(cur_column_name, axis="columns")
            if self.drop_sparse_columns and cur_column_sum == 1:
                df = df.drop(cur_column_name, axis="columns")

        X = df.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X)

        return X, y


class Iris(BaseDataset):
    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        return "iris"

    def load_X_y(self):
        X, y = load_iris(return_X_y=True)
        X = scale(X)
        y = np.where(y == 1, 1, -1)

        return X, y


class Example2D(BaseDataset):
    def __init__(self):
        super().__init__(add_intercept=True, use_caching=False)

    def get_name(self):
        return "example-2d"

    def load_X_y(self):
        centers = np.array([[-1, -1], [1, 1], [4.5, 5]])
        X, y = make_blobs(
            n_samples=[80, 80, 15],
            n_features=2,
            centers=centers,
            cluster_std=[1, 1, 0.5],
            random_state=1,
        )
        y = np.where(y == 2, 0, y)

        y = 2 * y - 1

        return X, y
