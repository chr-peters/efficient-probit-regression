from setuptools import find_packages, setup

setup(
    name="efficient-probit-regression",
    version="0.1.0",
    author="Christian Peters",
    packages=find_packages(),
    install_requires=[
        "joblib",
        "numba",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "tqdm",   #progress bar
    ],
)
