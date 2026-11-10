from .__main__ import AutoML
from .config import N_JOBS, N_TRIALS, RANDOM_STATE, TIME_BUDGET
from .data import Dataset, lireDataset, listerDatasets

_default = AutoML()

__all__ = [
    "N_JOBS",
    "N_TRIALS",
    "RANDOM_STATE",
    "TIME_BUDGET",
    "AutoML",
    "Dataset",
    "lireDataset",
    "listerDatasets",
]


def fit(dataDest):
    return _default.fit(dataDest)


def eval():
    return _default.eval()


def predict(test_path):
    return _default.predict(test_path)

def print():
    _default.print()