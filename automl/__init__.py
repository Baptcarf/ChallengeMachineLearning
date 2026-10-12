from .__main__ import AutoML

_default = AutoML()

__all__ = ["AutoML"]


def fit(dataDest):
    return _default.fit(dataDest)


def eval():
    return _default.eval()


def predict(test_path):
    return _default.predict(test_path)

def print():
    _default.print()