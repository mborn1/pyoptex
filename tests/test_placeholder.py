"""Smoke tests verifying the pyoptex package imports and basic utilities work."""

import numpy as np


def test_import():
    import pyoptex  # noqa: F401


def test_version():
    import pyoptex

    assert isinstance(pyoptex.__version__, str)
    parts = pyoptex.__version__.split(".")
    assert len(parts) >= 2


def test_numpy_available():
    arr = np.array([1.0, 2.0, 3.0])
    assert arr.sum() == 6.0
