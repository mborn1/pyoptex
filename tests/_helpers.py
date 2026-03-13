"""Shared helpers for regression tests."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_REFERENCE_ROOT = Path(__file__).parent / "reference_data"


def load_reference(name: str) -> dict:
    """Load a reference JSON file by its key (e.g. 'analysis/simple_model')."""
    path = _REFERENCE_ROOT / f"{name}.json"
    with open(path) as f:
        return json.load(f)


_TIME_RE = re.compile(r"(Time:\s+)\d{2}:\d{2}:\d{2}")


def normalize_summary(s: str) -> str:
    """Remove the time field from a statsmodels summary string so it is stable across runs."""
    return _TIME_RE.sub(r"\g<1>HH:MM:SS", s)


def assert_summary_equal(actual_summary, ref_str: str) -> None:
    """Assert a statsmodels Summary object matches its reference string, ignoring the time."""
    actual_str = str(actual_summary)
    assert normalize_summary(actual_str) == normalize_summary(ref_str), (
        f"Summary mismatch.\nActual:\n{actual_str}\nExpected:\n{ref_str}"
    )


def assert_array_equal(actual: np.ndarray, ref_data: dict, *, rtol: float = 1e-6, atol: float = 0.0) -> None:
    """Assert a numpy array matches its serialised reference."""
    expected = np.array(ref_data["data"], dtype=ref_data["dtype"]).reshape(ref_data["shape"])
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_frame_equal(actual: pd.DataFrame, ref_data: dict, *, rtol: float = 1e-6) -> None:
    """Assert a DataFrame matches its serialised reference (shape, columns, numeric values)."""
    assert list(actual.shape) == ref_data["shape"], (
        f"DataFrame shape mismatch: {list(actual.shape)} != {ref_data['shape']}"
    )
    assert actual.columns.tolist() == ref_data["columns"], (
        f"Column mismatch: {actual.columns.tolist()} != {ref_data['columns']}"
    )
    expected = pd.DataFrame(ref_data["data"], columns=ref_data["columns"])
    for col in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[col]):
            np.testing.assert_allclose(
                actual[col].values,
                expected[col].values,
                rtol=rtol,
                err_msg=f"Column '{col}' mismatch",
            )
        else:
            assert actual[col].tolist() == expected[col].tolist(), (
                f"Column '{col}' string mismatch"
            )
