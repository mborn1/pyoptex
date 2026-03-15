"""Temporary helper for capturing regression reference data from examples."""

import json

import numpy as np
import pandas as pd


def _serialize(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "shape": list(value.shape),
            "columns": value.columns.tolist(),
            "data": value.values.tolist(),
            "dtypes": {c: str(value[c].dtype) for c in value.columns},
        }
    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "name": value.name,
            "index": value.index.tolist(),
            "data": value.tolist(),
        }
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    # Fallback: try to convert to string
    return str(value)


def log_checkpoint(name, value):
    """Print a structured JSON checkpoint line for later parsing."""
    serialized = _serialize(value)
    print(f"@@CHECKPOINT@@{json.dumps({'name': name, 'value': serialized})}", flush=True)
