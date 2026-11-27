#!/usr/bin/env python3

"""
Example: crossed array for robust parameter design.

Taguchi's robust parameter design crosses an inner array with the control
factors (here an L8 with 4 factors) with an outer array with the noise
factors (here an L4 with 3 factors): every control setting is run at every
noise setting, 8 x 4 = 32 runs. The response is then summarized per control
setting over the noise runs (e.g. a signal-to-noise ratio).
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

import numpy as np

from pyoptex.doe.classical import crossed_array, taguchi
from pyoptex.utils import Factor

control = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
    Factor("Time", min=10, max=60),
    Factor("Catalyst", type="categorical", levels=["A", "B"]),
]
noise = [
    Factor("Humidity", min=20, max=80),
    Factor("Ambient", min=15, max=30),
    Factor("Batch", type="categorical", levels=["Old", "New"]),
]

inner = taguchi("L8", control, columns=[1, 2, 4, 7])
outer = taguchi("L4", noise)
Y = crossed_array(inner, outer)
print("=== Crossed array: L8 (control) x L4 (noise) ===")
print(f"Runs: {len(Y)}")
print(Y.head(8))

# Taguchi's wide format: one response column per noise run
Y["inner_run"] = np.repeat(np.arange(len(inner)), len(outer))
Y["outer_run"] = np.tile(np.arange(len(outer)), len(inner))
Y["response"] = np.nan  # to be measured
wide = Y.pivot(index="inner_run", columns="outer_run", values="response")
print(f"Wide format: {wide.shape[0]} control settings x {wide.shape[1]} noise runs")

log_checkpoint("crossed_array_shape", [len(Y), len(inner.columns) + len(outer.columns)])
