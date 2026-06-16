#!/usr/bin/env python3

"""
Example: Central Composite Design (face-centered) with evaluation.

Demonstrates a face-centered CCD for 3 continuous factors,
evaluates D/A/G-efficiency, saves the design to CSV, and
optionally shows a heatmap and correlation map.
"""

import os

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex._seed import set_seed
from pyoptex.doe.classical import central_composite
from pyoptex.doe.classical.evaluate import design_info
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
    Factor("Time", min=10, max=60),
]

Y = central_composite(factors, alpha="face-centered", n_center=3)
print("=== CCD Face-Centered (3 factors) ===")
print(f"Runs: {len(Y)}")
print(Y)

model = partial_rsm_names({"Temperature": "quad", "Pressure": "quad", "Time": "quad"})
Y2X = model2Y2X(model, factors)
info = design_info(Y, factors, Y2X, n_samples=2000)
print(f"D-efficiency: {info['d_efficiency']:.2f}%")
print(f"A-efficiency: {info['a_efficiency']:.2f}%")

log_checkpoint("ccd_shape", list(Y.shape))

root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, "example_central_composite.csv"), index=False)
print("Saved CCD design to example_central_composite.csv")

from pyoptex.doe.utils.evaluate import design_heatmap, plot_correlation_map

design_heatmap(Y, factors).show()
plot_correlation_map(Y, factors, Y2X, model=model).show()
