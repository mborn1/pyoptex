#!/usr/bin/env python3

"""
Example: Box-Behnken design with D-efficiency evaluation.

Demonstrates a Box-Behnken design for 3 continuous factors and
evaluates its D-efficiency under a quadratic model.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import box_behnken
from pyoptex.doe.classical.evaluate import d_efficiency
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, partial_rsm_names

factors = [
    Factor("Speed", min=100, max=500),
    Factor("Feed", min=0.05, max=0.25),
    Factor("Depth", min=0.5, max=2.0),
]

Y = box_behnken(factors, n_center=3)
print("=== Box-Behnken (3 factors) ===")
print(f"Runs: {len(Y)}")
print(Y)

model = partial_rsm_names({"Speed": "quad", "Feed": "quad", "Depth": "quad"})
Y2X = model2Y2X(model, factors)
d_eff = d_efficiency(Y, factors, Y2X)
print(f"D-efficiency: {d_eff:.2f}%")

log_checkpoint("bb_shape", list(Y.shape))
log_checkpoint("bb_d_efficiency", d_eff)
