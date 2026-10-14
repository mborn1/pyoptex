#!/usr/bin/env python3

"""
Example: blocked classical designs.

When not all runs can be done under the same conditions (e.g. 8 runs per
day), the design is split into blocks. A good blocking scheme makes the
block effects orthogonal to the terms of interest:

* a 2^5 factorial in 4 blocks, which confounds only higher-order
  interactions with the blocks,
* a central composite design in orthogonal blocks (Box & Hunter, 1957),
* a Box-Behnken design in the orthogonal blocks of Box & Behnken (1960).

The runs are randomized within each block. Finally, the blocked CCD is
evaluated with the block as a random effect, using the same machinery as
pyoptex's optimal designs.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.classical import box_behnken, central_composite, full_factorial, randomize
from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_parameters, default_fn
from pyoptex.doe.fixed_structure.evaluate import evaluate_metrics
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

# 2^5 factorial in 4 blocks of 8 runs
factors = [Factor(name) for name in "ABCDE"]
Y = full_factorial(factors, n_blocks=4)
print("=== 2^5 factorial in 4 blocks ===")
print(f"Runs per block: {Y['Block'].value_counts().sort_index().tolist()}")
print(f"Confounded with blocks: {[''.join(e) for e in Y.attrs['confounded_with_blocks']]}")

# Central composite design in 3 orthogonal blocks: 2 cube blocks and 1 axial block.
# With alpha > 1 the axial runs lie outside the factor ranges (here even a negative
# Time), so choose min and max such that alpha times the half-range is still feasible.
factors = [
    Factor("Temperature", min=150, max=200),
    Factor("Pressure", min=1, max=5),
    Factor("Time", min=10, max=60),
]
Y_ccd = central_composite(factors, alpha="orthogonal", n_blocks=3)
print("\n=== CCD in 3 orthogonal blocks ===")
print(f"Runs per block: {Y_ccd['Block'].value_counts().sort_index().tolist()}")
print(f"Coded alpha for orthogonal blocking: {(Y_ccd['Pressure'].max() - 3) / 2:.4f}")
print(randomize(Y_ccd).round(2))

log_checkpoint("blocked_ccd_shape", list(Y_ccd.shape))

# Box-Behnken design for 4 factors in the 3 orthogonal blocks of the paper
Y_bbd = box_behnken([Factor(name) for name in "ABCD"], n_blocks=3)
print("\n=== Box-Behnken (4 factors) in 3 blocks ===")
print(f"Runs per block: {Y_bbd['Block'].value_counts().sort_index().tolist()}")

# Evaluate the blocked CCD with a random block effect (variance ratio 1)
model = partial_rsm_names({str(f.name): "quad" for f in factors})
Y2X = model2Y2X(model, factors)
blocks = RandomEffect(Y_ccd["Block"].to_numpy() - 1, ratio=1.0)
params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), len(Y_ccd), block_effects=[blocks])
(d_random,) = evaluate_metrics(Y_ccd, params, [Dopt()])
params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), len(Y_ccd))
(d_none,) = evaluate_metrics(Y_ccd, params, [Dopt()])
print("\n=== Blocked CCD, D-criterion ===")
print(f"Without block effects: {d_none:.3f}")
print(f"With random block effects: {d_random:.3f}")

log_checkpoint("blocked_ccd_dopt", float(np.round(d_random, 6)))
