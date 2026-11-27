#!/usr/bin/env python3

"""
Example: simplex lattice and simplex centroid mixture designs.

Demonstrates both designs for a 3-component blend. As for all pyoptex
mixtures, only the first q - 1 components are factors; the last one is
one minus the sum of the others.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import simplex_centroid, simplex_lattice
from pyoptex.utils import Factor

# A 3-component blend: Polymer, Filler and (implied) Plasticizer
factors = [
    Factor("Polymer", type="mixture"),
    Factor("Filler", type="mixture"),
]

# {3, 2} simplex lattice: proportions 0, 1/2, 1 -> pure components and 50/50 blends
Y_lattice = simplex_lattice(factors, m=2)
Y_lattice["Plasticizer"] = 1 - Y_lattice.sum(axis=1)
print("=== {3, 2} Simplex lattice ===")
print(f"Runs: {len(Y_lattice)}")
print(Y_lattice)

log_checkpoint("simplex_lattice_shape", list(Y_lattice.shape))

# Simplex centroid: pure components, 50/50 blends and the 1/3-1/3-1/3 centroid
Y_centroid = simplex_centroid(factors)
Y_centroid["Plasticizer"] = 1 - Y_centroid.sum(axis=1)
print("=== Simplex centroid ===")
print(f"Runs: {len(Y_centroid)}")
print(Y_centroid.round(3))

log_checkpoint("simplex_centroid_shape", list(Y_centroid.shape))

# With lower bounds: at least 10% polymer, 20% filler and 10% plasticizer.
# The design is built in pseudocomponents, so the region is still a simplex,
# and augment=True adds 3 axial check blends to test the model's fit.
bounded = [
    Factor("Polymer", type="mixture", levels=[0.1, 1]),
    Factor("Filler", type="mixture", levels=[0.2, 1]),
]
Y_bounded = simplex_centroid(bounded, last_min=0.1, augment=True)
Y_bounded["Plasticizer"] = 1 - Y_bounded.sum(axis=1)
print("=== Simplex centroid with lower bounds and check blends ===")
print(f"Runs: {len(Y_bounded)}")
print(Y_bounded.round(3))

log_checkpoint("simplex_centroid_bounded_shape", list(Y_bounded.shape))
