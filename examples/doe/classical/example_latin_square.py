#!/usr/bin/env python3

"""
Example: Latin and Graeco-Latin squares.

A 4 x 4 Latin square compares 4 treatments while blocking on 4 operators
and 4 days: every treatment is used once by every operator and once on
every day. A Graeco-Latin square adds a fourth blocking factor (here the
machine), orthogonal to the other three.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import latin_square
from pyoptex.utils import Factor

operator = Factor("Operator", type="categorical", levels=["Ann", "Bob", "Cas", "Dirk"])
day = Factor("Day", type="categorical", levels=["Mon", "Tue", "Wed", "Thu"])
treatment = Factor("Treatment", type="categorical", levels=["A", "B", "C", "D"])
machine = Factor("Machine", type="categorical", levels=["M1", "M2", "M3", "M4"])

Y = latin_square([operator, day, treatment])
print("=== 4 x 4 Latin square ===")
print(Y.pivot(index="Operator", columns="Day", values="Treatment").reindex(columns=day.levels))

log_checkpoint("latin_square_shape", list(Y.shape))

Y2 = latin_square([operator, day, treatment, machine])
Y2["Cell"] = Y2["Treatment"] + "/" + Y2["Machine"]
print("=== 4 x 4 Graeco-Latin square (treatment / machine) ===")
print(Y2.pivot(index="Operator", columns="Day", values="Cell").reindex(columns=day.levels))

# In practice, randomly permute the rows, columns and labels before running it.

log_checkpoint("graeco_latin_square_shape", [len(Y2), 4])
