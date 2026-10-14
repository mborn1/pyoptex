import pandas as pd
import pytest

from pyoptex.doe import cost_optimal, fixed_structure
from pyoptex.doe.classical import (
    Factor,
    box_behnken,
    central_composite,
    definitive_screening,
    full_factorial,
    plackett_burman,
    taguchi,
)
from pyoptex.doe.classical.evaluate import design_info
from pyoptex.utils.model import model2Y2X, partial_rsm_names

_FACTOR_CLASSES = [Factor, fixed_structure.Factor, cost_optimal.Factor]


@pytest.mark.parametrize("factor_class", _FACTOR_CLASSES)
@pytest.mark.parametrize(
    "design",
    [
        lambda f: full_factorial(f),
        lambda f: plackett_burman(f),
        lambda f: box_behnken(f),
        lambda f: central_composite(f),
        lambda f: definitive_screening(f),
        lambda f: taguchi("L9", f),
    ],
)
def test_classical_designs_accept_every_factor_class(factor_class, design):
    """The same design results whichever pyoptex Factor class is used."""
    names = ["A", "B", "C", "D"]
    Y = design([factor_class(n, min=0, max=10) for n in names])
    Y_ref = design([Factor(n, min=0, max=10) for n in names])
    pd.testing.assert_frame_equal(Y, Y_ref)


def test_classical_design_with_optimizer_factors():
    """A classical design built from fixed_structure factors can be evaluated and passed to the optimizer."""
    factors = [fixed_structure.Factor(n) for n in "ABC"]
    Y2X = model2Y2X(partial_rsm_names({n: "quad" for n in "ABC"}), factors)
    Y = central_composite(factors, alpha="face-centered")
    assert design_info(Y, factors, Y2X, n_samples=100)["n_runs"] == len(Y)
    params = fixed_structure.create_parameters(factors, fixed_structure.default_fn(factors, None, Y2X), len(Y))
    assert params.nruns == len(Y)
