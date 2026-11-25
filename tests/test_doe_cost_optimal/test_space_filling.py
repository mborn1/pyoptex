import itertools

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.metric import Coverage
from pyoptex.doe.utils.space_filling import coverage, distance_map
from pyoptex.utils.design import encode_design
from tests._helpers import assert_frame_equal, load_reference

LEVELS21 = list(np.linspace(-1, 1, 21))


def _params(factors, budget=1000, nsims=5):
    """CODEX parameters with the coverage criterion, no model and a budget."""
    cost = parallel_worker_cost({str(f.name): 1 for f in factors}, factors, budget)
    return create_parameters(factors, default_fn(nsims, factors, cost, Coverage(), None))


def _coords(params, Ynorm):
    """The distance coordinates of a normalized, decoded design."""
    Yenc = encode_design(np.array(Ynorm, dtype=float), params.effect_types, params.coords)
    offset, W = distance_map(params.factors, params.coords)
    return (Yenc - offset) @ W


def test_coverage():
    """Runs at -1 and 1 and samples at -1, 0 and 1: the distances to the nearest run are 0, 0.5 and 0."""
    params = _params([Factor("x", grouped=False)])
    runs = _coords(params, [[-1], [1]])
    samples = _coords(params, [[-1], [0], [1]])
    np.testing.assert_allclose(coverage(runs, samples, 2), -np.sqrt(0.25 / 3))


def test_distance_map():
    """Every two levels of a categorical factor are at distance 1, also with its own encoding; a continuous range
    counts as 1."""
    factors = [
        Factor("C", type="categorical", levels=["L1", "L2", "L3"], grouped=False),
        Factor(
            "D",
            type="categorical",
            levels=["M1", "M2", "M3", "M4"],
            coords=np.array([[-1, -1, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            grouped=False,
        ),
        Factor("x", min=2, max=5, grouped=False),
    ]
    params = _params(factors)
    # (L1, M1, 2), (L2, M1, 2), (L1, M1, 2), (L1, M1, 5), then M2, M3 and M4 with (L1, 2)
    F = _coords(params, [[0, 0, -1], [1, 0, -1], [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 2, -1], [0, 3, -1]])

    def distance(i, j):
        return np.linalg.norm(F[i] - F[j])

    np.testing.assert_allclose([distance(0, 1), distance(0, 2), distance(0, 3)], [1, 0, 1])
    np.testing.assert_allclose([distance(i, j) for i, j in itertools.combinations([0, 4, 5, 6], 2)], 1)


def test_codex_uses_the_budget():
    """The score improves with every run added, so CODEX spreads about 6 runs over the range (maximin would stop at 2)."""
    set_seed(42)
    params = _params([Factor("x", grouped=False, levels=LEVELS21)], budget=12)
    Y, _ = create_cost_optimal_codex_design(params, nsims=5, nreps=1)
    x = np.sort(Y["x"].to_numpy())
    assert len(x) >= 5
    assert np.diff(x).max() < 0.5


def test_codex_categorical_factors():
    """Only categorical factors: a budget for all 9 combinations covers each of them (a fully covered region)."""
    set_seed(42)
    factors = [Factor(name, type="categorical", levels=["L1", "L2", "L3"], grouped=False) for name in ("A", "B")]
    Y, _ = create_cost_optimal_codex_design(_params(factors, budget=20, nsims=20), nsims=20, nreps=1)
    assert set(zip(Y["A"], Y["B"], strict=True)) == set(itertools.product(["L1", "L2", "L3"], repeat=2))


def test_codex_coverage():
    """The CODEX coverage example: a prior of 4 runs in a corner, and a budget for about 8 new runs."""
    ref = load_reference("doe_cost_optimal/coverage")
    set_seed(42)

    factors = [
        Factor("A", type="continuous", grouped=False, levels=np.linspace(-1, 1, 11)),
        Factor("B", type="continuous", grouped=False, min=2, max=5, levels=np.linspace(2, 5, 11)),
        Factor("C", type="categorical", grouped=False, levels=["L1", "L2", "L3"]),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    prior = pd.DataFrame([[-1, 2, "L1"], [-0.6, 2.3, "L1"], [-1, 2.6, "L2"], [-0.4, 2, "L3"]], columns=["A", "B", "C"])
    cost_fn = parallel_worker_cost({"A": 1, "B": 1, "C": 1}, factors, 24)

    nsims = 20
    fn = default_fn(nsims, factors, cost_fn, Coverage(), None)
    params = create_parameters(factors, fn, prior=prior)

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=1, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
