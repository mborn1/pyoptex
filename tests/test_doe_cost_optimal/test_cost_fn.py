from multiprocessing.reduction import ForkingPickler

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_parameters
from pyoptex.doe.cost_optimal.cost import (
    combine_costs,
    cost_fn,
    fixed_runs_cost,
    level_parallel_worker_cost,
    level_single_worker_cost,
    max_changes_cost,
    parallel_worker_cost,
    single_worker_cost,
)
from pyoptex.utils.design import encode_design

FACTORS = [
    Factor("A", type="categorical", levels=["L1", "L2"]),
    Factor("B", type="continuous", min=10, max=40),
]
# Decoded, normalized design: A as level index, B in [-1, 1]
Y_DEC = np.array([[0.0, -1.0], [1.0, 1.0], [1.0, 0.0]])


def _received(**kwargs):
    # Return what the decorated cost function is called with
    params = create_parameters(FACTORS, None)
    Yenc = encode_design(Y_DEC, params.effect_types, coords=params.coords)
    received = []

    def cost(Y):
        received.append(Y)
        return [(np.ones(len(Y)), 1, np.arange(len(Y)))]

    decorated = cost_fn(**kwargs)(cost) if kwargs else cost_fn(cost)
    decorated(Yenc, params)
    return received[0], Yenc


def test_cost_fn_denormalized():
    Y, _ = _received()
    expected = pd.DataFrame({"A": ["L1", "L2", "L2"], "B": [10.0, 40.0, 25.0]})
    pd.testing.assert_frame_equal(Y, expected, check_dtype=False)


def test_cost_fn_decoded():
    Y, _ = _received(denormalize=False)
    expected = pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [-1.0, 1.0, 0.0]})
    pd.testing.assert_frame_equal(Y, expected, check_dtype=False)


def test_cost_fn_encoded():
    Y, Yenc = _received(denormalize=False, decoded=False)
    assert isinstance(Y, np.ndarray)
    np.testing.assert_array_equal(Y, Yenc)


def test_cost_fn_bound_method():
    """A bound method with a docstring can be a cost function."""

    class Costs:
        def __init__(self, max_runs):
            self.max_runs = max_runs

        def runs(self, Y, params):
            """One unit of cost per run."""
            return [(np.ones(len(Y)), self.max_runs, np.arange(len(Y)))]

    obj = Costs(10)
    fn = cost_fn(obj.runs, denormalize=False, decoded=False, contains_params=True)

    params = create_parameters(FACTORS, None)
    Yenc = encode_design(Y_DEC, params.effect_types, coords=params.coords)
    for (c, m, i), (rc, rm, ri) in zip(obj.runs(Yenc, params), fn(Yenc, params), strict=True):
        np.testing.assert_array_equal(rc, c)
        assert rm == m
        np.testing.assert_array_equal(ri, i)


def test_cost_functions_picklable():
    """Parallel generation sends the cost function to worker processes, which requires pickling."""
    params = create_parameters(FACTORS, None)
    Yenc = encode_design(Y_DEC, params.effect_types, coords=params.coords)
    transitions = {"A": 2, "B": 3}
    combined = combine_costs(
        [parallel_worker_cost(transitions, FACTORS, 10), parallel_worker_cost(transitions, FACTORS, 20)]
    )

    for fn in (fixed_runs_cost(10), combined, max_changes_cost("A", FACTORS, 3)):
        # The unpickled function computes the same costs
        restored = ForkingPickler.loads(ForkingPickler.dumps(fn))
        for (c, m, i), (rc, rm, ri) in zip(fn(Yenc, params), restored(Yenc, params), strict=True):
            np.testing.assert_array_equal(rc, c)
            assert rm == m
            np.testing.assert_array_equal(ri, i)


def test_level_worker_costs():
    """Per-run costs: execution cost, plus the level costs, plus the change from the previous run (from-to matrix)."""
    factors = [
        Factor("A", type="categorical", levels=["L1", "L2", "L3"]),
        Factor("T", type="continuous", levels=[20, 40, 60], min=20, max=60),
    ]
    transition_costs = {"A": np.array([[0, 3, 7], [1, 0, 4], [2, 6, 0]]), "T": 5}
    level_costs = {"T": np.array([0, 2, 5])}

    # A: L1 -> L3 -> L1 -> L1 (7, then 2 back), T: 20 -> 20 -> 60 -> 40 (normalized -1, -1, 1, 0)
    params = create_parameters(factors, None)
    Y = np.array([[0.0, -1.0], [2.0, -1.0], [0.0, 1.0], [0.0, 0.0]])
    Yenc = encode_design(Y, params.effect_types, coords=params.coords)

    # Execution cost 1, level costs 0, 0, 5, 2, and the changes: none, A 7, max(A 2, T 5) or 2 + 5, T 5
    parallel = level_parallel_worker_cost(transition_costs, factors, 100, 1, level_costs)
    single = level_single_worker_cost(transition_costs, factors, 100, 1, level_costs)
    np.testing.assert_allclose(parallel(Yenc, params)[0][0], [1, 8, 11, 8])
    np.testing.assert_allclose(single(Yenc, params)[0][0], [1, 8, 13, 8])

    # Picklable for multiprocessing: the unpickled function computes the same costs
    for fn in (parallel, single):
        restored = ForkingPickler.loads(ForkingPickler.dumps(fn))
        for (c, m, i), (rc, rm, ri) in zip(fn(Yenc, params), restored(Yenc, params), strict=True):
            np.testing.assert_array_equal(rc, c)
            assert rm == m
            np.testing.assert_array_equal(ri, i)

    # A factor with level costs only takes its levels: a value in between raises an error
    Y = np.array([[0.0, -0.75]])  # T = 25, between the levels 20 and 40
    Yenc = encode_design(Y, params.effect_types, coords=params.coords)
    with pytest.raises(ValueError, match="not one of its levels"):
        parallel(Yenc, params)


def test_single_worker_cost():
    """Per-run costs: execution cost plus the change cost of every factor that changes from the previous run."""
    params = create_parameters(FACTORS, None)

    # Run 1 is the first, then A changes, A and B change, and nothing changes
    Y = np.array([[0.0, -1.0], [1.0, -1.0], [0.0, 1.0], [0.0, 1.0]])
    Yenc = encode_design(Y, params.effect_types, coords=params.coords)

    cost = single_worker_cost({"A": 30, "B": 5}, FACTORS, 100, execution_cost=10)
    np.testing.assert_allclose(cost(Yenc, params)[0][0], [10, 40, 45, 10])
