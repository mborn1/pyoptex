from multiprocessing.reduction import ForkingPickler

import numpy as np
import pandas as pd

from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_parameters
from pyoptex.doe.cost_optimal.cost import combine_costs, cost_fn, fixed_runs_cost, parallel_worker_cost
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


def test_cost_functions_picklable():
    """Parallel generation sends the cost function to worker processes, which requires pickling."""
    params = create_parameters(FACTORS, None)
    Yenc = encode_design(Y_DEC, params.effect_types, coords=params.coords)
    transitions = {"A": 2, "B": 3}
    combined = combine_costs(
        [parallel_worker_cost(transitions, FACTORS, 10), parallel_worker_cost(transitions, FACTORS, 20)]
    )

    for fn in (fixed_runs_cost(10), combined):
        # The unpickled function computes the same costs
        restored = ForkingPickler.loads(ForkingPickler.dumps(fn))
        for (c, m, i), (rc, rm, ri) in zip(fn(Yenc, params), restored(Yenc, params), strict=True):
            np.testing.assert_array_equal(rc, c)
            assert rm == m
            np.testing.assert_array_equal(ri, i)
