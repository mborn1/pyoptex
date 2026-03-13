import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import scaled_parallel_worker_cost
from pyoptex.doe.cost_optimal.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

from tests._helpers import assert_frame_equal, load_reference


def test_scaled():
    ref = load_reference("doe_cost_optimal/scaled")
    set_seed(42)

    factors = [
        Factor("A", type="continuous", min=2, max=5),
        Factor("B", type="continuous"),
        Factor("E", type="continuous", grouped=False),
        Factor("F", type="continuous", grouped=False),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    model = partial_rsm_names({"A": "quad", "B": "quad", "E": "quad", "F": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Dopt()
    max_transition_cost = 3 * 4 * 60
    transition_costs = {
        "A": (0, 0, 2 * 60, 2 * 60),
        "B": (60, 60, 0, 0),
        "E": (1, 1, 0, 0),
        "F": (1, 1, 0, 0),
    }
    execution_cost = 5
    cost_fn = scaled_parallel_worker_cost(transition_costs, factors, max_transition_cost, execution_cost)

    nsims = 10
    nreps = 1
    fn = default_fn(nsims, factors, cost_fn, metric, Y2X)
    params = create_parameters(factors, fn)

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
