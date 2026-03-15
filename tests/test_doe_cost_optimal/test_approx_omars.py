import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import fixed_runs_cost
from pyoptex.doe.cost_optimal.metric import Aliasing
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from tests._helpers import assert_frame_equal, load_reference


def test_approx_omars():
    ref = load_reference("doe_cost_optimal/approx_omars")
    set_seed(42)

    factors = [
        Factor("A", type="continuous", grouped=False),
        Factor("B", type="continuous", grouped=False),
        Factor("C", type="continuous", grouped=False),
        Factor("D", type="continuous", grouped=False),
        Factor("E", type="continuous", grouped=False),
        Factor("F", type="continuous", grouped=False),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    model = partial_rsm_names({"A": "quad", "B": "quad", "C": "quad", "D": "quad", "E": "quad", "F": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    n1, n2 = len(factors), len(model) - 2 * len(factors) - 1
    w1, w2 = 1 / ((n1 + 1) * (n1 + 1)), 1 / ((n2 + n1) * (n1 + 1))
    W = np.block(
        [
            [w1 * np.ones((1, 1)), w1 * np.ones((1, n1)), w2 * np.ones((1, n2)), w2 * np.zeros((1, n1))],
            [w1 * np.ones((n1, 1)), w1 * np.ones((n1, n1)), w2 * np.ones((n1, n2)), w2 * np.ones((n1, n1))],
        ]
    )
    W[np.arange(len(W)), np.arange(len(W))] = 0

    main_effects = np.arange(len(factors) + 1)
    metric = Aliasing(effects=main_effects, alias=np.arange(len(model)), W=W)
    nruns = 30
    cost_fn = fixed_runs_cost(nruns)

    nsims = 10
    nreps = 1
    fn = default_fn(nsims, factors, cost_fn, metric, Y2X)
    params = create_parameters(factors, fn)

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
