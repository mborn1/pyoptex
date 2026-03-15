import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor, cost_fn
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.utils.model import mixtureY2X
from tests._helpers import assert_frame_equal, load_reference


def test_mixture():
    ref = load_reference("doe_cost_optimal/mixture")
    set_seed(42)

    factors = [
        Factor("A", type="mixture", grouped=False, levels=np.arange(0, 1.0001, 0.05)),
        Factor("B", type="mixture", grouped=False, levels=np.arange(0, 1.0001, 0.05)),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    Y2X = mixtureY2X(factors, mixture_effects=(("A", "B"), "tfi"))
    metric = Iopt()

    max_cost = np.array([2.5, 4, 10])

    @cost_fn(denormalize=False, decoded=False, contains_params=False)
    def cost(Y):
        c1 = Y[:, 0]
        c2 = Y[:, 1]
        return [
            (c1, max_cost[0], np.arange(len(Y))),
            (c2, max_cost[1], np.arange(len(Y))),
            (1 - c1 - c2, max_cost[2], np.arange(len(Y))),
        ]

    nsims = 10
    nreps = 1
    fn = default_fn(nsims, factors, cost, metric, Y2X)
    params = create_parameters(factors, fn)

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps)
    Y["C"] = 1 - Y.sum(axis=1)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
