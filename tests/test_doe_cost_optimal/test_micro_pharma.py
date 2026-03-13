import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.cost_optimal import Factor, cost_fn
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

from tests._helpers import assert_frame_equal, load_reference


def test_micro_pharma():
    ref = load_reference("doe_cost_optimal/micro_pharma")
    set_seed(42)

    factors = [
        Factor("X1", type="continuous", grouped=False, min=-1, max=1, levels=[-1, 0, 1]),
        Factor("X2", type="continuous", grouped=False, min=6, max=36, levels=np.linspace(6, 36, 11)),
        Factor("X3", type="continuous", grouped=False, min=12, max=36, levels=np.linspace(12, 36, 9)),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    model = partial_rsm_names({"X1": "quad", "X2": "quad", "X3": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Dopt()
    constraints = parse_constraints_script("(`X2` <= `X3`)", factors, exclude=False)

    max_units = 150

    @cost_fn(denormalize=False, decoded=False, contains_params=False)
    def cost(Y):
        units = 2 + (Y[:, 0] + 1) * 6
        return [(units, max_units, np.arange(len(Y)))]

    nsims = 10
    nreps = 10
    fn = default_fn(nsims, factors, cost, metric, Y2X, constraints=constraints)
    params = create_parameters(factors, fn)

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
