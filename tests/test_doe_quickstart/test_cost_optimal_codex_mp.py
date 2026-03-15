import numpy as np
import pytest

from pyoptex.utils.runtime import set_nb_cores

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:multiprocessing.popen_fork")

set_nb_cores(1)

from pyoptex._seed import set_seed
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from pyoptex.utils.runtime import parallel_generation
from tests._helpers import assert_frame_equal, load_reference


def test_cost_optimal_codex_mp():
    ref = load_reference("doe_quickstart/cost_optimal_codex_mp")
    set_seed(42)

    factors = [
        Factor("A", type="categorical", levels=["L1", "L2", "L3", "L4"]),
        Factor("E", type="continuous", grouped=False),
        Factor("F", type="continuous", grouped=False, min=2, max=5),
        Factor("G", type="continuous", grouped=False),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    model = partial_rsm_names({"A": "tfi", "E": "quad", "F": "quad", "G": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Iopt()
    max_transition_cost = 3 * 4 * 60
    transition_costs = {"A": 2 * 60, "E": 1, "F": 1, "G": 1}
    execution_cost = 5
    cost_fn = parallel_worker_cost(transition_costs, factors, max_transition_cost, execution_cost)

    nsims = 10
    nreps = 10
    fn = default_fn(nsims, factors, cost_fn, metric, Y2X)
    params = create_parameters(factors, fn)

    Y, state = parallel_generation(create_cost_optimal_codex_design, params, nsims=nsims, nreps=nreps, ncores=2)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
