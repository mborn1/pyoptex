import pytest

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed

pytestmark = pytest.mark.filterwarnings("ignore:Metric is .*:UserWarning")
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.cov import cov_time_trend
from pyoptex.doe.cost_optimal.metric import Iopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

from tests._helpers import assert_frame_equal, load_reference


def test_codex():
    ref = load_reference("doe_cost_optimal/codex")
    set_seed(42)

    factors = [
        Factor(
            "A1",
            type="categorical",
            levels=["L1", "L2", "L3", "L4"],
            coords=np.array([[-1, -1, -1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            ratio=[0.1, 1, 10],
        ),
        Factor("E", type="continuous", grouped=False),
        Factor("F", type="continuous", grouped=False, levels=[2, 3, 4, 5], min=2, max=5),
        Factor("G", type="continuous", grouped=False),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    model = partial_rsm_names({"A1": "tfi", "E": "tfi", "F": "quad", "G": "tfi"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Iopt(cov=cov_time_trend(time=60))
    prior = pd.DataFrame([["L1", 0, 2, 0]], columns=["A1", "E", "F", "G"])

    max_transition_cost = 3 * 4 * 60
    transition_costs = {"A1": 2 * 60, "E": 1, "F": 1, "G": 1}
    execution_cost = 5
    cost_fn = parallel_worker_cost(transition_costs, factors, max_transition_cost, execution_cost)

    constraints = parse_constraints_script('(`A1` == "L1") & (`E` < -0.5-0.25)', factors, exclude=True)

    nsims = 10
    nreps = 1
    fn = default_fn(nsims, factors, cost_fn, metric, Y2X, constraints=constraints)
    params = create_parameters(factors, fn, prior=prior)

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
