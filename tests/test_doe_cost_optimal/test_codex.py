from multiprocessing.reduction import ForkingPickler

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyoptex._seed import set_seed

pytestmark = pytest.mark.filterwarnings("ignore:Metric is .*:UserWarning")
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.cost_optimal import Factor
from pyoptex.doe.cost_optimal.codex import create_cost_optimal_codex_design, create_parameters, default_fn
from pyoptex.doe.cost_optimal.cost import fixed_runs_cost, parallel_worker_cost
from pyoptex.doe.cost_optimal.cov import cov_time_trend
from pyoptex.doe.cost_optimal.evaluate import power
from pyoptex.doe.cost_optimal.metric import Dopt, Iopt
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
    prior_before = prior.copy()
    params = create_parameters(factors, fn, prior=prior)
    pd.testing.assert_frame_equal(prior, prior_before)  # the caller's prior is not changed

    Y, state = create_cost_optimal_codex_design(params, nsims=nsims, nreps=nreps, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert len(state.Y) == ref["n_experiments"]
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})

    # After a design run, the parameters (with the time trend) can still go to worker processes
    restored = ForkingPickler.loads(ForkingPickler.dumps(params))
    args = (state.Y, state.X, state.Zs, state.Vinv, state.costs)
    Yr, Xr, _, _ = restored.fn.metric.cov(*args)
    Yp, Xp, _, _ = params.fn.metric.cov(*args)
    np.testing.assert_array_equal(Yr, Yp)
    np.testing.assert_array_equal(Xr, Xp)


@pytest.mark.parametrize("ddf", ["kenward-roger", "satterthwaite"])
@pytest.mark.parametrize("ratio", [1, 0])
def test_power_split_plot(ddf, ratio):
    """
    The balanced split-plot design (4 whole plots of 4 runs, W grouped, S within) as a CODEX design. With ratio 1,
    W is tested against the whole plots (2 df, noncentrality 3.2), S and W * S within them (10 df, noncentrality 16).
    With ratio 0, the grouping only constrains the levels: every term is tested against the residual, as in
    a completely randomized design (12 df, noncentrality 16). This checks that the columns of params.ratios
    are matched to the groupings, with S ungrouped.
    """
    factors = [Factor("W", grouped=True, ratio=ratio), Factor("S", grouped=False)]
    model = partial_rsm_names({"W": "tfi", "S": "tfi"})
    Y = pd.DataFrame({"W": np.repeat([-1, 1, -1, 1], 4), "S": np.tile([-1, 1, -1, 1], 4)})
    fn = default_fn(1, factors, fixed_runs_cost(len(Y)), Dopt(), model2Y2X(model, factors))

    pw = power(Y, create_parameters(factors, fn), ddf=ddf)
    assert list(pw.columns) == ["W", "S", "W * S"]
    if ratio == 1:
        np.testing.assert_allclose(pw["W"], stats.ncf.sf(stats.f.ppf(0.95, 1, 2), 1, 2, 3.2))
        np.testing.assert_allclose(pw[["S", "W * S"]].to_numpy(), stats.ncf.sf(stats.f.ppf(0.95, 1, 10), 1, 10, 16))
    else:
        np.testing.assert_allclose(pw.to_numpy(), stats.ncf.sf(stats.f.ppf(0.95, 1, 12), 1, 12, 16))
