import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_fixed_structure_design, create_parameters, default_fn
from pyoptex.doe.fixed_structure.cov import cov_double_time_trend
from pyoptex.doe.fixed_structure.evaluate import estimation_variance, evaluate_metrics
from pyoptex.doe.fixed_structure.metric import Aopt, Dopt, Iopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

from tests._helpers import assert_frame_equal, load_reference


def test_strip_plot():
    ref = load_reference("doe_fixed_structure/strip_plot")
    set_seed(42)

    nruns = 20
    nplots = 5
    assert nruns // nplots == nruns / nplots
    re = RandomEffect(np.tile(np.arange(nplots), nruns // nplots), ratio=[0.1, 10])

    factors = [
        Factor("A", re, type="categorical", levels=["L1", "L2", "L3"]),
        Factor("B", type="continuous"),
        Factor("C", type="continuous", min=2, max=5),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert nruns == ref["nruns"]
    assert nplots == ref["nplots"]

    model = partial_rsm_names({"A": "tfi", "B": "quad", "C": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Dopt(cov=cov_double_time_trend(nplots, nruns // nplots, nruns))
    constraints = parse_constraints_script('(`A` == "L1") & (`B` < -0.5-0.25)', factors, exclude=True)

    n_tries = 10
    fn = default_fn(factors, metric, Y2X, constraints=constraints)
    params = create_parameters(factors, fn, nruns)

    Y, state = create_fixed_structure_design(params, n_tries=n_tries)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})

    # Validate evaluate_metrics (tolerance for Monte Carlo-based metrics)
    metrics_result = evaluate_metrics(Y, params, [metric, Dopt(), Iopt(), Aopt()])
    np.testing.assert_allclose(metrics_result, ref["evaluate_metrics"], rtol=0.05)

    # Validate estimation_variance (Monte Carlo-based, mild tolerance)
    ev = estimation_variance(Y, params)
    np.testing.assert_allclose(ev, ref["estimation_variance"], rtol=0.05)
