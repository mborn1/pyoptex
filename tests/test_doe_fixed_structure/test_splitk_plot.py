from multiprocessing.reduction import ForkingPickler

import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.fixed_structure import Factor
from pyoptex.doe.fixed_structure.cov import cov_double_time_trend
from pyoptex.doe.fixed_structure.splitk_plot import Plot, create_parameters, create_splitk_plot_design, default_fn
from pyoptex.doe.fixed_structure.splitk_plot.metric import Aopt, Iopt
from pyoptex.doe.fixed_structure.splitk_plot.utils import validate_plot_sizes
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from tests._helpers import assert_frame_equal, load_reference


def test_splitk_plot():
    ref = load_reference("doe_fixed_structure/splitk_plot")
    set_seed(42)

    etc = Plot(level=0, size=4, ratio=1)
    htc = Plot(level=1, size=8, ratio=[0.1, 10])
    nruns = np.prod([etc.size, htc.size])

    factors = [
        Factor("A", htc, type="categorical", levels=["L1", "L2", "L3"]),
        Factor("B", etc, type="continuous"),
        Factor("C", etc, type="continuous", min=2, max=5),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert int(nruns) == ref["nruns"]

    model = partial_rsm_names({"A": "tfi", "B": "quad", "C": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Aopt(cov=cov_double_time_trend(htc.size, etc.size, nruns))
    prior = (
        pd.DataFrame(
            [
                ["L1", 0, 2],
                ["L1", 1, 5],
                ["L2", -1, 3.5],
                ["L2", 0, 2],
            ],
            columns=["A", "B", "C"],
        ),
        [Plot(level=0, size=2), Plot(level=1, size=2)],
    )
    constraints = parse_constraints_script('(`A` == "L1") & (`B` < -0.5-0.25)', factors, exclude=True)

    validate_plot_sizes(factors, model)
    n_tries = 10
    fn = default_fn(factors, metric, Y2X, constraints=constraints)
    params = create_parameters(factors, fn, prior=prior)

    Y, state = create_splitk_plot_design(params, n_tries=n_tries, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})

    # After a design run, the parameters (with the compiled constraints and the covariates)
    # can still go to worker processes
    restored = ForkingPickler.loads(ForkingPickler.dumps(params))
    np.testing.assert_array_equal(restored.fn.constraints(state.Y), params.fn.constraints(state.Y))
    for a, b in zip(restored.fn.metric.cov(state.Y, state.X), params.fn.metric.cov(state.Y, state.X), strict=True):
        np.testing.assert_array_equal(a, b)


def test_splitk_plot_initial_design_estimable_with_covariates():
    """The initial design must be full rank including the covariates: with a supplier and supplier x Temperature on
    5 whole plots, a start that is only full rank in the factors makes the metric's inverse singular."""
    set_seed(42)

    wp = Plot(level=1, size=5, ratio=1)
    sp = Plot(level=0, size=4, ratio=1)
    factors = [Factor("Temperature", wp), Factor("Pressure", sp)]
    Y2X = model2Y2X(partial_rsm_names({"Temperature": "quad", "Pressure": "lin"}), factors)

    # A whole-plot supplier (+1 / -1) and its interaction with Temperature (the first column of the encoded design)
    supplier = np.repeat([1.0, -1.0, 1.0, -1.0, 1.0], sp.size).reshape(-1, 1)

    def cov(Y, X, random=False, subset=slice(None, None)):
        s = np.random.choice([-1.0, 1.0], size=(Y.shape[0], 1)) if random else supplier[subset]
        return np.concatenate((Y, s), axis=1), np.concatenate((X, s, s * Y[:, :1]), axis=1)

    fn = default_fn(factors, Iopt(cov=cov), Y2X)
    params = create_parameters(factors, fn)
    _, state = create_splitk_plot_design(params, n_tries=3)

    assert np.isfinite(state.metric)
