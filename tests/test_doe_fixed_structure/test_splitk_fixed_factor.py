import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import Factor
from pyoptex.doe.fixed_structure.splitk_plot import Plot, create_parameters, create_splitk_plot_design, default_fn
from pyoptex.doe.fixed_structure.splitk_plot.metric import Aopt
from pyoptex.doe.fixed_structure.splitk_plot.utils import validate_plot_sizes
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from tests._helpers import assert_frame_equal, load_reference


def test_splitk_fixed_factor():
    ref = load_reference("doe_fixed_structure/splitk_fixed_factor")
    set_seed(42)

    etc = Plot(level=0, size=4)
    htc = Plot(level=1, size=8)
    nruns = np.prod([etc.size, htc.size])

    factors = [
        Factor("A", htc, type="categorical", levels=["L1", "L2", "L3"]),
        Factor("B", etc, type="continuous"),
        Factor("C", etc, type="continuous"),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert int(nruns) == ref["nruns"]

    model = partial_rsm_names({"A": "tfi", "B": "quad", "C": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Aopt()
    prior = (
        pd.DataFrame(
            [
                ["L1"],
                ["L1"],
                ["L1"],
                ["L1"],
                ["L2"],
                ["L2"],
                ["L2"],
                ["L2"],
                ["L3"],
                ["L3"],
                ["L3"],
                ["L3"],
                ["L2"],
                ["L2"],
                ["L2"],
                ["L2"],
                ["L1"],
                ["L1"],
                ["L1"],
                ["L1"],
                ["L3"],
                ["L3"],
                ["L3"],
                ["L3"],
                ["L1"],
                ["L1"],
                ["L1"],
                ["L1"],
                ["L2"],
                ["L2"],
                ["L2"],
                ["L2"],
            ],
            columns=["A"],
        ).assign(B=0, C=0),
        [Plot(level=0, size=4), Plot(level=1, size=8)],
    )
    grps = [np.array([]), np.arange(nruns), np.arange(nruns)]

    validate_plot_sizes(factors, model)
    n_tries = 10
    fn = default_fn(factors, metric, Y2X)
    params = create_parameters(factors, fn, prior=prior, grps=grps)

    Y, state = create_splitk_plot_design(params, n_tries=n_tries, validate=True)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
