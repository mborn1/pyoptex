import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_fixed_structure_design, create_parameters, default_fn
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

from tests._helpers import assert_frame_equal, load_reference


def test_splitplot_fs():
    ref = load_reference("doe_quickstart/splitplot_fs")
    set_seed(42)

    nruns = 20
    nplots = 5
    assert nruns // nplots == nruns / nplots
    re = RandomEffect(np.repeat(np.arange(nplots), nruns // nplots), ratio=0.1)

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

    metric = Dopt()
    n_tries = 10
    fn = default_fn(factors, metric, Y2X)
    params = create_parameters(factors, fn, nruns)

    Y, state = create_fixed_structure_design(params, n_tries=n_tries)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
