import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import (
    Factor,
    RandomEffect,
    create_fixed_structure_design,
    create_parameters,
    default_fn,
)
from pyoptex.doe.fixed_structure.metric import Aopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from tests._helpers import assert_frame_equal, load_reference


def test_strip_plot_augment():
    ref = load_reference("doe_fixed_structure/strip_plot_augment")
    set_seed(42)

    nruns = 20
    nplots = 5
    Z_full = np.concatenate([np.repeat(np.arange(nplots), 3), np.arange(nplots)]).astype(np.int64)
    Z_full[-1] = 5
    re = RandomEffect(Z_full)

    factors = [
        Factor("A", re, type="categorical", levels=["L1", "L2", "L3"]),
        Factor("B", type="continuous"),
        Factor("C", type="continuous"),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    model = partial_rsm_names({"A": "tfi", "B": "quad", "C": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    metric = Aopt()
    prior = pd.DataFrame(
        [
            ["L1", -1.0, 0.0],
            ["L1", 0.0, 1.0],
            ["L1", 1.0, -1.0],
            ["L2", -1.0, -1.0],
            ["L2", 0.0, 0.0],
            ["L2", 1.0, 1.0],
            ["L3", -1.0, 1.0],
            ["L3", 0.0, -1.0],
            ["L3", 1.0, 0.0],
            ["L1", -0.5, 0.5],
            ["L1", 0.5, -0.5],
            ["L1", 0.0, 0.0],
            ["L2", -0.5, -0.5],
            ["L2", 0.5, 0.5],
            ["L2", 0.0, 0.0],
        ],
        columns=["A", "B", "C"],
    )

    n_tries = 10
    fn = default_fn(factors, metric, Y2X)
    params = create_parameters(factors, fn, nruns, prior=prior)

    Y, state = create_fixed_structure_design(params, n_tries=n_tries)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
