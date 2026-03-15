import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import Factor, create_fixed_structure_design, create_parameters, default_fn
from pyoptex.doe.fixed_structure.metric import Aliasing
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from tests._helpers import assert_frame_equal, load_reference


def test_approx_omars():
    ref = load_reference("doe_fixed_structure/approx_omars")
    set_seed(42)

    nruns = 30
    factors = [
        Factor("A", type="continuous"),
        Factor("B", type="continuous"),
        Factor("C", type="continuous"),
        Factor("D", type="continuous"),
        Factor("E", type="continuous"),
        Factor("F", type="continuous"),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert nruns == ref["nruns"]

    model = partial_rsm_names({"A": "quad", "B": "quad", "C": "quad", "D": "quad", "E": "quad", "F": "quad"})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    n1, n2 = len(factors), len(model) - 2 * len(factors) - 1
    w1, w2 = 1 / ((n1 + 1) * (n1 + 1)), 1 / ((n2 + n1) * (n1 + 1))
    W = np.block(
        [
            [w1 * np.ones((1, 1)), w1 * np.ones((1, n1)), w2 * np.ones((1, n2)), w2 * np.zeros((1, n1))],
            [w1 * np.ones((n1, 1)), w1 * np.ones((n1, n1)), w2 * np.ones((n1, n2)), w2 * np.ones((n1, n1))],
        ]
    )
    W[np.arange(len(W)), np.arange(len(W))] = 0

    main_effects = np.arange(len(factors) + 1)
    metric = Aliasing(effects=main_effects, alias=np.arange(len(model)), W=W)

    n_tries = 10
    fn = default_fn(factors, metric, Y2X)
    params = create_parameters(factors, fn, nruns)

    Y, state = create_fixed_structure_design(params, n_tries=n_tries)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})
