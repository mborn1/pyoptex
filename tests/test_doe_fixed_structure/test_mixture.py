import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.fixed_structure import Factor, create_fixed_structure_design, create_parameters, default_fn
from pyoptex.doe.fixed_structure.init import init_random
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.utils.model import mixtureY2X, model2Y2X, partial_rsm_names
from tests._helpers import assert_frame_equal, load_reference


def test_mixture():
    ref = load_reference("doe_fixed_structure/mixture")
    set_seed(42)

    nruns = 20
    factors = [
        Factor("A", type="mixture", levels=np.arange(0, 1.001, 0.05)),
        Factor("B", type="mixture", levels=np.arange(0, 1.001, 0.05)),
        Factor("C", type="mixture", levels=np.arange(0.2, 0.501, 0.05)),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]
    assert nruns == ref["nruns"]

    Y2X = mixtureY2X(factors, mixture_effects=(("A", "B", "C"), "tfi"))
    metric = Dopt()

    n_tries = 10
    fn = default_fn(factors, metric, Y2X)
    params = create_parameters(factors, fn, nruns)

    Y, state = create_fixed_structure_design(params, n_tries=n_tries)
    Y["D"] = 1 - Y.sum(axis=1)

    assert list(Y.shape) == ref["Y_shape"]
    assert Y.columns.tolist() == ref["Y_columns"]
    np.testing.assert_allclose(state.metric, ref["metric"], rtol=1e-6)
    assert_frame_equal(Y, {"shape": ref["Y_shape"], "columns": ref["Y_columns"], "data": ref["Y_values"]})


def test_mixture_region_sampling():
    """The I-optimal moments sample the mixture region: each component within the range of its levels, the shares
    summing to 1; without bounds, uniformly on the simplex (4 components, so every share is 1/4 on average)."""
    set_seed(42)
    for factors in (
        [Factor("A", type="mixture", levels=[0.5, 0.8]), Factor("B", type="mixture", levels=[0, 0.1]), Factor("T")],
        [Factor(f"X{i}", type="mixture", levels=[0, 1]) for i in range(3)],
    ):
        Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
        params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), 20)
        Y = init_random(params, 20000, complete=True)
        mixture = Y[:, [f.is_mixture for f in factors]]
        shares = np.column_stack([mixture, 1 - mixture.sum(axis=1)])
        lower = [min(f.levels) for f in factors if f.is_mixture]
        upper = [max(f.levels) for f in factors if f.is_mixture]
        assert np.all((mixture >= lower) & (mixture <= upper)) and np.all(shares[:, -1] >= -1e-6)
    np.testing.assert_allclose(shares.mean(axis=0), 0.25, atol=0.01)
