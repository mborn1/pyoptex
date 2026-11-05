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
    summing to 1, and uniformly. Without bounds, every share of 4 components is 1/4 on average. With two components
    of 0 to 5% next to six of 0 to 100% (narrow ones from their box, the others from the simplex), a narrow share has
    the uniform mean 0.02369 = integral of x (1 - x - y)^6 / integral of (1 - x - y)^6 over [0, 0.05]^2. The mixed
    set uses 200,000 samples, so that an error of one in the exponent of the acceptance step (about 1%) is detected."""
    set_seed(42)
    for factors, n, mean, rtol in (
        (
            [Factor("A", type="mixture", levels=[0.5, 0.8]), Factor("B", type="mixture", levels=[0, 0.1]), Factor("T")],
            20000,
            None,
            None,
        ),
        ([Factor(f"X{i}", type="mixture", levels=[0, 0.05 if i < 2 else 1]) for i in range(8)], 200000, 0.02369, 0.005),
        ([Factor(f"X{i}", type="mixture", levels=[0, 1]) for i in range(3)], 20000, 0.25, 0.02),
    ):
        Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
        params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), 20)
        Y = init_random(params, n, complete=True)
        mixture = Y[:, [f.is_mixture for f in factors]]
        shares = np.column_stack([mixture, 1 - mixture.sum(axis=1)])
        lower = [min(f.levels) for f in factors if f.is_mixture]
        upper = [max(f.levels) for f in factors if f.is_mixture]
        assert np.all((mixture >= lower) & (mixture <= upper)) and np.all(shares[:, -1] >= -1e-6)
        if mean is not None:
            np.testing.assert_allclose(shares[:, 0].mean(), mean, rtol=rtol)

    # The last set is uniform on the simplex of 4 shares: mean square 1/10 (a less spread distribution gives less)
    np.testing.assert_allclose(np.mean(shares**2, axis=0), 0.1, rtol=0.03)
