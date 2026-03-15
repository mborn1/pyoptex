import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.analysis import SimpleRegressor
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, partial_rsm_names
from tests._helpers import load_reference


def test_simple_model_mixedlm():
    ref = load_reference("analysis/simple_model_mixedlm")
    set_seed(42)

    factors = [Factor("A"), Factor("B"), Factor("C")]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    N = 200
    nre = 5
    data = pd.DataFrame(np.random.rand(N, 3) * 2 - 1, columns=[str(f.name) for f in factors])
    data["RE"] = np.array([f"L{i}" for i in range(nre)])[np.repeat(np.arange(nre), N // nre)]
    data["Y"] = (
        2 * data["A"]
        + 3 * data["C"]
        - 4 * data["A"] * data["B"]
        + 5
        + np.random.normal(0, 1, N)
        + np.repeat(np.random.normal(0, 1, nre), N // nre)
    )
    assert list(data.shape) == ref["data_shape"]
    np.testing.assert_allclose(data["Y"].mean(), ref["data_Y_mean"], rtol=1e-10)

    model = partial_rsm_names({str(f.name): "quad" for f in factors})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    random_effects = ("RE",)
    regr = SimpleRegressor(factors, Y2X, random_effects)
    regr.fit(data.drop(columns="Y"), data["Y"])
    assert regr.model_formula(model=model) == ref["model_formula"]

    data["pred"] = regr.predict(data.drop(columns="Y"))
    np.testing.assert_allclose(data["pred"].mean(), ref["pred_mean"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].std(), ref["pred_std"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].values, ref["predictions"], rtol=1e-10)
