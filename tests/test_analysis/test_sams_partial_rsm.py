import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names
from pyoptex.analysis import SamsRegressor

from tests._helpers import load_reference


def test_sams_partial_rsm():
    ref = load_reference("analysis/sams_partial_rsm")
    set_seed(42)

    factors = [Factor("A"), Factor("B"), Factor("C"), Factor("D"), Factor("E"), Factor("F")]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    N = 200
    data = pd.DataFrame(np.random.rand(N, len(factors)) * 2 - 1, columns=[str(f.name) for f in factors])
    data["Y"] = 2 * data["A"] + 3 * data["C"] - 4 * data["A"] * data["B"] + 5 + np.random.normal(0, 1, N)
    assert list(data.shape) == ref["data_shape"]
    np.testing.assert_allclose(data["Y"].mean(), ref["data_Y_mean"], rtol=1e-10)

    model_order = {str(f.name): "quad" for f in factors}
    model = partial_rsm_names(model_order)
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    dependencies = order_dependencies(model, factors)
    assert dependencies.tolist() == ref["dependencies"]

    regr = SamsRegressor(
        factors,
        Y2X,
        dependencies=dependencies,
        mode="weak",
        forced_model=np.array([0], np.int64),
        model_size=8,
        nb_models=5000,
        skipn=3000,
        entropy_model_order=model_order,
    )
    regr.fit(data.drop(columns="Y"), data["Y"])
    assert len(regr.models_) == ref["nb_models"]
    assert [regr.model_formula(model=model, idx=i) for i in range(len(regr.models_))] == ref["model_formulas"]
    assert [m.tolist() for m in regr.models_] == ref["selected_models"]

    for i in range(len(regr.models_)):
        np.testing.assert_allclose(regr.model_coef_[i], ref[f"model_coef_{i}"], rtol=1e-10)
        assert regr.models_[i].tolist() == ref[f"models_{i}"]

    data["pred"] = regr.predict(data.drop(columns="Y"))
    np.testing.assert_allclose(data["pred"].mean(), ref["pred_mean"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].std(), ref["pred_std"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].values, ref["predictions"], rtol=1e-10)
