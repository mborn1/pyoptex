import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names
from pyoptex.analysis import SamsRegressor

from tests._helpers import assert_summary_equal, load_reference


def test_sams_generic():
    ref = load_reference("analysis/sams_generic")
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

    dependencies = order_dependencies(model, factors)

    regr = SamsRegressor(
        factors,
        Y2X,
        dependencies=dependencies,
        mode="weak",
        forced_model=np.array([0], np.int64),
        model_size=8,
        nb_models=5000,
        skipn=3000,
    )
    regr.fit(data.drop(columns="Y"), data["Y"])

    assert_summary_equal(regr.summary(), ref["summary"])
    assert len(regr.models_) == ref["nb_models"]
    assert [regr.model_formula(model=model, idx=i) for i in range(len(regr.models_))] == ref["model_formulas"]
    assert [m.tolist() for m in regr.models_] == ref["selected_models"]

    data["pred"] = regr.predict(data.drop(columns="Y"))
    np.testing.assert_allclose(data["pred"].mean(), ref["pred_mean"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].std(), ref["pred_std"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].values, ref["predictions"], rtol=1e-10)
