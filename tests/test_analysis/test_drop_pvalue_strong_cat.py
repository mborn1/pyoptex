import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import decode_term, model2Y2X, order_dependencies, partial_rsm_names, term2strong
from pyoptex.analysis import PValueDropRegressor, SimpleRegressor

from tests._helpers import load_reference


def test_drop_pvalue_strong_cat():
    ref = load_reference("analysis/drop_pvalue_strong_cat")
    set_seed(42)

    factors = [
        Factor("A", type="categorical", levels=["A_0", "A_1", "A_2"], coords=[[1, 0], [0, 1], [0, 0]]),
        Factor("B"),
        Factor("C"),
    ]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    N = 200
    data = pd.DataFrame(np.random.rand(N, 0) * 2 - 1, columns=[])
    for factor in factors:
        if factor.is_continuous:
            data[str(factor.name)] = np.random.rand(N) * 2 - 1
        else:
            data[str(factor.name)] = np.random.choice(factor.levels, N, replace=True)
    data["Y"] = (
        2 * np.where(data["A"] == "A_0", 1, 0)
        + 3 * data["C"]
        - 4 * np.where(data["A"] == "A_0", 1, 0) * data["B"]
        + 5
        + np.random.normal(0, 1, N)
    )
    assert list(data.shape) == ref["data_shape"]
    np.testing.assert_allclose(data["Y"].mean(), ref["data_Y_mean"], rtol=1e-10)

    model = partial_rsm_names({str(f.name): "quad" for f in factors})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]
    assert model.values.tolist() == ref["model_values"]

    dependencies = order_dependencies(model, factors)
    assert dependencies.tolist() == ref["dependencies"]

    regr = PValueDropRegressor(factors, Y2X, threshold=0.05, dependencies=dependencies, mode="weak")
    regr.fit(data.drop(columns="Y"), data["Y"])
    assert regr.terms_.tolist() == ref["weak_terms"]
    assert regr.model_formula(model=model) == ref["weak_formula"]

    terms_strong = term2strong(regr.terms_, dependencies)
    terms_strong = decode_term(terms_strong, model, factors)
    assert terms_strong.tolist() == ref["strong_terms"]

    model = model.iloc[terms_strong]
    Y2X = model2Y2X(model, factors)

    regr_simple = SimpleRegressor(factors, Y2X).fit(data.drop(columns="Y"), data["Y"])
    assert regr_simple.model_formula(model=model) == ref["model_formula"]

    data["pred"] = regr_simple.predict(data.drop(columns="Y"))
    np.testing.assert_allclose(data["pred"].mean(), ref["pred_mean"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].std(), ref["pred_std"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].values, ref["predictions"], rtol=1e-10)
