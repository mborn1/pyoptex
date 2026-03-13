import numpy as np
import pandas as pd

from pyoptex._seed import set_seed
from pyoptex.utils import Factor
from pyoptex.utils.model import model2Y2X, order_dependencies, partial_rsm_names
from pyoptex.analysis import PValueDropRegressor, QuantileOutliersTransformer

from tests._helpers import assert_summary_equal, load_reference


def test_drop_pvalue_outliers():
    ref = load_reference("analysis/drop_pvalue_outliers")
    set_seed(42)

    factors = [Factor("A"), Factor("B"), Factor("C")]
    assert [str(f.name) for f in factors] == ref["factor_names"]

    N = 200
    data = pd.DataFrame(np.random.rand(N, 3) * 2 - 1, columns=[str(f.name) for f in factors])
    data["Y"] = 2 * data["A"] + 3 * data["C"] - 4 * data["A"] * data["B"] + 5 + np.random.normal(0, 1, N)
    data.loc[np.arange(N // 100) * 100, "Y"] += 100

    assert list(data.shape) == ref["data_shape"]
    np.testing.assert_allclose(data["Y"].mean(), ref["data_Y_mean"], rtol=1e-10)

    model = partial_rsm_names({str(f.name): "quad" for f in factors})
    Y2X = model2Y2X(model, factors)
    assert list(model.shape) == ref["model_shape"]

    dependencies = order_dependencies(model, factors)

    X = data.drop(columns="Y")
    y = data["Y"]

    # Fit without outlier removal
    regr = PValueDropRegressor(factors, Y2X, threshold=0.05, dependencies=dependencies, mode="weak")
    regr.fit(X, y)
    assert regr.model_formula(model=model) == ref["formula_without_outlier_removal"]
    assert regr.terms_.tolist() == ref["terms_without_outlier_removal"]

    # Detect and remove outliers
    outlier_transformer = QuantileOutliersTransformer(factors, Y2X, threshold=5, stat="norm")
    X_clean, y_clean = outlier_transformer.fit_transform(X, y)
    assert int(outlier_transformer.outliers_.sum()) == ref["n_outliers"]
    assert np.where(outlier_transformer.outliers_)[0].tolist() == ref["outlier_indices"]

    # Fit with outlier removal
    regr = PValueDropRegressor(factors, Y2X, threshold=0.05, dependencies=dependencies, mode="weak")
    regr.fit(X_clean, y_clean)
    assert regr.model_formula(model=model) == ref["formula_with_outlier_removal"]
    assert regr.terms_.tolist() == ref["terms_with_outlier_removal"]
    assert_summary_equal(regr.summary(), ref["summary"])

    data["pred"] = regr.predict(data.drop(columns="Y"))
    np.testing.assert_allclose(data["pred"].mean(), ref["pred_mean"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].std(), ref["pred_std"], rtol=1e-10)
    np.testing.assert_allclose(data["pred"].values, ref["predictions"], rtol=1e-10)
