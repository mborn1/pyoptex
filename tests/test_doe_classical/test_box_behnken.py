import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, box_behnken


class TestBoxBehnken:
    def test_three_factors(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = box_behnken(factors, n_center=3)
        assert isinstance(Y, pd.DataFrame)
        # 3 pairs * 4 + 3 center = 15
        assert len(Y) == 15

    def test_four_factors(self):
        factors = [Factor(f"X{i}") for i in range(4)]
        Y = box_behnken(factors, n_center=1)
        # 6 pairs * 4 + 1 center = 25
        assert len(Y) == 25

    def test_three_levels_only(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = box_behnken(factors)
        for col in Y.columns:
            assert set(Y[col].unique()) <= {-1.0, 0.0, 1.0}

    def test_no_corner_points(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = box_behnken(factors)
        Ynorm = Y.copy()
        for f in factors:
            Ynorm[str(f.name)] = f.normalize(Ynorm[str(f.name)])
        vals = Ynorm.values
        for row in vals:
            nonzero = np.count_nonzero(row)
            assert nonzero <= 2, "Box-Behnken should have at most 2 non-zero factors per run"

    def test_custom_range(self):
        factors = [Factor("A", min=10, max=20), Factor("B", min=0, max=5), Factor("C", min=-3, max=3)]
        Y = box_behnken(factors)
        assert Y["A"].min() >= 10.0
        assert Y["A"].max() <= 20.0

    def test_too_few_factors_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(AssertionError, match="at least 3 factors"):
            box_behnken(factors)

    def test_categorical_raises(self):
        factors = [
            Factor("A"),
            Factor("B"),
            Factor("C", type="categorical", levels=["X", "Y"]),
        ]
        with pytest.raises(AssertionError, match="must be continuous"):
            box_behnken(factors)
