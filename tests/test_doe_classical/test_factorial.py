import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, full_factorial


class TestFullFactorial:
    def test_two_continuous_default(self):
        """Default continuous factors use 3 levels [-1, 0, 1]."""
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors)
        assert isinstance(Y, pd.DataFrame)
        assert list(Y.columns) == ["A", "B"]
        assert len(Y) == 9  # 3^2

    def test_three_continuous_default(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = full_factorial(factors)
        assert len(Y) == 27  # 3^3, default 3 levels per continuous

    def test_two_level_via_explicit_levels(self):
        """Users can get a 2-level factorial by specifying levels."""
        factors = [Factor("A", levels=[-1, 1]), Factor("B", levels=[-1, 1])]
        Y = full_factorial(factors)
        assert len(Y) == 4  # 2^2

    def test_continuous_custom_levels(self):
        factors = [Factor("A", levels=[-1, 0, 1]), Factor("B", levels=[-1, 0, 1])]
        Y = full_factorial(factors)
        assert len(Y) == 9  # 3^2

    def test_continuous_custom_range(self):
        factors = [Factor("A", min=10, max=20), Factor("B", min=0, max=5)]
        Y = full_factorial(factors)
        assert len(Y) == 9  # 3^2, default 3 levels (min, mid, max)
        assert Y["A"].min() == pytest.approx(10.0)
        assert Y["A"].max() == pytest.approx(20.0)
        assert Y["B"].min() == pytest.approx(0.0)
        assert Y["B"].max() == pytest.approx(5.0)

    def test_categorical(self):
        factors = [
            Factor("A", type="categorical", levels=["L1", "L2", "L3"]),
            Factor("B", type="categorical", levels=["X", "Y"]),
        ]
        Y = full_factorial(factors)
        assert len(Y) == 6  # 3 * 2
        assert set(Y["A"].unique()) == {"L1", "L2", "L3"}
        assert set(Y["B"].unique()) == {"X", "Y"}

    def test_mixed_factors(self):
        factors = [
            Factor("A"),
            Factor("B", type="categorical", levels=["X", "Y"]),
        ]
        Y = full_factorial(factors)
        assert len(Y) == 6  # 3 * 2

    def test_center_points(self):
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors, n_center=3)
        assert len(Y) == 12  # 9 factorial + 3 center
        centers = Y.iloc[9:]
        np.testing.assert_allclose(centers["A"].values, 0.0)
        np.testing.assert_allclose(centers["B"].values, 0.0)

    def test_empty_factors_raises(self):
        with pytest.raises(AssertionError, match="At least one factor"):
            full_factorial([])
