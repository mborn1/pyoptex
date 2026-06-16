import numpy as np
import pandas as pd
import pytest

from pyoptex._seed import set_seed
from pyoptex.doe.classical import Factor, latin_hypercube


class TestLatinHypercube:
    def test_basic(self):
        set_seed(42)
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = latin_hypercube(factors, n=10)
        assert isinstance(Y, pd.DataFrame)
        assert len(Y) == 10
        assert list(Y.columns) == ["A", "B", "C"]

    def test_values_in_range(self):
        set_seed(42)
        factors = [Factor("A", min=5, max=15), Factor("B", min=0, max=100)]
        Y = latin_hypercube(factors, n=20)
        assert Y["A"].min() >= 5.0
        assert Y["A"].max() <= 15.0
        assert Y["B"].min() >= 0.0
        assert Y["B"].max() <= 100.0

    def test_stratification(self):
        """Each stratum must contain exactly one sample per factor."""
        set_seed(42)
        n = 10
        factors = [Factor("A"), Factor("B")]
        Y = latin_hypercube(factors, n=n, iterations=1)
        for col in ["A", "B"]:
            vals = Y[col].values
            normalized = (vals - (-1)) / 2  # map [-1,1] to [0,1]
            strata = np.floor(normalized * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert len(np.unique(strata)) == n, "Not all strata are covered"

    def test_deterministic_with_seed(self):
        factors = [Factor("A"), Factor("B")]
        set_seed(42)
        Y1 = latin_hypercube(factors, n=10, iterations=5)
        set_seed(42)
        Y2 = latin_hypercube(factors, n=10, iterations=5)
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_categorical_raises(self):
        factors = [Factor("A", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="must be continuous"):
            latin_hypercube(factors, n=5)

    def test_invalid_iterations_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(AssertionError, match="iterations must be at least 1"):
            latin_hypercube(factors, n=5, iterations=0)
