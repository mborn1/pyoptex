import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

from pyoptex._seed import set_seed
from pyoptex.doe.classical import Factor, halton, sobol, uniform

_GENERATORS = [sobol, halton, uniform]


@pytest.mark.parametrize("design", _GENERATORS)
class TestSpaceFilling:
    def test_shape(self, design):
        set_seed(42)
        Y = design([Factor("A"), Factor("B"), Factor("C")], 16)
        assert isinstance(Y, pd.DataFrame)
        assert Y.shape == (16, 3)
        assert list(Y.columns) == ["A", "B", "C"]

    def test_values_in_range(self, design):
        set_seed(42)
        Y = design([Factor("A", min=5, max=15), Factor("B", min=0, max=100)], 32)
        assert Y["A"].between(5, 15).all()
        assert Y["B"].between(0, 100).all()

    def test_deterministic_with_set_seed(self, design):
        factors = [Factor("A"), Factor("B")]
        set_seed(42)
        Y1 = design(factors, 16)
        set_seed(42)
        Y2 = design(factors, 16)
        pd.testing.assert_frame_equal(Y1, Y2)

    def test_different_seeds_differ(self, design):
        factors = [Factor("A"), Factor("B")]
        set_seed(1)
        Y1 = design(factors, 16)
        set_seed(2)
        Y2 = design(factors, 16)
        assert not np.allclose(Y1.values, Y2.values)

    def test_lower_discrepancy_than_random(self, design):
        """Space-filling designs spread more evenly than independent uniform samples."""
        set_seed(42)
        unit = (design([Factor("A"), Factor("B")], 64).to_numpy() + 1) / 2
        random = np.random.uniform(size=(64, 2))
        assert qmc.discrepancy(unit) < qmc.discrepancy(random)

    def test_categorical_raises(self, design):
        with pytest.raises(AssertionError, match="must be continuous"):
            design([Factor("A"), Factor("B", type="categorical", levels=["X", "Y"])], 8)


def test_sobol_warns_if_not_power_of_two():
    with pytest.warns(UserWarning, match="power of 2"):
        sobol([Factor("A"), Factor("B")], 10)


def test_uniform_is_u_type():
    """Every factor takes the n centered levels (2i - 1) / (2n) of its range, each exactly once."""
    set_seed(42)
    n = 7
    Y = uniform([Factor("A", min=0, max=1), Factor("B", min=0, max=1), Factor("C", min=0, max=1)], n)
    levels = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    for col in Y.columns:
        np.testing.assert_allclose(np.sort(Y[col].values), levels, atol=1e-12)
