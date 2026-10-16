from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, latin_square


def _factors(t, k):
    """k categorical factors with t levels."""
    names = ["Row", "Column"] + [f"S{a}" for a in range(1, k - 1)]
    return [Factor(n, type="categorical", levels=[f"{n}{lv}" for lv in range(t)]) for n in names]


def _indices(Y, factors):
    """Level indices 0..t-1 of every factor."""
    return np.column_stack([Y[str(f.name)].map({lv: n for n, lv in enumerate(f.levels)}).to_numpy() for f in factors])


class TestLatinSquare:
    @pytest.mark.parametrize("t", [2, 3, 4, 5, 6, 7, 10])
    def test_latin_square(self, t):
        """Every symbol exactly once in every row and every column."""
        factors = _factors(t, 3)
        Y = latin_square(factors)
        assert isinstance(Y, pd.DataFrame)
        assert Y.shape == (t * t, 3)
        X = _indices(Y, factors)
        for r in range(t):
            assert sorted(X[X[:, 0] == r, 2]) == list(range(t))
            assert sorted(X[X[:, 1] == r, 2]) == list(range(t))

    @pytest.mark.parametrize("t", [3, 4, 5, 7, 8, 9])
    def test_complete_set_of_orthogonal_squares(self, t):
        """All t - 1 squares: every pair of the t + 1 factors contains each combination of levels exactly once."""
        factors = _factors(t, t + 1)
        X = _indices(latin_square(factors), factors)
        assert X.shape == (t * t, t + 1)
        for a, b in combinations(range(t + 1), 2):
            assert len({(x, y) for x, y in X[:, [a, b]]}) == t * t, f"factors {a} and {b}"

    def test_graeco_latin_l9(self):
        """For t = 3 the four factors form an orthogonal array of strength 2, like the Taguchi L9."""
        factors = _factors(3, 4)
        X = _indices(latin_square(factors), factors)
        for a, b in combinations(range(4), 2):
            pairs, counts = np.unique(X[:, [a, b]], axis=0, return_counts=True)
            assert len(pairs) == 9 and np.all(counts == 1)

    def test_row_and_column_factors(self):
        factors = _factors(4, 3)
        Y = latin_square(factors)
        assert Y["Row"].tolist() == [f"Row{i}" for i in range(4) for _ in range(4)]
        assert Y["Column"].tolist() == [f"Column{j}" for _ in range(4) for j in range(4)]

    def test_order_six_graeco_latin_raises(self):
        with pytest.raises(AssertionError, match="No two orthogonal Latin squares of order 6"):
            latin_square(_factors(6, 4))

    def test_order_ten_graeco_latin_raises(self):
        with pytest.raises(AssertionError, match="only supported when 10 is a prime power"):
            latin_square(_factors(10, 4))

    def test_too_many_factors_raises(self):
        with pytest.raises(AssertionError, match="At most 6 factors"):
            latin_square(_factors(5, 7))
        with pytest.raises(AssertionError, match=r"At most 3 factors.*at most 1 mutually orthogonal"):
            latin_square(_factors(2, 4))

    def test_different_number_of_levels_raises(self):
        factors = _factors(3, 3)[:2] + [Factor("S", type="categorical", levels=["a", "b"])]
        with pytest.raises(AssertionError, match="same number of levels"):
            latin_square(factors)

    def test_continuous_factor_raises(self):
        factors = _factors(3, 2) + [Factor("S")]
        with pytest.raises(AssertionError, match="must be categorical"):
            latin_square(factors)

    def test_too_few_factors_raises(self):
        with pytest.raises(AssertionError, match="at least one symbol factor"):
            latin_square(_factors(3, 2))
