from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import (
    Factor,
    foldover,
    fractional_factorial,
    full_factorial,
    min_aberration_generators,
    plackett_burman,
)


def _factors(k):
    return [Factor(f"X{i}") for i in range(k)]


def _coded(Y, factors):
    return Y[[str(f.name) for f in factors]].to_numpy(dtype=np.float64)


def _resolution_iii_7_factors():
    """The saturated 2^(7-4) of resolution III."""
    factors = _factors(7)
    return fractional_factorial(factors, min_aberration_generators(factors, n_runs=8)), factors


class TestFoldover:
    def test_plackett_burman_12(self):
        """Fold-over of PB12 with 11 factors: 24 runs, and every product of three columns sums to 0."""
        factors = _factors(11)
        Yf = foldover(plackett_burman(factors), factors)
        X = _coded(Yf, factors)
        assert X.shape == (24, 11)
        for cols in combinations(range(11), 3):
            assert abs(np.prod(X[:, list(cols)], axis=1).sum()) < 1e-10

    def test_full_foldover_gives_resolution_iv(self):
        """The 2^(7-4) has resolution III; after the full fold-over no product of three columns is constant."""
        Y, factors = _resolution_iii_7_factors()
        X0 = _coded(Y, factors)
        assert any(abs(np.prod(X0[:, list(c)], axis=1).sum()) == 8 for c in combinations(range(7), 3))
        X = _coded(foldover(Y, factors), factors)
        assert len(X) == 16
        for cols in combinations(range(7), 3):
            assert abs(np.prod(X[:, list(cols)], axis=1).sum()) < 16

    def test_single_factor_foldover(self):
        """Folding on X0 frees X0 and its two-factor interactions from the other main effects and interactions."""
        Y, factors = _resolution_iii_7_factors()
        X = _coded(foldover(Y, factors, fold=["X0"]), factors)
        np.testing.assert_array_equal(X[8:, 1:], X[:8, 1:])  # only X0 is switched
        with_a = [X[:, 0]] + [X[:, 0] * X[:, j] for j in range(1, 7)]
        without_a = [X[:, j] for j in range(1, 7)] + [X[:, i] * X[:, j] for i, j in combinations(range(1, 7), 2)]
        for u in with_a:
            for v in without_a:
                assert abs(u @ v) < 1e-10

    def test_center_points_stay_center_points(self):
        factors = [Factor("A", min=0, max=10), Factor("B", min=2, max=4), Factor("C", min=-1, max=1)]
        Y = fractional_factorial(factors, {"C": ["A", "B"]}, n_center=2)
        Yf = foldover(Y, factors)
        np.testing.assert_allclose(Yf.iloc[-2:][["A", "B", "C"]].values, [[5, 3, 0], [5, 3, 0]])

    def test_reflection_in_real_units(self):
        factors = [Factor("A", min=10, max=20), Factor("B", levels=[2, 4], min=0, max=10)]
        Y = pd.DataFrame({"A": [10.0, 20.0, 12.0], "B": [2.0, 4.0, 3.0]})
        Yf = foldover(Y, factors)
        assert Yf["A"].tolist()[3:] == [20.0, 10.0, 18.0]
        assert Yf["B"].tolist()[3:] == [4.0, 2.0, 3.0]  # reflected within the explicit levels

    def test_two_level_categorical_swaps(self):
        factors = [Factor("A"), Factor("Cat", type="categorical", levels=["Lo", "Hi"])]
        Y = full_factorial(factors)
        Yf = foldover(Y, factors)
        assert Yf["Cat"].tolist()[4:] == [{"Lo": "Hi", "Hi": "Lo"}[v] for v in Y["Cat"]]

    def test_three_level_categorical_raises(self):
        factors = [Factor("A"), Factor("Cat", type="categorical", levels=["X", "Y", "Z"])]
        Y = full_factorial(factors)
        with pytest.raises(AssertionError, match="only two-level factors can be folded"):
            foldover(Y, factors)
        Yf = foldover(Y, factors, fold=["A"])  # folding only the continuous factor is fine
        assert Yf["Cat"].tolist() == Y["Cat"].tolist() * 2

    def test_block_column_added(self):
        Y, factors = _resolution_iii_7_factors()
        Yf = foldover(Y, factors)
        assert Yf["Block"].tolist() == [1] * 8 + [2] * 8
        assert list(Yf.index) == list(range(16))

    def test_existing_blocks_are_offset(self):
        factors = _factors(3)
        Y = full_factorial(factors, n_blocks=2)
        Yf = foldover(Y, factors)
        assert Yf["Block"].tolist() == Y["Block"].tolist() + [b + 2 for b in Y["Block"]]
        assert "confounded_with_blocks" not in Yf.attrs

    def test_other_columns_kept_for_original_runs(self):
        factors = _factors(2)
        Y = full_factorial(factors)
        Y["Response"] = [1.0, 2.0, 3.0, 4.0]
        Yf = foldover(Y, factors)
        assert Yf["Response"].tolist()[:4] == [1.0, 2.0, 3.0, 4.0]
        assert Yf["Response"].iloc[4:].isna().all()

    def test_unknown_factor_in_fold_raises(self):
        factors = _factors(2)
        with pytest.raises(AssertionError, match="not one of the factors"):
            foldover(full_factorial(factors), factors, fold=["Z"])
