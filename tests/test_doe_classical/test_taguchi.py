from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, taguchi
from pyoptex.doe.classical.taguchi import _ARRAYS

# Printed tables (levels 1, 2, 3) for row-by-row comparison: L4, L8 and L9 as in
# Taguchi's standard tables, L12, L18 and L27 from the University of York tables.
# fmt: off
_PRINTED = {
    "L4": ["111", "122", "212", "221"],
    "L8": ["1111111", "1112222", "1221122", "1222211", "2121212", "2122121", "2211221", "2212112"],
    "L9": ["1111", "1222", "1333", "2123", "2231", "2312", "3132", "3213", "3321"],
    "L12": [
        "11111111111", "11111222222", "11222111222", "12122122112", "12212212121", "12221221211",
        "21221122121", "21212221112", "21122212211", "22211112212", "22121211122", "22112121221",
    ],
    "L18": [
        "11111111", "11222222", "11333333", "12112233", "12223311", "12331122",
        "13121323", "13232131", "13313212", "21133221", "21211332", "21322113",
        "22123132", "22231213", "22312321", "23132312", "23213123", "23321231",
    ],
    "L27": [
        "1111111111111", "1111222222222", "1111333333333", "1222111222333", "1222222333111", "1222333111222",
        "1333111333222", "1333222111333", "1333333222111", "2123123123123", "2123231231231", "2123312312312",
        "2231123231312", "2231231312123", "2231312123231", "2312123312231", "2312231123312", "2312312231123",
        "3132132132132", "3132213213213", "3132321321321", "3213132213321", "3213213321132", "3213321132213",
        "3321132321213", "3321213132321", "3321321213132",
    ],
}

_SHAPES = {
    "L4": (4, 3), "L8": (8, 7), "L9": (9, 4), "L12": (12, 11), "L16": (16, 15), "L18": (18, 8), "L27": (27, 13),
}
# fmt: on


def _array(name):
    """The full array with levels 1, 2, 3 as in the printed tables."""
    return _ARRAYS[name]() + 1


@pytest.mark.parametrize("name", list(_SHAPES))
def test_orthogonal_array_of_strength_two(name):
    """Every pair of columns contains every level combination equally often."""
    A = _array(name)
    assert A.shape == _SHAPES[name]
    for i, j in combinations(range(A.shape[1]), 2):
        pairs, counts = np.unique(A[:, [i, j]], axis=0, return_counts=True)
        assert len(pairs) == len(np.unique(A[:, i])) * len(np.unique(A[:, j])), f"columns {i + 1}, {j + 1}"
        assert len(set(counts)) == 1, f"columns {i + 1}, {j + 1}"


@pytest.mark.parametrize("name", list(_PRINTED))
def test_matches_printed_table(name):
    expected = np.array([[int(c) for c in row] for row in _PRINTED[name]])
    np.testing.assert_array_equal(_array(name), expected)


def test_l16_first_rows():
    A = _array("L16")
    np.testing.assert_array_equal(A[0], 1)
    np.testing.assert_array_equal(A[1], [1] * 7 + [2] * 8)
    np.testing.assert_array_equal(A[:, 0], [1] * 8 + [2] * 8)
    np.testing.assert_array_equal(A[:, 7], [1, 2] * 8)


@pytest.mark.parametrize("name", ["L4", "L8", "L16"])
def test_interaction_table(name):
    """In +/-1 coding, the product of columns a and b is column a XOR b."""
    X = 3 - 2 * _array(name)  # level 1 -> +1, level 2 -> -1
    n_cols = X.shape[1]
    for a, b in combinations(range(1, n_cols + 1), 2):
        np.testing.assert_array_equal(X[:, a - 1] * X[:, b - 1], X[:, (a ^ b) - 1])


def test_l18_interaction_of_first_two_columns():
    """Columns 1 and 2 merged into one 6-level column are orthogonal to the other columns."""
    A = _array("L18")
    merged = 3 * (A[:, 0] - 1) + A[:, 1]
    for j in range(2, 8):
        pairs, counts = np.unique(np.column_stack([merged, A[:, j]]), axis=0, return_counts=True)
        assert len(pairs) == 18 and np.all(counts == 1)


class TestTaguchi:
    def test_l9_four_factors(self):
        factors = [Factor(n, min=0, max=10) for n in "ABCD"]
        Y = taguchi("L9", factors)
        assert isinstance(Y, pd.DataFrame)
        assert Y.shape == (9, 4)
        for col in Y.columns:
            assert sorted(Y[col].unique()) == [0.0, 5.0, 10.0]

    def test_fewer_factors_than_columns(self):
        Y = taguchi("L8", [Factor("A"), Factor("B"), Factor("C")])
        assert Y.shape == (8, 3)

    def test_explicit_columns(self):
        """Assign factors to columns 1, 2 and 4 of the L8, keeping column 3 for the A x B interaction."""
        Y = taguchi("L8", [Factor("A"), Factor("B"), Factor("C")], columns=[1, 2, 4])
        A = 3 - 2 * _array("L8")
        np.testing.assert_array_equal(Y["C"].values, -A[:, 3])

    def test_l18_mixed_levels(self):
        factors = [Factor("Machine", type="categorical", levels=["M1", "M2"])] + [
            Factor(f"X{i}", min=0, max=2) for i in range(7)
        ]
        Y = taguchi("L18", factors)
        assert set(Y["Machine"]) == {"M1", "M2"}
        assert sorted(Y["X0"].unique()) == [0.0, 1.0, 2.0]

    def test_level_mismatch_raises(self):
        factors = [Factor("A", type="categorical", levels=["X", "Y", "Z"])]
        with pytest.raises(AssertionError, match="assign it to another column"):
            taguchi("L18", factors)
        Y = taguchi("L18", factors, columns=[2])
        assert set(Y["A"]) == {"X", "Y", "Z"}

    def test_too_many_factors_raises(self):
        with pytest.raises(AssertionError, match="columns 1 to 3"):
            taguchi("L4", [Factor(f"X{i}") for i in range(4)])

    def test_duplicate_column_raises(self):
        with pytest.raises(AssertionError, match="only once"):
            taguchi("L8", [Factor("A"), Factor("B")], columns=[1, 1])

    def test_unknown_array_raises(self):
        with pytest.raises(AssertionError, match="Unknown array"):
            taguchi("L32", [Factor("A")])
