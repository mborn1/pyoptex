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
    "L16b": (16, 5), "L25": (25, 6), "L32": (32, 31), "L32b": (32, 10), "L36": (36, 23), "L50": (50, 12),
    "L54": (54, 26), "L64": (64, 63), "L64b": (64, 21), "L81": (81, 40),
}

# Number of levels of every column
_LEVELS = {
    "L4": [2] * 3, "L8": [2] * 7, "L9": [3] * 4, "L12": [2] * 11, "L16": [2] * 15, "L16b": [4] * 5,
    "L18": [2] + [3] * 7, "L25": [5] * 6, "L27": [3] * 13, "L32": [2] * 31, "L32b": [2] + [4] * 9,
    "L36": [2] * 11 + [3] * 12, "L50": [2] + [5] * 11, "L54": [2] + [3] * 25, "L64": [2] * 63, "L64b": [4] * 21,
    "L81": [3] * 40,
}
# fmt: on

# York's printed two-level L32 and L64, verbatim. The L64 page lists only the first 31 of its
# 63 columns, and prints run 31, columns 28 and 29 as 1 instead of 2.
# fmt: off
_YORK_TWO_LEVEL = {
    "L32": [
        "1111111111111111111111111111111", "1111111111111112222222222222222", "1111111222222221111111122222222",
        "1111111222222222222222211111111", "1112222111122221111222211112222", "1112222111122222222111122221111",
        "1112222222211111111222222221111", "1112222222211112222111111112222", "1221122112211221122112211221122",
        "1221122112211222211221122112211", "1221122221122111122112222112211", "1221122221122112211221111221122",
        "1222211112222111122221111222211", "1222211112222112211112222111122", "1222211221111221122221122111122",
        "1222211221111222211112211222211", "2121212121212121212121212121212", "2121212121212122121212121212121",
        "2121212212121211212121221212121", "2121212212121212121212112121212", "2122121121221211212212112122121",
        "2122121121221212121121221211212", "2122121212112121212212121211212", "2122121212112122121121212122121",
        "2211221122112211221122112211221", "2211221122112212112211221122112", "2211221211221121221122121122112",
        "2211221211221122112211212211221", "2212112122121121221211212212112", "2212112122121122112122121121221",
        "2212112211212211221211221121221", "2212112211212212112122112212112",
    ],
    "L64": [
        "1111111111111111111111111111111", "1111111111111111111111111111111", "1111111111111112222222222222222",
        "1111111111111112222222222222222", "1111111222222221111111122222222", "1111111222222221111111122222222",
        "1111111222222222222222211111111", "1111111222222222222222211111111", "1112222111122221111222211112222",
        "1112222111122221111222211112222", "1112222111122222222111122221111", "1112222111122222222111122221111",
        "1112222222211111111222222221111", "1112222222211111111222222221111", "1112222222211112222111111112222",
        "1112222222211112222111111112222", "1221122112211221122112211221122", "1221122112211221122112211221122",
        "1221122112211222211221122112211", "1221122112211222211221122112211", "1221122221122111122112222112211",
        "1221122221122111122112222112211", "1221122221122112211221111221122", "1221122221122112211221111221122",
        "1222211112222111122221111222211", "1222211112222111122221111222211", "1222211112222112211112222111122",
        "1222211112222112211112222111122", "1222211221111221122221122111122", "1222211221111221122221122111122",
        "1222211221111222211112211221111", "1222211221111222211112211222211", "2121212121212121212121212121212",
        "2121212121212121212121212121212", "2121212121212122121212121212121", "2121212121212122121212121212121",
        "2121212212121211212121221212121", "2121212212121211212121221212121", "2121212212121212121212112121212",
        "2121212212121212121212112121212", "2122121121221211212212112122121", "2122121121221211212212112122121",
        "2122121121221212121121221211212", "2122121121221212121121221211212", "2122121212112121212212121211212",
        "2122121212112121212212121211212", "2122121212112122121121212122121", "2122121212112122121121212122121",
        "2211221122112211221122112211221", "2211221122112211221122112211221", "2211221122112212112211221122112",
        "2211221122112212112211221122112", "2211221211221121221122121122112", "2211221211221121221122121122112",
        "2211221211221122112211212211221", "2211221211221122112211212211221", "2212112122121121221211212212112",
        "2212112122121121221211212212112", "2212112122121122112122121121221", "2212112122121122112122121121221",
        "2212112211212211221211221121221", "2212112211212211221211221121221", "2212112211212212112122112212112",
        "2212112211212212112122112212112",
    ],
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


@pytest.mark.parametrize("name", list(_LEVELS))
def test_levels_per_column(name):
    A = _array(name)
    assert A.shape == _SHAPES[name]
    assert [len(np.unique(A[:, c])) for c in range(A.shape[1])] == _LEVELS[name]
    np.testing.assert_array_equal(A.min(axis=0), 1)


def test_every_array_is_tested():
    assert set(_ARRAYS) == set(_SHAPES) == set(_LEVELS)


@pytest.mark.parametrize("name", list(_PRINTED))
def test_matches_printed_table(name):
    expected = np.array([[int(c) for c in row] for row in _PRINTED[name]])
    np.testing.assert_array_equal(_array(name), expected)


def test_l32_matches_york():
    expected = np.array([[int(c) for c in row] for row in _YORK_TWO_LEVEL["L32"]])
    np.testing.assert_array_equal(_array("L32"), expected)


def test_l64_matches_york_except_misprint():
    """York lists the first 31 of the 63 columns; only its misprint in run 31, columns 28 and 29 differs."""
    york = np.array([[int(c) for c in row] for row in _YORK_TWO_LEVEL["L64"]])
    generated = _array("L64")[:, :31]
    differ = np.argwhere(generated != york)
    assert differ.tolist() == [[30, 27], [30, 28]]
    assert york[30, 27:29].tolist() == [1, 1] and generated[30, 27:29].tolist() == [2, 2]


def test_corrected_york_misprints():
    """The L36 and L64b contain corrections of misprints in the York tables, which make them orthogonal."""
    np.testing.assert_array_equal(_array("L36")[32, 14:20], [2, 1, 2, 1, 1, 3])  # York: 1 2 1 1 3 1
    np.testing.assert_array_equal(_array("L64b")[38:40, 10], [2, 1])  # York: 1, 2


def test_l16_first_rows():
    A = _array("L16")
    np.testing.assert_array_equal(A[0], 1)
    np.testing.assert_array_equal(A[1], [1] * 7 + [2] * 8)
    np.testing.assert_array_equal(A[:, 0], [1] * 8 + [2] * 8)
    np.testing.assert_array_equal(A[:, 7], [1, 2] * 8)


@pytest.mark.parametrize("name", ["L4", "L8", "L16", "L32", "L64"])
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
            taguchi("L128", [Factor("A")])

    def test_l50_mixed_levels(self):
        """Continuous factors take the number of levels of their column: 2 in column 1, 5 in the others."""
        factors = [Factor("A", min=0, max=1)] + [Factor(f"X{i}", min=0, max=8) for i in range(3)]
        Y = taguchi("L50", factors)
        assert Y.shape == (50, 4)
        assert sorted(Y["A"].unique()) == [0.0, 1.0]
        assert sorted(Y["X0"].unique()) == [0.0, 2.0, 4.0, 6.0, 8.0]
