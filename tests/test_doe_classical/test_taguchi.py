from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import Factor, crossed_array, full_factorial, taguchi
from pyoptex.doe.classical.taguchi import _ARRAYS

# Printed tables (levels 1, 2, 3) of Phadke (1989), Appendix C, for row-by-row comparison.
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
    "L16b": (16, 5), "L25": (25, 6), "L32": (32, 31), "L32b": (32, 10), "L36": (36, 23), "L36b": (36, 16),
    "L50": (50, 12),
    "L54": (54, 26), "L64": (64, 63), "L64b": (64, 21), "L81": (81, 40),
}

# Number of levels of every column
_LEVELS = {
    "L4": [2] * 3, "L8": [2] * 7, "L9": [3] * 4, "L12": [2] * 11, "L16": [2] * 15, "L16b": [4] * 5,
    "L18": [2] + [3] * 7, "L25": [5] * 6, "L27": [3] * 13, "L32": [2] * 31, "L32b": [2] + [4] * 9,
    "L36": [2] * 11 + [3] * 12, "L36b": [2] * 3 + [3] * 13, "L50": [2] + [5] * 11, "L54": [2] + [3] * 25, "L64": [2] * 63, "L64b": [4] * 21,
    "L81": [3] * 40,
}
# fmt: on

# The printed two-level L32 and the first 31 columns of the printed L64 (Phadke, 1989), verbatim,
# including the misprint in the L64: run 31, columns 28 and 29 as 1 instead of 2 (p. 311).
# fmt: off
_PRINTED_TWO_LEVEL = {
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


@pytest.mark.parametrize("name", list(_ARRAYS))
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


def test_l32_matches_printed_table():
    expected = np.array([[int(c) for c in row] for row in _PRINTED_TWO_LEVEL["L32"]])
    np.testing.assert_array_equal(_array("L32"), expected)


def test_l64_matches_printed_table_except_misprint():
    """In the first 31 columns only the misprint in run 31, columns 28 and 29 differs."""
    printed = np.array([[int(c) for c in row] for row in _PRINTED_TWO_LEVEL["L64"]])
    generated = _array("L64")[:, :31]
    differ = np.argwhere(generated != printed)
    assert differ.tolist() == [[30, 27], [30, 28]]
    assert printed[30, 27:29].tolist() == [1, 1] and generated[30, 27:29].tolist() == [2, 2]


def _balanced(a, b):
    """Every combination of the levels of a and b occurs equally often."""
    pairs, counts = np.unique(np.column_stack([a, b]), axis=0, return_counts=True)
    return len(pairs) == len(np.unique(a, axis=0)) * len(np.unique(b)) and len(set(counts)) == 1


class TestL36b:
    """Phadke's L'36 (1989, p. 304)."""

    def test_columns_5_to_16_are_l36_columns_12_to_23(self):
        np.testing.assert_array_equal(_array("L36b")[:, 4:], _array("L36")[:, 11:])

    def test_column_3_is_interaction_of_columns_1_and_2(self):
        X = 3 - 2 * _array("L36b")[:, :3]  # level 1 -> +1, level 2 -> -1
        np.testing.assert_array_equal(X[:, 2], X[:, 0] * X[:, 1])

    @pytest.mark.parametrize("two_level", [0, 1, 2])
    def test_interactions_with_column_4_orthogonal_to_other_columns(self, two_level):
        """The 6-level interaction of column 1, 2 or 3 with column 4 is balanced against every other column."""
        A = _array("L36b")
        interaction = 3 * (A[:, two_level] - 1) + A[:, 3]
        for j in range(16):
            if j not in (two_level, 3):
                assert _balanced(interaction, A[:, j]), f"column {two_level + 1} x 4 vs column {j + 1}"

    def test_columns_1_2_and_4_form_a_12_level_factor(self):
        """With column 3 left empty, columns 1, 2 and 4 form a 12-level factor orthogonal to columns 5-16."""
        A = _array("L36b")
        twelve = 6 * (A[:, 0] - 1) + 3 * (A[:, 1] - 1) + A[:, 3]
        assert len(np.unique(twelve)) == 12
        for j in range(4, 16):
            assert _balanced(twelve, A[:, j]), f"column {j + 1}"

    def test_first_four_columns_per_group_of_three_runs(self):
        A = _array("L36b")
        groups = ["1111", "1221", "2121", "2211", "1112", "1222", "2122", "2212", "1113", "1223", "2123", "2213"]
        for g, pattern in enumerate(groups):
            np.testing.assert_array_equal(A[3 * g : 3 * g + 3, :4], [[int(c) for c in pattern]] * 3)


def test_corrected_misprints():
    """The four misprints in Phadke (1989) that break the orthogonality are corrected."""
    np.testing.assert_array_equal(_array("L36b")[33, 14:16], [3, 1])  # Phadke, p. 304: 2 3
    assert _array("L54")[48, 7] == 3  # Phadke, p. 309: 2
    np.testing.assert_array_equal(_array("L64")[30, 27:29], [2, 2])  # Phadke, p. 311: 1 1
    np.testing.assert_array_equal(_array("L64b")[38:40, 10], [2, 1])  # Phadke, p. 314: 1, 2


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


class TestCrossedArray:
    def test_l8_by_l4(self):
        """Every inner run appears exactly 4 times, once with each outer run."""
        inner = taguchi("L8", [Factor(f"C{i}") for i in range(7)])
        outer = taguchi("L4", [Factor(f"N{i}") for i in range(3)])
        Y = crossed_array(inner, outer)
        assert Y.shape == (32, 10)
        assert list(Y.columns) == list(inner.columns) + list(outer.columns)
        for i in range(8):
            block = Y.iloc[4 * i : 4 * (i + 1)]
            np.testing.assert_array_equal(
                block[inner.columns].to_numpy(), np.repeat(inner.iloc[[i]].to_numpy(), 4, axis=0)
            )
            np.testing.assert_array_equal(block[outer.columns].to_numpy(), outer.to_numpy())

    def test_overlapping_columns_raise(self):
        inner = taguchi("L4", [Factor("A"), Factor("B")])
        outer = taguchi("L4", [Factor("B"), Factor("N")])
        with pytest.raises(AssertionError, match=r"different columns \(both have \['B'\]\)"):
            crossed_array(inner, outer)

    def test_blocked_design_raises(self):
        inner = full_factorial([Factor("A"), Factor("B")], n_blocks=2)
        outer = taguchi("L4", [Factor("N")])
        with pytest.raises(AssertionError, match="has blocks"):
            crossed_array(inner, outer)


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
