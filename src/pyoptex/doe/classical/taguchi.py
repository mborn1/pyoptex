"""
Module for Taguchi orthogonal array generation.

Arrays in the layout of Taguchi's standard tables (e.g. Phadke, 1989,
"Quality Engineering Using Robust Design", Appendix C). The L12, L18 and L27
follow the row and column order of the tables of the University of York
(york.ac.uk/depts/maths/tables/orthogonal.htm); the L12 and L18 are the arrays
of the library of orthogonal arrays of N. J. A. Sloane (neilsloane.com/oadir).
"""

import numpy as np

from .utils import _levels, _to_dataframe, _validate_factors

# fmt: off
_L12 = [
    "11111111111", "11111222222", "11222111222", "12122122112", "12212212121", "12221221211",
    "21221122121", "21212221112", "21122212211", "22211112212", "22121211122", "22112121221",
]
_L18 = [
    "11111111", "11222222", "11333333", "12112233", "12223311", "12331122", "13121323", "13232131", "13313212",
    "21133221", "21211332", "21322113", "22123132", "22231213", "22312321", "23132312", "23213123", "23321231",
]
# fmt: on


def _parse(rows):
    """Array with 0-based levels from rows of 1-based level digits."""
    return np.array([[int(c) - 1 for c in row] for row in rows], dtype=np.int64)


def _two_level_array(n_runs):
    """
    Taguchi's two-level array with *n_runs* (a power of 2) rows and
    n_runs - 1 columns: column c is the parity of the bits of the run index
    selected by c (with reversed bit order), so the interaction of columns a
    and b is column ``a XOR b``.
    """
    bits = int(np.log2(n_runs))
    runs = np.arange(n_runs)
    cols = np.arange(1, n_runs)
    rev = np.array([int(format(c, f"0{bits}b")[::-1], 2) for c in cols])
    selected = runs[:, None] & rev[None, :]
    return np.array([[bin(v).count("1") % 2 for v in row] for row in selected], dtype=np.int64)


def _l9():
    """L9: columns A, B, A + B and 2A + B (mod 3) of the 3^2 full factorial."""
    a, b = np.divmod(np.arange(9), 3)
    return np.column_stack([a, b, (a + b) % 3, (2 * a + b) % 3])


def _l27():
    """
    L27: the 3^3 full factorial in A, B, C (columns 1, 2 and 5) and the
    columns xA + yB + zC (mod 3) in the order of the printed table.
    """
    a, rest = np.divmod(np.arange(27), 9)
    b, c = np.divmod(rest, 3)
    coefs = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1),
             (0, 1, 1), (1, 1, 1), (2, 1, 1), (0, 2, 1), (1, 2, 1), (2, 2, 1)]  # fmt: skip
    return np.column_stack([(x * a + y * b + z * c) % 3 for x, y, z in coefs])


_ARRAYS = {
    "L4": lambda: _two_level_array(4),
    "L8": lambda: _two_level_array(8),
    "L9": _l9,
    "L12": lambda: _parse(_L12),
    "L16": lambda: _two_level_array(16),
    "L18": lambda: _parse(_L18),
    "L27": _l27,
}


def taguchi(name, factors, columns=None):
    """
    Generate a Taguchi orthogonal array design.

    Every pair of columns contains every combination of levels equally
    often (an orthogonal array of strength 2), so all main effects are
    orthogonal.  The arrays follow the row and column order of Taguchi's
    standard tables, so factors can be assigned to columns with his
    linear graphs and interaction tables:

    * ``'L4'``, ``'L8'``, ``'L16'`` -- 3, 7 and 15 two-level columns.
      The interaction of columns *a* and *b* falls in column
      ``a XOR b`` (e.g. 1 x 2 in column 3 of the L8).
    * ``'L9'`` -- 4 three-level columns (the 3^(4-2) with C = A + B and
      D = 2A + B).
    * ``'L12'`` -- 11 two-level columns.  Interactions are spread over
      all columns, so it is only suitable for main effects.
    * ``'L18'`` -- column 1 has two levels, columns 2 to 8 three levels.
      The interaction between columns 1 and 2 is orthogonal to all other
      columns and can be estimated without giving up a column.
    * ``'L27'`` -- 13 three-level columns (the 3^(13-10) with base
      columns 1, 2 and 5).  The interaction of two columns occupies two
      other columns (e.g. 1 x 2 in columns 3 and 4).

    Parameters
    ----------
    name : str
        The array: ``'L4'``, ``'L8'``, ``'L9'``, ``'L12'``, ``'L16'``,
        ``'L18'`` or ``'L27'``.
    factors : list(Factor)
        The factors.  Categorical factors and continuous factors with
        explicit ``levels`` must have as many levels as their column;
        continuous factors without explicit levels take the column's
        number of levels, equally spaced from ``min`` to ``max``.
    columns : list(int) or None
        The (1-based) column of the array for each factor, as in the
        printed tables.  ``None`` (default) assigns the factors to
        columns 1, 2, 3, ... in order.

    Returns
    -------
    Y : pd.DataFrame
        The design (denormalized), with only the assigned columns.
    """
    assert name in _ARRAYS, f"Unknown array '{name}'. Use one of {list(_ARRAYS)}."
    _validate_factors(factors)
    array = _ARRAYS[name]()
    n_cols = array.shape[1]

    if columns is None:
        columns = list(range(1, len(factors) + 1))
    assert len(columns) == len(factors), (
        f"columns must give one column per factor ({len(factors)} factors, {len(columns)} columns)"
    )
    assert len(set(columns)) == len(columns), f"Every column can be assigned only once (got {columns})"
    for c in columns:
        assert 1 <= c <= n_cols, f"The {name} has columns 1 to {n_cols} (got {c})"

    design = np.zeros((array.shape[0], len(factors)), dtype=np.float64)
    for j, (f, c) in enumerate(zip(factors, columns)):
        col = array[:, c - 1]
        n_levels = int(col.max()) + 1
        lv = _levels(f, n_levels)
        assert len(lv) == n_levels, (
            f"Factor {f.name} has {len(lv)} levels but column {c} of the {name} has {n_levels}; "
            f"assign it to another column with columns=[...]"
        )
        design[:, j] = lv[col]

    return _to_dataframe(design, factors)
