"""
Module for Taguchi orthogonal array generation.

Arrays in the layout of Taguchi's standard tables (e.g. Phadke, 1989,
"Quality Engineering Using Robust Design", Appendix C; Taguchi, "System of
Experimental Design", Vol. 2). The L12, L16b, L18, L25, L27, L32, L32b, L36,
L50, L54, L64, L64b and L81 follow the row and column order of the tables of
the University of York (york.ac.uk/depts/maths/tables/orthogonal.htm). The
York tables contain misprints in the L36, L64 and L64b, which break their
orthogonality: the L36 and L64b here are corrected (see the comments at the
arrays) and the two-level L64 is generated. The York page of the L64 also
lists only the first 31 of its 63 columns. The L12 and L18 are the arrays of
the library of orthogonal arrays of N. J. A. Sloane (neilsloane.com/oadir).
"""

import numpy as np
import pandas as pd

from .utils import BLOCK_COLUMN, _levels, _to_dataframe, _validate_factors

# fmt: off
_L12 = [
    "11111111111", "11111222222", "11222111222", "12122122112", "12212212121", "12221221211",
    "21221122121", "21212221112", "21122212211", "22211112212", "22121211122", "22112121221",
]
_L18 = [
    "11111111", "11222222", "11333333", "12112233", "12223311", "12331122", "13121323", "13232131", "13313212",
    "21133221", "21211332", "21322113", "22123132", "22231213", "22312321", "23132312", "23213123", "23321231",
]
_L16B = [
    "11111", "12222", "13333", "14444", "21234", "22143", "23412", "24321", "31342", "32431", "33124", "34213",
    "41423", "42314", "43241", "44132",
]
_L25 = [
    "111111", "122222", "133333", "144444", "155555", "212345", "223451", "234512", "245123", "251234", "313524",
    "324135", "335241", "341352", "352413", "414253", "425314", "431425", "442531", "453142", "515432", "521543",
    "532154", "543215", "554321",
]
_L32B = [
    "1111111111", "1122222222", "1133333333", "1144444444", "1211223344", "1222114433", "1233441122", "1244332211",
    "1312341234", "1321432143", "1334123412", "1343214321", "1412433421", "1421344312", "1434211243", "1443122134",
    "2114142323", "2123231414", "2132324141", "2141413232", "2214234132", "2223143241", "2232412314", "2241321423",
    "2313312442", "2324421331", "2331134224", "2342243113", "2413424213", "2424313124", "2431242431", "2442131342",
]
# York prints run 33, columns 15-20 as 1 2 1 1 3 1, which breaks the orthogonality (the
# entries are shifted by one column); every group of 3 runs cycles the 3-level columns,
# so run 33 = run 31 + 2 gives 2 1 2 1 1 3, as used here.
_L36 = [
    "11111111111111111111111", "11111111111222222222222", "11111111111333333333333", "11111222222111122223333",
    "11111222222222233331111", "11111222222333311112222", "11222111222112312331223", "11222111222223123112331",
    "11222111222331231223112", "12122122112113213232132", "12122122112221321313213", "12122122112332132121321",
    "12212212121123132133212", "12212212121231213211323", "12212212121312321322131", "12221221211123211323321",
    "12221221211231322131132", "12221221211312133212213", "21221122121121333122123", "21221122121232111233231",
    "21221122121313222311312", "21212221112122331211332", "21212221112233112322113", "21212221112311223133221",
    "21122212211132123313122", "21122212211213231121233", "21122212211321312232311", "22211112212132221132313",
    "22211112212213332213121", "22211112212321113321232", "22121211122133323221211", "22121211122211131332322",
    "22121211122322212113133", "22112121221131232312231", "22112121221212313123312", "22112121221323121231123",
]
_L50 = [
    "111111111111", "112222222222", "113333333333", "114444444444", "115555555555", "121234512345", "122345123451",
    "123451234512", "124512345123", "125123451234", "131352441352", "132413552413", "133524113524", "134135224135",
    "135241335241", "141425353142", "142531414253", "143142525314", "144253131425", "145314242531", "151543243215",
    "152154354321", "153215415432", "154321521543", "155432132154", "211145432523", "212251543134", "213312154245",
    "214423215351", "215534321412", "221213324554", "222324435115", "223435541221", "224541152332", "225152213443",
    "231331255424", "232442311535", "233553422141", "234114533252", "235225144313", "241454125233", "242515231344",
    "243121342455", "244232453511", "245343514122", "251522534431", "252133145542", "253244251153", "254355312214",
    "255411423325",
]
_L54 = [
    "11111111111111111111111111", "11111111222222222222222222", "11111111333333333333333333",
    "11222222111111232323232323", "11222222222222313131313131", "11222222333333121212121212",
    "11333333111111323232323232", "11333333222222131313131313", "11333333333333212121212121",
    "12112233112233111123233232", "12112233223311222231311313", "12112233331122333312122121",
    "12223311112233232332321111", "12223311223311313113132222", "12223311331122121221213333",
    "12331122112233323211112323", "12331122223311131322223131", "12331122331122212133331212",
    "13121323121323112311322332", "13121323232131223122133113", "13121323313212331233211221",
    "13232131121323233223113211", "13232131232131311331221322", "13232131313212122112332133",
    "13313212121323321132231123", "13313212232131132213312231", "13313212313212213321123312",
    "21133221133221113232232311", "21133221211332221313313122", "21133221322113332121121233",
    "21211332133221231111323223", "21211332211332312222131331", "21211332322113123333212112",
    "21322113133221322323111132", "21322113211332133131222213", "21322113322113211212333321",
    "22123132123132112332113223", "22123132231213223113221331", "22123132312321331221332112",
    "22231213123132233211231132", "22231213231213311322312213", "22231213312321122133123321",
    "22312321123132321123322311", "22312321231213132231133122", "22312321312321213312211233",
    "23132312132312113223321123", "23132312213123221331132231", "23132312321231332112213312",
    "23213123132312231132112332", "23213123213123312213223113", "23213123321231123321331221",
    "23321231132312322311233211", "23321231213123133122311322", "23321231321231211233122133",
]
# York prints column 11 of runs 39 and 40 as 1 and 2, which breaks the orthogonality;
# the GF(4) structure of the array (every column a linear combination of columns 1, 2
# and 6) gives 2 and 1, as used here.
_L64B = [
    "111111111111111111111", "111112222222222222222", "111113333333333333333", "111114444444444444444",
    "122221111222233334444", "122222222111144443333", "122223333444411112222", "122224444333322221111",
    "133331111333344442222", "133332222444433331111", "133333333111122224444", "133334444222211113333",
    "144441111444422223333", "144442222333311114444", "144443333222244441111", "144444444111133332222",
    "212341234123412341234", "212342143214321432143", "212343412341234123412", "212344321432143214321",
    "221431234214334124321", "221432143123443213412", "221433412432112342143", "221434321341221431234",
    "234121234341243212143", "234122143432134121234", "234123412123421434321", "234124321214312343412",
    "243211234432121433412", "243212143341212344321", "243213412214343211234", "243214321123434122143",
    "313421342134213421342", "313422431243124312431", "313423124312431243124", "313424213421342134213",
    "324311342243131244213", "324312431134242133124", "324313124421313422431", "324314213312424311342",
    "331241342312442132431", "331242431421331241342", "331243124134224314213", "331244213243113423124",
    "342131342421324313124", "342132431312413424213", "342133124243142131342", "342134213134231242431",
    "414231423142314231423", "414232314231423142314", "414233241324132413241", "414234132413241324132",
    "423141423231432414132", "423142314142341323241", "423143241413214232314", "423144132324123141423",
    "432411423324141322314", "432412314413232411423", "432413241142323144132", "432414132231414233241",
    "441321423413223143241", "441322314324114234132", "441323241231441321423", "441324132142332412314",
]
_L81 = [
    "1111111111111111111111111111111111111111", "1111111111111222222222222222222222222222",
    "1111111111111333333333333333333333333333", "1111222222222111111111222222222333333333",
    "1111222222222222222222333333333111111111", "1111222222222333333333111111111222222222",
    "1111333333333111111111333333333222222222", "1111333333333222222222111111111333333333",
    "1111333333333333333333222222222111111111", "1222111222333111222333111222333111222333",
    "1222111222333222333111222333111222333111", "1222111222333333111222333111222333111222",
    "1222222333111111222333222333111333111222", "1222222333111222333111333111222111222333",
    "1222222333111333111222111222333222333111", "1222333111222111222333333111222222333111",
    "1222333111222222333111111222333333111222", "1222333111222333111222222333111111222333",
    "1333111333222111333222111333222111333222", "1333111333222222111333222111333222111333",
    "1333111333222333222111333222111333222111", "1333222111333111333222222111333333222111",
    "1333222111333222111333333222111111333222", "1333222111333333222111111333222222111333",
    "1333333222111111333222333222111222111333", "1333333222111222111333111333222333222111",
    "1333333222111333222111222111333111333222", "2123123123123123123123123123123123123123",
    "2123123123123231231231231231231231231231", "2123123123123312312312312312312312312312",
    "2123231231231123123123231231231312312312", "2123231231231231231231312312312123123123",
    "2123231231231312312312123123123231231231", "2123312312312123123123312312312231231231",
    "2123312312312231231231123123123312312312", "2123312312312312312312231231231123123123",
    "2231123231312123231312123231312123231312", "2231123231312231312123231312123231312123",
    "2231123231312312123231312123231312123231", "2231231312123123231312231312123312123231",
    "2231231312123231312123312123231123231312", "2231231312123312123231123231312231312123",
    "2231312123231123231312312123231231312123", "2231312123231231312123123231312312123231",
    "2231312123231312123231231312123123231312", "2312123312231123312231123312231123312231",
    "2312123312231231123312231123312231123312", "2312123312231312231123312231123312231123",
    "2312231123312123312231231123312312231123", "2312231123312231123312312231123123312231",
    "2312231123312312231123123312231231123312", "2312312231123123312231312231123231123312",
    "2312312231123231123312123312231312231123", "2312312231123312231123231123312123312231",
    "3132132132132132132132132132132132132132", "3132132132132213213213213213213213213213",
    "3132132132132321321321321321321321321321", "3132213213213132132132213213213321321321",
    "3132213213213213213213321321321132132132", "3132213213213321321321132132132213213213",
    "3132321321321132132132321321321213213213", "3132321321321213213213132132132321321321",
    "3132321321321321321321213213213132132132", "3213132213321132213321132213321132213321",
    "3213132213321213321132213321132213321132", "3213132213321321132213321132213321132213",
    "3213213321132132213321213321132321132213", "3213213321132213321132321132213132213321",
    "3213213321132321132213132213321213321132", "3213321132213132213321321132213213321132",
    "3213321132213213321132132213321321132213", "3213321132213321132213213321132132213321",
    "3321132321213132321213132321213132321213", "3321132321213213132321213132321213132321",
    "3321132321213321213132321213132321213132", "3321213132321132321213213132321321213132",
    "3321213132321213132321321213132132321213", "3321213132321321213132132321213213132321",
    "3321321213132132321213321213132213132321", "3321321213132213132321132321213321213132",
    "3321321213132321213132213132321132321213",
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
    "L16b": lambda: _parse(_L16B),
    "L18": lambda: _parse(_L18),
    "L25": lambda: _parse(_L25),
    "L27": _l27,
    "L32": lambda: _two_level_array(32),
    "L32b": lambda: _parse(_L32B),
    "L36": lambda: _parse(_L36),
    "L50": lambda: _parse(_L50),
    "L54": lambda: _parse(_L54),
    "L64": lambda: _two_level_array(64),
    "L64b": lambda: _parse(_L64B),
    "L81": lambda: _parse(_L81),
}


def taguchi(name, factors, columns=None):
    """
    Generate a Taguchi orthogonal array design.

    Every pair of columns contains every combination of levels equally
    often (an orthogonal array of strength 2), so all main effects are
    orthogonal.  The arrays follow the row and column order of Taguchi's
    standard tables, so factors can be assigned to columns with his
    linear graphs and interaction tables:

    * ``'L4'``, ``'L8'``, ``'L16'``, ``'L32'``, ``'L64'`` -- 3, 7, 15,
      31 and 63 two-level columns.  The interaction of columns *a* and
      *b* falls in column ``a XOR b`` (e.g. 1 x 2 in column 3 of the L8).
    * ``'L9'`` -- 4 three-level columns (the 3^(4-2) with C = A + B and
      D = 2A + B).
    * ``'L12'`` -- 11 two-level columns.  Interactions are spread over
      all columns, so it is only suitable for main effects.
    * ``'L16b'`` -- 5 four-level columns (L16(4^5)).
    * ``'L18'`` -- column 1 has two levels, columns 2 to 8 three levels.
      The interaction between columns 1 and 2 is orthogonal to all other
      columns and can be estimated without giving up a column.
    * ``'L25'`` -- 6 five-level columns (L25(5^6)).
    * ``'L27'`` -- 13 three-level columns (the 3^(13-10) with base
      columns 1, 2 and 5).  The interaction of two columns occupies two
      other columns (e.g. 1 x 2 in columns 3 and 4).
    * ``'L32b'`` -- column 1 has two levels, columns 2 to 10 four levels
      (L32(2^1 4^9)).
    * ``'L36'`` -- columns 1 to 11 have two levels, columns 12 to 23
      three levels (L36(2^11 3^12)).
    * ``'L50'`` -- column 1 has two levels, columns 2 to 12 five levels
      (L50(2^1 5^11)).
    * ``'L54'`` -- column 1 has two levels, columns 2 to 26 three levels
      (L54(2^1 3^25)).
    * ``'L64b'`` -- 21 four-level columns (L64(4^21)).
    * ``'L81'`` -- 40 three-level columns (L81(3^40)).

    The arrays with a ``b`` are the second array of that run size in
    Taguchi's tables.

    Parameters
    ----------
    name : str
        The array: ``'L4'``, ``'L8'``, ``'L9'``, ``'L12'``, ``'L16'``,
        ``'L16b'``, ``'L18'``, ``'L25'``, ``'L27'``, ``'L32'``,
        ``'L32b'``, ``'L36'``, ``'L50'``, ``'L54'``, ``'L64'``,
        ``'L64b'`` or ``'L81'``.
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
    for j, (f, c) in enumerate(zip(factors, columns, strict=True)):
        col = array[:, c - 1]
        n_levels = int(col.max()) + 1
        lv = _levels(f, n_levels)
        assert len(lv) == n_levels, (
            f"Factor {f.name} has {len(lv)} levels but column {c} of the {name} has {n_levels}; "
            f"assign it to another column with columns=[...]"
        )
        design[:, j] = lv[col]

    return _to_dataframe(design, factors)


def crossed_array(inner, outer):
    """
    Cross an inner (control) design with an outer (noise) design.

    Taguchi's robust parameter design runs every run of the inner array,
    with the control factors, at every run of the outer array, with the
    noise factors.  The variation of the response over the outer runs then
    measures how robust each control setting is to the noise.  Any two
    designs can be crossed, not only Taguchi arrays.

    The result is in long format: ``len(inner) * len(outer)`` runs with
    the inner columns followed by the outer columns, ordered by inner run
    with the outer runs nested within each.  Taguchi's wide format, with
    one response column per outer run, follows from the long format with
    a pandas pivot once the response is measured, e.g.::

        Y["outer_run"] = np.tile(np.arange(len(outer)), len(inner))
        Y["inner_run"] = np.repeat(np.arange(len(inner)), len(outer))
        wide = Y.pivot(index="inner_run", columns="outer_run", values="response")

    A combined array, with the control and noise factors in a single
    design (e.g. an optimal design created with pyoptex), is the modern
    alternative and usually needs far fewer runs (Wu & Hamada, 2009,
    "Experiments: Planning, Analysis, and Optimization", 2nd ed., Wiley).

    Parameters
    ----------
    inner : pd.DataFrame
        The inner (control) design.
    outer : pd.DataFrame
        The outer (noise) design.

    Returns
    -------
    Y : pd.DataFrame
        The crossed design in long format.
    """
    assert isinstance(inner, pd.DataFrame) and isinstance(outer, pd.DataFrame), "inner and outer must be DataFrames"
    overlap = sorted(set(inner.columns) & set(outer.columns))
    assert not overlap, f"inner and outer must have different columns (both have {overlap})"
    for name, Y in (("inner", inner), ("outer", outer)):
        assert BLOCK_COLUMN not in Y.columns, f"The {name} design has blocks, which crossed arrays do not support"

    rows_inner = np.repeat(np.arange(len(inner)), len(outer))
    rows_outer = np.tile(np.arange(len(outer)), len(inner))
    return pd.concat(
        [inner.iloc[rows_inner].reset_index(drop=True), outer.iloc[rows_outer].reset_index(drop=True)], axis=1
    )
