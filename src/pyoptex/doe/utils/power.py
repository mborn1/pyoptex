"""
Module with the power of the tests of the effects, shared by the fixed structure and the cost optimal designs.
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ...utils.model import _model2termcols, encode_model


def _power(
    X: np.ndarray,
    Vinv: np.ndarray,
    Gs: Sequence[np.ndarray],
    terms: Sequence[np.ndarray],
    sizes: np.ndarray,
    alpha: float,
    ddf: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The power of the test of each term, for one set of variance ratios.

    Parameters
    ----------
    X : np.array(2d)
        The model matrix, with the covariate columns.
    Vinv : np.array(2d)
        The inverse of the observation covariance matrix V = I + sum of r_j Z_j Z_j'.
    Gs : list(np.array(2d))
        The derivatives of V to the variance components: I, Z_1 Z_1', Z_2 Z_2', ...
    terms : list(np.array(1d))
        The columns of X of each tested term.
    sizes : np.array(1d)
        The size of each term's coefficients, in units of the residual standard deviation.
    alpha : float
        The significance level.
    ddf : str
        The denominator degrees of freedom, "kenward-roger" or "satterthwaite".

    Returns
    -------
    power : np.array(1d)
        The power of each term. NaN when the design cannot estimate a variance component the test needs,
        or when power is not defined for the test.
    undefined : np.array(1d)
        Whether power is not defined for each term: its degrees of freedom are below 1,
        or the Kenward-Roger approximation does not hold in this small design.
    """
    VinvX = Vinv @ X
    Phi = np.linalg.inv(X.T @ VinvX)
    Ptilde = Vinv - VinvX @ Phi @ VinvX.T
    GVinvX = [G @ VinvX for G in Gs]
    VinvGVinvX = [Vinv @ g for g in GVinvX]
    P = np.array([VinvX.T @ g for g in GVinvX])
    Q = np.array([[gi.T @ vgj for vgj in VinvGVinvX] for gi in GVinvX])
    PhiPPhi = Phi @ P @ Phi

    # The information of the variance components, inverted on the directions the design can estimate
    PG = [Ptilde @ G for G in Gs]
    info = np.array([[np.sum(PGi * PGj.T) / 2 for PGj in PG] for PGi in PG])

    # The information of a component shrinks with its ratio, so the threshold needs a scale-free form:
    # each component is scaled by its information without fixed effects
    VinvG = [Vinv @ G for G in Gs]
    s = 1 / np.sqrt([np.sum(VG * VG.T) / 2 for VG in VinvG])
    eigval, eigvec = np.linalg.eigh(s[:, np.newaxis] * info * s[np.newaxis, :])
    eigvec = s[:, np.newaxis] * eigvec
    # After the scaling the eigenvalues have no units: when the whole information is rounding noise
    # (a saturated design), nothing is kept
    keep = eigval > 1e-10 * max(eigval.max(), 1)
    W = (eigvec[:, keep] / eigval[keep]) @ eigvec[:, keep].T
    unestimable = eigvec[:, ~keep]

    if ddf == "kenward-roger":
        Lambda = np.einsum("ij,ijab->ab", W, Q) - np.einsum("ij,iab,bc,jcd->ad", W, P, Phi, P)
        PhiA = Phi + 2 * Phi @ Lambda @ Phi

    power = np.full(len(terms), np.nan)
    undefined = np.zeros(len(terms), dtype=np.bool_)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t, (c, size) in enumerate(zip(terms, sizes, strict=True)):
            k = len(c)
            b = size * (-1.0) ** np.arange(k)
            LPhiL = Phi[np.ix_(c, c)]
            dPhi = PhiPPhi[:, c][:, :, c]

            # A test that leans on a variance component the design cannot estimate has no power
            lean = np.einsum("iu,ijk->ujk", unestimable, dPhi)
            if np.any(np.linalg.norm(lean, axis=(1, 2)) > 1e-8 * np.linalg.norm(LPhiL)):
                continue

            if ddf == "kenward-roger":
                M = np.linalg.solve(LPhiL, dPhi)
                traces = np.trace(M, axis1=1, axis2=2)
                A1 = traces @ W @ traces
                A2 = np.sum(W * np.einsum("iab,jba->ij", M, M))
                if k == 1:
                    df, lam = 2 / A2, 1.0
                else:
                    # As pbkrtest: without E, so that A2 = k does not divide by zero
                    B = (A1 + 6 * A2) / (2 * k)
                    g = ((k + 1) * A1 - (k + 4) * A2) / ((k + 2) * A2)
                    denom = 3 * k + 2 * (1 - g)
                    c1, c2, c3 = g / denom, (k - g) / denom, (k + 2 - g) / denom
                    num, den = 1 - A2 / k, 1 - c2 * B
                    r = 1.0 if abs(num) < 1e-14 and abs(den) < 1e-14 else num / den
                    rho = r**2 * (1 + c1 * B) / (k * (1 - c3 * B))
                    df = 4 + (k + 2) / (k * rho - 1)
                    lam = 1.0 if abs(df - 2) < 0.01 else df * (1 - A2 / k) / (df - 2)
                # Phi_A must be positive definite, whatever the sizes
                PhiAc = PhiA[np.ix_(c, c)]
                if not np.linalg.eigvalsh(PhiAc).min() > 0:
                    undefined[t] = True
                    continue
                noncentrality = lam * b @ np.linalg.solve(PhiAc, b)
            else:
                # Satterthwaite per direction of L'PhiL (Fai and Cornelius for k > 1, as lmerTest and pbkrtest)
                D, U = np.linalg.eigh(LPhiL)
                g = np.einsum("au,iab,bu->ui", U, dPhi, U)
                nus = 2 * D**2 / np.einsum("ui,ij,uj->u", g, W, g)
                if k == 1 or np.allclose(nus, nus[0]):
                    df = nus[0]
                elif np.any(nus <= 2):
                    df = 2.0
                else:
                    E = np.sum(nus / (nus - 2))
                    df = 2 * E / (E - k)
                lam = 1.0
                noncentrality = b @ np.linalg.solve(LPhiL, b)

            # A df below 1 is treated as "the design cannot test the term in practice": scipy is accurate there,
            # so this is a choice, not a numerical workaround. Kenward-Roger's F-scaling, noncentrality or Phi_A
            # can be invalid in very small designs (as in pbkrtest)
            if not (np.isfinite(df) and df >= 1 and lam > 0 and np.isfinite(noncentrality) and noncentrality >= 0):
                undefined[t] = True
                continue
            critical = stats.f.ppf(1 - alpha, k, df)
            if not np.isfinite(critical):
                undefined[t] = True
                continue
            # scipy's ncf.sf returns sf - 1 for a noncentrality of 0, so the central F is used there
            power[t] = (
                stats.f.sf(critical, k, df) if noncentrality == 0 else stats.ncf.sf(critical, k, df, noncentrality)
            )

    return power, undefined


def _term_columns(params: Any, model: pd.DataFrame | None, nmodel: int) -> list[tuple[str, np.ndarray]]:
    """
    The tests of a model: one per term of the model (except the intercept), with its encoded columns,
    or without a model one per column of Y2X. The covariate columns (from `nmodel` on) are not tested.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters, for the factors, their effect types and the Y2X function.
    model : pd.DataFrame or None
        The model, one row per term, or None to test each column of Y2X on its own.
    nmodel : int
        The number of columns of Y2X, where the covariate columns begin.

    Returns
    -------
    tests : list(tuple(str, np.array(1d)))
        The name and the columns of the model matrix of each test.
    """
    assert not any(fac.is_mixture for fac in params.factors), "Power for mixture designs is not supported yet"

    # Without a model, each column of Y2X on its own
    if model is None:
        return [(str(i), np.array([i])) for i in range(nmodel)]

    effect_types = params.effect_types
    model = model[[str(fac.name) for fac in params.factors]]
    termcols = _model2termcols(model, effect_types)
    modelenc = getattr(params.fn.Y2X, "keywords", {}).get("modelenc")
    if modelenc is not None:
        # A Y2X function from model2Y2X
        assert np.array_equal(modelenc, encode_model(model.to_numpy(), effect_types)), (
            "The model must be the one given to model2Y2X, with its rows in the same order"
        )
    ncols = sum(len(cols) for _, cols in termcols)
    assert ncols == nmodel, (
        f"The model has {ncols} columns but Y2X gives {nmodel}; "
        "with a custom Y2X, its columns must follow the order of encode_model"
    )

    # Every term except the intercept
    return [termcol for termcol, term in zip(termcols, model.to_numpy(), strict=True) if np.any(term != 0)]


def _power_table(
    tests: Sequence[tuple[str, np.ndarray]],
    X: np.ndarray,
    Zs: Sequence[np.ndarray | None],
    Vinv: np.ndarray,
    sizes: float | Sequence[float] | Mapping[str, float] | pd.Series,
    alpha: float,
    ddf: str,
) -> pd.DataFrame:
    """
    The power of each test, for each set of variance ratios.

    Parameters
    ----------
    tests : list(tuple(str, np.array(1d)))
        The name and the columns of the model matrix of each test.
    X : np.array(2d)
        The model matrix, with the covariate columns.
    Zs : list(np.array(1d) or None)
        The groupings of the random effects, None for a factor without one.
    Vinv : np.array(3d)
        The inverse of the observation covariance matrix, for each set of variance ratios.
    sizes : float, list(float), dict(str, float) or pd.Series
        The size of the coefficients of each test, or of all tests.
        A dict or pd.Series is matched by name, a list by position.
    alpha : float
        The significance level.
    ddf : str
        The denominator degrees of freedom, "kenward-roger" or "satterthwaite".

    Returns
    -------
    power : pd.DataFrame
        The power, one row per set of variance ratios and one column per test.
        NaN when the design cannot estimate a variance component the test needs.
        NaN also, with a warning, when power is not defined for the test: its degrees of freedom are below 1,
        or the Kenward-Roger approximation does not hold in this small design (ddf="satterthwaite" may give a value).
    """
    assert ddf in ("kenward-roger", "satterthwaite"), f"ddf must be 'kenward-roger' or 'satterthwaite', but is {ddf}"
    assert 0 < alpha < 1, f"alpha must be between 0 and 1, but is {alpha}"
    names = [name for name, _ in tests]
    columns = [cols for _, cols in tests]
    if isinstance(sizes, Mapping | pd.Series):
        # Iterating a Series gives its values, not its names
        sizes = dict(sizes)
        missing = [name for name in names if name not in sizes]
        assert not missing, f"sizes has no value for the tests {', '.join(missing)}"
        unknown = [str(key) for key in sizes if key not in names]
        assert not unknown, f"sizes has values for unknown tests {', '.join(unknown)}; the tests are {', '.join(names)}"
        size = np.array([sizes[name] for name in names], dtype=np.float64)
    elif np.ndim(sizes) == 0:
        size = np.full(len(tests), sizes, dtype=np.float64)
    else:
        size = np.asarray(sizes, np.float64)
        assert len(size) == len(tests), f"sizes must have one value per test ({len(tests)}), but has {len(size)}"

    # Every coefficient must be estimable
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1], (
        f"The model cannot be estimated from this design: its model matrix has rank {rank} for {X.shape[1]} columns"
    )

    # V is linear in the variance components: its derivatives are I and the ZZ' of each grouping
    n = len(X)
    Gs = [np.eye(n), *((Z[:, np.newaxis] == Z[np.newaxis, :]).astype(np.float64) for Z in Zs if Z is not None)]
    for Vi in Vinv:
        V = np.linalg.inv(Vi)
        coef = np.linalg.lstsq(np.array([G.ravel() for G in Gs]).T, V.ravel(), rcond=None)[0]
        assert np.allclose(np.tensordot(coef, Gs, axes=1), V), (
            "Power does not support the block effects of fixed-structure designs (block_effects) yet, "
            "nor covariate functions that change Vinv without adding their groupings to Zs"
        )

    # The power for each set of variance ratios, and the tests whose power is not defined in any
    powers = []
    undefined = np.zeros(len(tests), dtype=np.bool_)
    for Vi in Vinv:
        power, undefined_i = _power(X, Vi, Gs, columns, size, alpha, ddf)
        powers.append(power)
        undefined |= undefined_i
    if np.any(undefined):
        message = (
            f"Power is not defined for {', '.join(np.array(names)[undefined])}: their degrees of freedom are "
            "below 1, or the Kenward-Roger approximation does not hold in this small design."
        )
        if ddf == "kenward-roger":
            message += " ddf='satterthwaite' may give a value."
        warnings.warn(message, stacklevel=3)

    return pd.DataFrame(powers, columns=names)
