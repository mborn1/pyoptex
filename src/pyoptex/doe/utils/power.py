"""
Module with the power of the tests of the effects, shared by the fixed structure and the cost optimal designs.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ...utils.model import encode_model, model2names


def _power(
    X: np.ndarray,
    Vinv: np.ndarray,
    Gs: Sequence[np.ndarray],
    terms: Sequence[np.ndarray],
    sizes: np.ndarray,
    alpha: float,
    ddf: str,
) -> np.ndarray:
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
        The power of each term, NaN when the design cannot estimate a variance
        component that the test depends on, and with Kenward-Roger when its degrees
        of freedom are not defined (a warning is issued).
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
    undefined_df = False
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
                noncentrality = lam * b @ np.linalg.solve(PhiA[np.ix_(c, c)], b)
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
                noncentrality = b @ np.linalg.solve(LPhiL, b)

            # Kenward-Roger's df can be negative in very small designs (as in pbkrtest)
            if not np.isfinite(df) or df <= 0:
                undefined_df = True
                continue
            critical = stats.f.ppf(1 - alpha, k, df)
            # scipy's ncf.sf returns sf - 1 for a noncentrality of 0, so the central F is used there
            power[t] = (
                stats.f.sf(critical, k, df) if noncentrality == 0 else stats.ncf.sf(critical, k, df, noncentrality)
            )

    if undefined_df and ddf == "kenward-roger":
        warnings.warn(
            "Kenward-Roger's degrees of freedom are not defined for some terms of this small design; "
            "their power is NaN. ddf='satterthwaite' gives a value.",
            stacklevel=2,
        )

    return power


def _power_table(
    params: Any,
    model: pd.DataFrame,
    X: np.ndarray,
    Zs: Sequence[np.ndarray | None],
    Vinv: np.ndarray,
    sizes: float | Sequence[float],
    alpha: float,
    ddf: str,
) -> pd.DataFrame:
    """
    The power of the test of each term of the model (except the intercept), for each set of variance ratios.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>` or :py:class:`Parameters <pyoptex.doe.cost_optimal.utils.Parameters>`
        The simulation parameters, for the factors, their effect types and the Y2X function.
    model : pd.DataFrame
        The model, one row per term.
    X : np.array(2d)
        The model matrix, with the covariate columns after the model's columns.
    Zs : list(np.array(1d) or None)
        The groupings of the random effects, None for a factor without one.
    Vinv : np.array(3d)
        The inverse of the observation covariance matrix, for each set of variance ratios.
    sizes : float or list(float)
        The size of the coefficients of each term, or of all terms.
    alpha : float
        The significance level.
    ddf : str
        The denominator degrees of freedom, "kenward-roger" or "satterthwaite".

    Returns
    -------
    power : pd.DataFrame
        The power, one row per set of variance ratios and one column per tested term.
    """
    assert ddf in ("kenward-roger", "satterthwaite"), f"ddf must be 'kenward-roger' or 'satterthwaite', but is {ddf}"
    assert not any(fac.is_mixture for fac in params.factors), "Power for mixture designs is not supported yet"
    effect_types = params.effect_types

    # The columns of each term: the rows that encode_model makes from it, in order
    col_names = [str(fac.name) for fac in params.factors]
    model = model[col_names]
    nenc = [len(encode_model(row[np.newaxis], effect_types)) for row in model.to_numpy()]
    starts = np.concatenate(([0], np.cumsum(nenc)))
    modelenc = getattr(params.fn.Y2X, "keywords", {}).get("modelenc")
    if modelenc is not None:
        # A Y2X function from model2Y2X
        assert np.array_equal(modelenc, encode_model(model.to_numpy(), effect_types)), (
            "The model must be the one given to model2Y2X, with its rows in the same order"
        )
    tested = np.flatnonzero(np.any(model.to_numpy() != 0, axis=1))
    terms = [np.arange(starts[i], starts[i + 1]) for i in tested]
    all_names = model2names(model)
    names = [str(all_names[i]) for i in tested]

    size = np.full(len(tested), sizes, dtype=np.float64) if np.ndim(sizes) == 0 else np.asarray(sizes, np.float64)
    assert len(size) == len(tested), f"sizes must have one value per tested term ({len(tested)}), but has {len(size)}"

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

    return pd.DataFrame([_power(X, Vi, Gs, terms, size, alpha, ddf) for Vi in Vinv], columns=names)
