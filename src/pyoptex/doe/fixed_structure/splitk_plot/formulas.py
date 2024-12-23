"""
Module containing the update formulas for the Vinv updates of the split^k-plot algorithm
"""

import numba
import numpy as np


@numba.njit
def compute_update_UD(
        level, grp, Xi_old, X, 
        plot_sizes, c, thetas, thetas_inv
    ):
    """
    Compute the update to the information matrix after making
    a single coordinate adjustment. This update is expressed
    in the form: :math:`M^* = M + U^T D U`. D is a diagonal
    matrix in this case.

    Parameters
    ----------
    level: int
        The stratum at which the update occurs (0 for the lowest).
    grp : int
        The group within this stratum for which the update occurs.
    Xi_old : np.array(2d)
        The old runs after the update.
    X : np.array(2d)
        The new design matrix X (after the update).
    plot_sizes : np.array(1d)
        The size of each stratum b_i.
    c : np.array(2d)
        The coefficients c (every row specifies one set of a priori variance ratios). 
        The second dimension is added for Bayesian approaches.
    thetas : np.array(1d)
        The array of thetas.
        thetas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    thetas_inv : np.array(1d)
        The array of 1/thetas.
        thetas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/thetas[1:])))

    Returns
    -------
    U : np.array(2d)
        The U-matrix of the update
    D : np.array(2d)
        The set of diagonal matrices corresponding to the c parameter. To first row of
        D specifies a diagonal matrix corresponding to the first row of c.
    """
    # First runs
    jmp = thetas[level]
    runs = slice(grp*jmp, (grp+1)*jmp)
    
    # Extract new X section
    Xi_star = X[runs]

    # Initialize U and D
    star_offset = int(Xi_old.shape[0] * (1 + thetas_inv[level])) + (plot_sizes.size - level - 1)
    U = np.zeros((2*star_offset, Xi_old.shape[1]))
    D = np.zeros((len(c), 2*star_offset))

    # Store level-0 results
    U[:Xi_old.shape[0]] = Xi_old
    U[star_offset: star_offset + Xi_old.shape[0]] = Xi_star
    D[:, :Xi_old.shape[0]] = -np.ones(Xi_old.shape[0])
    D[:, star_offset: star_offset + Xi_old.shape[0]] = np.ones(Xi_old.shape[0])
    co = Xi_old.shape[0]

    # Loop before (= summations)
    if level != 0:
        # Reshape for summation
        Xi_old = Xi_old.reshape((-1, plot_sizes[0], Xi_old.shape[1]))
        Xi_star = Xi_star.reshape((-1, plot_sizes[0], Xi_star.shape[1]))
        for i in range(1, level):
            # Sum all smaller sections
            Xi_sum = np.sum(Xi_old, axis=1)
            Xi_star_sum = np.sum(Xi_star, axis=1)
            
            # Store entire matrix
            coe = co + Xi_sum.shape[0]
            U[co:coe] = Xi_sum
            U[star_offset+co: star_offset+coe] = Xi_star_sum
            D[:, co:coe] = -c[:, i-1]
            D[:, star_offset+co: star_offset+coe] = c[:, i-1]
            co = coe

            # Reshape for next iteration
            Xi_old = Xi_sum.reshape((-1, plot_sizes[i], Xi_sum.shape[1]))
            Xi_star = Xi_star_sum.reshape((-1, plot_sizes[i], Xi_star_sum.shape[1]))

        # Sum level-section
        Xi_old = np.sum(Xi_old, axis=1)
        Xi_star = np.sum(Xi_star, axis=1)

        # Store results
        U[co] = Xi_old
        U[star_offset+co] = Xi_star
        D[:, co] = -c[:, level-1]
        D[:, star_offset+co] = c[:, level-1]
        co += 1

    # Flatten the arrays for the next step
    Xi_old = Xi_old.flatten()
    Xi_star = Xi_star.flatten()

    # Loop after (= updates)
    for j in range(level, plot_sizes.size - 1):
        # Adjust group one level higher
        jmp *= plot_sizes[j]
        grp = grp // plot_sizes[j]

        # Compute section sum
        r_star = np.sum(X[grp*jmp: (grp+1)*jmp], axis=0)
        r = r_star - Xi_star + Xi_old

        # Store the results
        U[co] = r
        U[star_offset+co] = r_star
        D[:, co] = -c[:, j]
        D[:, star_offset+co] = c[:, j]
        co += 1

        # Set variables for next iteration
        Xi_old = r
        Xi_star = r_star

    # Return values
    return U, D

@numba.njit
def det_update_UD(U, D, Minv):
    """
    Compute the determinant adjustment as a factor.
    In other words: :math:`|M^*|=\\alpha*|M|`. The new
    information matrix originates from the following update
    formula: :math:`M^* = M + U^T D U`.

    The actual update is described as

    .. math::

        \\alpha = |D| |P| = |D| |D^{-1} + U M^{-1} U.T|

    Parameters
    ----------
    U : np.array(2d)
        The U matrix in the update.
    D : np.array(2d)
        The diagonal D matrix in the update. It is
        inserted as a 1d array representing the diagonal
        for each set of a-priori variance ratios.
    Minv: np.array(3d)
        The current inverses of the information matrices
        for each set of a-priori variance ratios.

    Returns
    -------
    alpha : float
        The update factor.
    P : np.array(3d)
        The P matrix of the update.
    """
    # Create updates
    P = np.zeros((len(D), D.shape[1], D.shape[1]))
    updates = np.zeros(len(D), dtype=np.float64)

    for j in range(len(D)):
        # Compute P
        P[j] = U @ Minv[j] @ U.T
        for i in range(P.shape[1]):
            P[j, i, i] += 1/D[j, i]

        # Compute update
        updates[j] = np.linalg.det(P[j]) * np.prod(D[j])

    # Compute determinant update
    return updates, P

@numba.njit
def inv_update_UD(U, D, Minv, P):
    """
    Compute the update of the inverse of the information matrix.
    In other words: :math:`M^{-1}^* = M^{-1} - M_{up}`. The new
    information matrix originates from the following update
    formula: :math:`M^* = M + U^T D U`.

    The actual update is described as

    .. math::

        M_{up} = M^{-1} U^T P^{-1} U M^{-1}

    .. math::
        P = D^{-1} + U M^{-1} U.T

    Parameters
    ----------
    U : np.array(2d)
        The U matrix in the update
    D : np.array(2d)
        The diagonal D matrix in the update. It is
        inserted as a 1d array representing the diagonal
        for each set of a-priori variance ratios.
    Minv: np.array(3d)
        The current inverses of the information matrices
        for each set of a-priori variance ratios.
    P : np.array(3d)
        The P matrix if already pre-computed.

    Returns
    -------
    Mup : np.array(3d)
        The updates to the inverses of the information
        matrices.
    """
    Mup = np.zeros_like(Minv)
    for i in range(len(Minv)):
        MU = Minv[i] @ U.T
        Mup[i] = (MU) @ np.linalg.solve(P[i], MU.T)
    return Mup

@numba.njit
def inv_update_UD_no_P(U, D, Minv):
    """
    See :py:func:`inv_update_UD <pyoptex.doe.splitk_plot.formulas.inv_update_UD>`,
    but without precomputing the P-matrix.
    """
    # Compute P
    P = np.zeros((len(D), D.shape[1], D.shape[1]))
    for j in range(len(D)):
        P[j] = U @ Minv[j] @ U.T
        for i in range(P.shape[1]):
            P[j, i, i] += 1/D[j, i]
    
    return inv_update_UD(U, D, Minv, P)
