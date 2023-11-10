import numba
import numpy as np

# TODO: transform update formulas to 3d

@numba.njit
def compute_update_UD(
        level, grp, Xi_star, X, 
        plot_sizes, c, thetas, thetas_inv
    ):
    """
    Compute the update to the information matrix after making
    a single coordinate adjustment. This update is expressed
    in the form: :math:`M^* = M + U^T D U`. D is a diagonal
    matrix in this case.

    Parameters
    ----------
    
    """
    # Append 1 again
    c = np.concatenate((np.array([1], dtype=np.float64), c))

    # First runs
    jmp = thetas[level]
    runs = slice(grp*jmp, (grp+1)*jmp)
    
    # Extract level-section
    Xi = X[runs]

    # Initialize U and D
    star_offset = int(Xi.shape[0] * (1 + thetas_inv[level])) + (plot_sizes.size - level - 1)
    U = np.zeros((2*star_offset, Xi.shape[1]))
    D = np.zeros(2*star_offset)

    # Store level-0 results
    U[:Xi.shape[0]] = Xi
    U[star_offset: star_offset + Xi.shape[0]] = Xi_star
    D[:Xi.shape[0]] = -np.ones(Xi.shape[0])
    D[star_offset: star_offset + Xi.shape[0]] = np.ones(Xi.shape[0])
    co = Xi.shape[0]

    # Loop before (= summations)
    if level != 0:
        # Reshape for summation
        Xi = Xi.reshape((-1, plot_sizes[0], Xi.shape[1]))
        Xi_star = Xi_star.reshape((-1, plot_sizes[0], Xi_star.shape[1]))
        for i in range(1, level):
            # Sum all smaller sections
            Xi_sum = np.sum(Xi, axis=1)
            Xi_star_sum = np.sum(Xi_star, axis=1)
            
            # Store entire matrix
            coe = co + Xi_sum.shape[0]
            U[co:coe] = Xi_sum
            U[star_offset+co: star_offset+coe] = Xi_star_sum
            D[co:coe] = -c[i]
            D[star_offset+co: star_offset+coe] = c[i]
            co = coe

            # Reshape for next iteration
            Xi = Xi_sum.reshape((-1, plot_sizes[i], Xi_sum.shape[1]))
            Xi_star = Xi_star_sum.reshape((-1, plot_sizes[i], Xi_star_sum.shape[1]))

        # Sum level-section
        Xi = np.sum(Xi, axis=1)
        Xi_star = np.sum(Xi_star, axis=1)

        # Store results
        U[co] = Xi
        U[star_offset+co] = Xi_star
        D[co] = -c[level]
        D[star_offset+co] = c[level]
        co += 1

    # Flatten the arrays for the next step
    Xi = Xi.flatten()
    Xi_star = Xi_star.flatten()

    # Loop after (= updates)
    for j in range(level, plot_sizes.size - 1):
        # Adjust group one level higher
        jmp *= plot_sizes[j]
        grp = grp // plot_sizes[j]

        # Compute section sum
        r = np.sum(X[grp*jmp: (grp+1)*jmp], axis=0)
        r_star = r - Xi + Xi_star

        # Store the results
        U[co] = r
        U[star_offset+co] = r_star
        D[co] = -c[j+1]
        D[star_offset+co] = c[j+1]
        co += 1

        # Set variables for next iteration
        Xi = r
        Xi_star = r_star

    # Return values
    return U, D

@numba.njit
def det_update_UD(U, D, Minv):
    """
    Compute the determinant adjustment as a factor.
    In other words: :math:`|M^*|=\alpha*|M|`. The new
    information matrix originates from the following update
    formula: :math:`M^* = M + U^T D U`.

    The actual update is described as

    .. math::

        \alpha = |D| |P| = |D| |D^{-1} + U M^{-1} U.T|

    Parameters
    ----------
    U : np.array
        The U matrix in the update
    D : np.array
        The diagonal D matrix in the update. It is
        inserted as a 1d array representing the diagonal.
    Minv: np.array
        The current inverse of the information matrix.

    Returns
    -------
    alpha : float
        The update factor
    P : np.array
        The P matrix of the update
    """
    # Compute P
    P = U @ Minv @ U.T
    for i in range(P.shape[0]):
        P[i, i] += 1/D[i]

    # Compute determinant update
    return np.linalg.det(P) * np.prod(D), P

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
    U : np.array
        The U matrix in the update
    D : np.array
        The diagonal D matrix in the update. It is
        inserted as a 1d array representing the diagonal.
    Minv: np.array
        The current inverse of the information matrix.
    P : np.array
        The P matrix if already pre-computed.

    Returns
    -------
    Mup : np.array
        The update to the inverse matrix.
    """
    return (Minv @ U.T) @ np.linalg.solve(P, U @ Minv)

@numba.njit
def inv_update_UD_no_P(U, D, Minv):
    P = U @ Minv @ U.T
    for i in range(P.shape[0]):
        P[i, i] += 1/D[i]
    return inv_update_UD(U, D, Minv, P)
