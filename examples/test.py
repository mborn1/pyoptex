import numpy as np

plot_sizes = np.array([4, 6])
ratios = np.array([1, 10])

# Alphas and betas
alphas = np.cumprod(plot_sizes[::-1])[::-1]
betas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))

# Betas inverse
betas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/betas[1:])))

# Compute c-coefficients
c = np.zeros(plot_sizes.size)
c[0] = 1
for i in range(1, c.size):
    print(np.sum(betas[:i] * c[:i]))
    print(np.sum(ratios[:i+1] * betas[:i+1]))
    c[i] = -ratios[i] * np.sum(betas[:i] * c[:i]) / np.sum(ratios[:i+1] * betas[:i+1])

X = np.array(
[[ 1.,  0.,  0., -1.,  0.,  1.,  0., -0.,  0.,  0., -0.,  0.,  0., -0., -1.,  0.,  0.,  1.],
 [ 1., -1., -1.,  1.,  0.,  1.,  1., -1., -0., -1., -1., -0., -1.,  0.,  1.,  1.,  1.,  1.],
 [ 1.,  0., -1.,  1.,  0.,  1., -0.,  0.,  0.,  0., -1., -0., -1.,  0.,  1.,  0.,  1.,  1.],
 [ 1.,  1., -1., -1.,  0.,  1., -1., -1.,  0.,  1.,  1., -0., -1., -0., -1.,  1.,  1.,  1.],
 [ 1., -1.,  0.,  1., -1., -1., -0., -1.,  1.,  1.,  0., -0., -0., -1., -1.,  1.,  0.,  1.],
 [ 1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
 [ 1.,  1.,  0.,  0., -1., -1.,  0.,  0., -1., -1.,  0., -0., -0., -0., -0.,  1.,  0.,  0.],
 [ 1.,  1.,  0.,  1., -1., -1.,  0.,  1., -1., -1.,  0., -0., -0., -1., -1.,  1.,  0.,  1.],
 [ 1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
 [ 1.,  0., -1.,  0.,  0.,  1., -0.,  0.,  0.,  0., -0., -0., -1.,  0.,  0.,  0.,  1.,  0.],
 [ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.],
 [ 1., -1.,  1.,  0.,  0.,  1., -1., -0., -0., -1.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.],
 [ 1., -1.,  1.,  0.,  1.,  0., -1., -0., -1., -0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.],
 [ 1.,  1., -1.,  1.,  1.,  0., -1.,  1.,  1.,  0., -1., -1., -0.,  1.,  0.,  1.,  1.,  1.],
 [ 1.,  1., -1., -1.,  1.,  0., -1., -1.,  1.,  0.,  1., -1., -0., -1., -0.,  1.,  1.,  1.],
 [ 1.,  0.,  1., -1.,  1.,  0.,  0., -0.,  0.,  0., -1.,  1.,  0., -1., -0.,  0.,  1.,  1.],
 [ 1., -1.,  1.,  0., -1., -1., -1., -0.,  1.,  1.,  0., -1., -1., -0., -0.,  1.,  1.,  0.],
 [ 1.,  1.,  0.,  1., -1., -1.,  0.,  1., -1., -1.,  0., -0., -0., -1., -1.,  1.,  0.,  1.],
 [ 1.,  0.,  1.,  0., -1., -1.,  0.,  0., -0., -0.,  0., -1., -1., -0., -0.,  0.,  1.,  0.],
 [ 1.,  0.,  0., -1., -1., -1.,  0., -0., -0., -0., -0., -0., -0.,  1.,  1.,  0.,  0.,  1.],
 [ 1., -1.,  1.,  0.,  1.,  0., -1., -0., -1., -0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.],
 [ 1., -1.,  1., -1.,  1.,  0., -1.,  1., -1., -0., -1.,  1.,  0., -1., -0.,  1.,  1.,  1.],
 [ 1.,  0., -1.,  1.,  1.,  0., -0.,  0.,  0.,  0., -1., -1., -0.,  1.,  0.,  0.,  1.,  1.],
 [ 1., -1.,  0., -1.,  1.,  0., -0.,  1., -1., -0., -0.,  0.,  0., -1., -0.,  1.,  0.,  1.],]
)

Xi_star = np.array(
[[ 1.,  0.,  0., -1.,  1.,  0.,  0., -0.,  0.,  0., -0.,  0.,  0., -1., -0.,  0.,  0.,  1.],
 [ 1., -1., -1.,  1.,  1.,  0.,  1., -1., -1., -0., -1., -1., -0.,  1.,  0.,  1.,  1.,  1.],
 [ 1.,  0., -1.,  1.,  1.,  0., -0.,  0.,  0.,  0., -1., -1., -0.,  1.,  0.,  0.,  1.,  1.],
 [ 1.,  1., -1., -1.,  1.,  0., -1., -1.,  1.,  0.,  1., -1., -0., -1., -0.,  1.,  1.,  1.],]
)
import numba

# @numba.njit
def compute_update(level, grp, X, Xi_star, plot_sizes, c, betas=None, betas_inv=None):
    """
    Compute the update to the information matrix after making
    a single coordinate adjustment. This update is expressed
    in the form: :math:`M^* = M + U^T D U`. D is a diagonal
    matrix in this case.

    Parameters
    ----------
    level : int
        The level at which to make the adjustment
    cols : (start, stop, step)
        The slice of columns to update (can be more than
        one if the column is categorical)
    grp : int
        The group to update (relative to the level)
    new_coords : np.array
        The new coordinates to insert. By selecting multiple
        columns at once, constraints can be added.
    state : (Y, X, c, plot_sizes, model)
        Other immutable variables required to compute the update.

    Returns
    -------
    U : np.array
        The U matrix of the update formula
    D : np.array
        The diagonal D matrix of the update formula.
        It is returned as a 1D array
    Xi_star : np.array
        The update to the model matrix (X)
    Yi_star : np.array
        The update to the design matrix (Y)
    runs : (start, stop, step)
        The updated runs. In other words, if the update
        is accepted:

        * X[runs] = Xi_star
        * Y[runs] = Yi_star
    """

    # First runs
    jmp = betas[level]
    runs = slice(grp*jmp, (grp+1)*jmp)
    
    # Extract level-section
    Xi = X[runs]

    # Initialize U and D
    star_offset = int(Xi.shape[0] * (1 + betas_inv[level])) + (plot_sizes.size - level - 1)
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

level = 1
grp = 0
jmp = betas[level]
U, D = compute_update(level, grp, X, Xi_star, plot_sizes.astype(np.int64), c, betas, betas_inv)

from pyoptex.doe.splitk_plot.utils import obs_var
V = obs_var(plot_sizes, ratios[1:].astype(np.float64))
M = X.T @ np.linalg.solve(V, X)

X2 = np.copy(X)
X2[grp*jmp:(grp+1)*jmp] = Xi_star
M2 = X2.T @ np.linalg.solve(V, X2)

diff = (M + U.T @ np.diag(D) @ U) - M2
print(np.linalg.norm(diff))
