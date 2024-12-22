"""
Module containing the update formulas for the Vinv updates of the CODEX algorithm
"""

import numba
import numpy as np

from ...._profile import profile

# Variable in a to indicate no update is necessary
NO_UPDATE = -1

def ce_update_vinv(Vinv, Zi, b, ratios):
    """
    Computes the update to Vinv based on the iterative application
    of the required operations in `b`.

    Parameters
    ----------
    Vinv : np.array(3d)
        The current inverses of the observation covariance matrix.
    Zi : np.array(1d)
        The current grouping matrix
    b : list(tuple(row_start, row_stop, group_from, group_to))
        A list of operations to apply changing the groups of certain rows.
    ratios : np.array(1d)
        The variance ratios of this column.

    Returns
    -------
    Zi : np.array(1d)
        The new grouping matrix (non-ascending)
    Vinv : np.array(3d)
        The updated inverses observation covariance matrix.
    """
    # Loop over all updates
    for x in b:
        # Expand update
        row_start, row_end, group_from, group_to = x

        # Update Vinv and Zi
        Vinv = group_update_vinv(Vinv, Zi, x, ratios)
        Zi[row_start:row_end] = group_to
    
    return Zi, Vinv

def insert_update_vinv(Vinv, Zs, pos, a, b, ratios):
    """
    Computes the update to Vinv based on the insertion of a row
    `a` into the design. `b` specifies possible additional group
    updates if necessary (e.g. when breaking up a group).

    Parameters
    ----------
    Vinv : np.array(3d)
        The current inverses of the observation covariance matrix.
    Zs : list(np.array(1d) or None)
        The grouping matrices of all factors.
    pos : int
        The position where to insert the new run.
    a : np.array(1d)
        The group of the newly inserted run.
    b : list(tuple(row_start, row_stop, group_from, group_to))
        A list of operations to apply changing the groups of certain rows.
    ratios : np.array(2d)
        The variance ratios of the factors in each row.

    Returns
    -------
    Zi : list(np.array(1d) or None)
        The new grouping matrices (non-ascending)
    Vinv : np.array(3d)
        The updated inverses observation covariance matrix.
    """
    # Insert run 'a' in grouping matrices
    Vinvn = add_update_vinv(Vinv, Zs, a, pos, ratios)
    Zsn = tuple([
        np.insert(Zi, pos, ai) if Zi is not None else None 
        for Zi, ai in zip(Zs, a)
    ])

    # Compute updates to Zi
    for i in range(len(Zsn)):
        if len(b[i]) > 0:
            # Expand update
            row_start, row_end, group_from, group_to = b[i]

            # Apply update
            Vinvn = group_update_vinv(Vinvn, Zsn[i], b[i], ratios[:, i])
            Zsn[i][row_start:row_end] = group_to

    return Zsn, Vinvn

def remove_update_vinv(Vinv, Zs, pos, b, ratios):
    """
    Computes the update to Vinv based on the removal of a row.
    `b` specifies possible additional group
    updates if necessary (e.g. when merging two groups).

    Parameters
    ----------
    Vinv : np.array(3d)
        The current inverses of the observation covariance matrix.
    Zs : list(np.array(1d) or None)
        The grouping matrices of all factors.
    pos : int
        The position where to insert the new run.
    b : list(tuple(row_start, row_stop, group_from, group_to))
        A list of operations to apply changing the groups of certain rows.
    ratios : np.array(2d)
        The variance ratios of the factors in each row.

    Returns
    -------
    Zi : list(np.array(1d) or None)
        The new grouping matrices (non-ascending)
    Vinv : np.array(3d)
        The updated inverses observation covariance matrix.
    """
    # Remove run from groupings
    Vinvn = del_vinv_update(Vinv, pos, ratios)
    Zsn = tuple([
        np.delete(Zi, pos) if Zi is not None else None 
        for Zi in Zs
    ])

    # Compute updates to Zi
    for i in range(len(Zsn)):
        if len(b[i]) > 0:
            # Expand update
            row_start, row_end, group_from, group_to = b[i]

            # Apply update
            Vinvn = group_update_vinv(Vinvn, Zsn[i], b[i], ratios[:, i])
            Zsn[i][row_start:row_end] = group_to  

    return Zsn, Vinvn

###################################

def detect_block_end_from_start(groups, start):
    """
    Detects the end of the block given the start of the block.
    A block is defined as a consecutive series of the same number.

    .. note::
        The block is assumed to be sorted.

    Parameters
    ----------
    groups : np.array(1d)
        The array to detect blocks from
    start : int
        The start run of the block

    Returns
    -------
    end : int
        The last index (excluded) of the block.
    """
    end = start + (groups.size - start) \
            - np.searchsorted((groups[start:] == groups[start])[::-1], True)
    return end

###################################

@profile
def group_update_vinv(Vinv, Zi, b, ratios):
    """
    Part of update formulas, see article for information.
    """
    # Expand change
    row_start, row_end, group_from, group_to = b
    
    # Add another dimension to ratios for broadcasting
    ratios = ratios[:, np.newaxis]

    # Initialize U and V
    U = np.zeros((Vinv.shape[0], len(Zi), 2))
    V = np.zeros((2, len(Zi)))

    # Create V
    V[0, Zi==group_from] = -1
    V[0, Zi==group_to] = 1
    V[0, row_start:row_end] = 2*(1 + V[0, row_start:row_end])
    V[1, row_start:row_end] = 1

    # Create U
    U[:, row_start:row_end, 0] = ratios
    U[:, :row_start, 1] = ratios * V[0, :row_start]
    U[:, row_end:, 1] = ratios * V[0, row_end:]

    # Perform the update
    VU = Vinv @ U
    P = V @ VU
    P[:, [0, 1], [0, 1]] += 1
    PpDinv = np.linalg.inv(P)
    Vinv -= VU @ (PpDinv @ (V @ Vinv))

    return Vinv

@profile
def add_update_vinv(Vinv, Zs, a, pos, ratios):
    """
    Part of update formulas, see article for information.
    """
    # Compute inner products Zi and ai
    Zs_valid = np.array([i for i in range(len(Zs)) if Zs[i] is not None])
    B = np.zeros((Vinv.shape[0], Vinv.shape[-1], len(Zs_valid)))
    for j, i in enumerate(Zs_valid):
        B[:, Zs[i] == a[i], j] = ratios[:, np.newaxis, i]
    B = np.sum(B, axis=-1)

    # Initialize matrix
    Vinvn = np.zeros((Vinv.shape[0], Vinv.shape[1] + 1, Vinv.shape[2] + 1))
    
    # Compute matrix parts
    VinvB = np.squeeze(Vinv @ B[:, :, np.newaxis], axis=-1)
    Pinv = 1 / (1 + np.sum(ratios[:, Zs_valid], axis=1) - np.sum(B * VinvB, axis=1))
    Bn = -Pinv[:, np.newaxis] * VinvB
    An = Vinv - Bn[:, :, np.newaxis] @ VinvB[:, np.newaxis, :]

    # Store results
    Vinvn[:, pos, pos] = Pinv
    Vinvn[:, :pos, pos] = Bn[:, :pos]
    Vinvn[:, pos, :pos] = Bn[:, :pos]
    Vinvn[:, :pos, :pos] = An[:, :pos, :pos]

    # If pos is not the last row
    if pos < Vinv.shape[1]:
        Vinvn[:, pos+1:, pos] = Bn[:, pos:]
        Vinvn[:, pos, pos+1:] = Bn[:, pos:]
        Vinvn[:, :pos, pos+1:] = An[:, :pos, pos:]
        Vinvn[:, pos+1:, :pos] = An[:, pos:, :pos]
        Vinvn[:, pos+1:, pos+1:] = An[:, pos:, pos:]

    return Vinvn

@profile
def del_vinv_update(Vinv, pos, ratios):
    """
    Part of update formulas, see article for information.
    """
    # Initialize
    keep = np.ones(Vinv.shape[1], dtype=np.bool_)
    keep[pos] = False

    # The baseline
    Vinvn = Vinv[:, keep][:, :, keep]

    # Update
    b = Vinv[:, pos, keep]
    dinv = 1/Vinv[:, pos, pos]
    Vinvn -= ((b.T * dinv).T)[:, :, np.newaxis] @ b[:, np.newaxis, :]
    
    return Vinvn

###################################

# Expanded multiplications with R and S
# Slower than regular matrix multiplications

@numba.njit
def inv_PpD_numba(P, ratios=1):
    """
    Part of update formulas, see article for information.
    """
    P[:, 0, 0] += 1/ratios
    P[:, 1, 1] += 1/ratios
    out = np.empty(P.shape)
    for i in range(len(P)):
        out[i] = np.linalg.inv(P[i])
    return out


@numba.njit
def _group_update_vinv(Vinv, Zi, b, ratios):
    """
    Part of update formulas, see article for information.
    """
    # Expand change
    row_start, row_end, group_from, group_to = b

    # Define T
    T_from = (Zi==group_from)
    T_to = (Zi==group_to)

    # Compute update submatrices
    VR = np.empty((Vinv.shape[0], Vinv.shape[1], 2))
    VR[:, :, 0] = np.sum(Vinv[:, :, row_start:row_end], axis=2)
    VR[:, :, 1] = np.sum(Vinv[:, :, :row_start][:, :, T_to[:row_start]], axis=2)\
                    + np.sum(Vinv[:, :, row_end:][:, :, T_to[row_end:]], axis=2)\
                    - np.sum(Vinv[:, :, :row_start][:, :, T_from[:row_start]], axis=2)\
                    - np.sum(Vinv[:, :, row_end:][:, :, T_from[row_end:]], axis=2)
    SV = np.empty((VR.shape[0], VR.shape[2], VR.shape[1]))
    SV[:, 0, :] = VR[:, :, 1] + (
        2*np.sum(Vinv[:, :, row_start:row_end][:, :, ~T_from[row_start:row_end]], axis=2)
        + 2*np.sum(Vinv[:, :, row_start:row_end][:, :, T_to[row_start:row_end]], axis=2)
    )
    SV[:, 1, :] = VR[:, :, 0]
    P = np.empty((Vinv.shape[0], 2, 2))
    P[:, :, 0] = np.sum(SV[:, :, row_start:row_end], axis=2)
    P[:, :, 1] = np.sum(SV[:, :, :row_start][:, :, T_to[:row_start]], axis=2)\
                    + np.sum(SV[:, :, row_end:][:, :, T_to[row_end:]], axis=2)\
                    - np.sum(SV[:, :, :row_start][:, :, T_from[:row_start]], axis=2)\
                    - np.sum(SV[:, :, row_end:][:, :, T_from[row_end:]], axis=2)

    # Perform the update
    PpDinv = inv_PpD_numba(P, ratios)
    for i in range(len(Vinv)):
        Vinv[i] -= VR[i] @ (PpDinv[i] @ SV[i])

    return Vinv
