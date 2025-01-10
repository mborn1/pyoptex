"""
Module for all init functions of fixed structure.
"""

import numba
import numpy as np
from numba.typed import List

from ..._profile import profile
from ...utils.design import encode_design
from ..utils.init import init_single_unconstrained


@numba.njit
def __init_unconstrained(effect_types, effect_levels, grps, 
                         coords, Zs, Y, complete=False):
    """
    This function generates a random design without 
    considering any design constraints.

    .. note::
        The resulting design matrix `Y` is not encoded.

    Parameters
    ----------
    effect_types : np.array(1d)
        The effect types of each factor, representing 
        a 1 for a continuous factor and the number of 
        levels for a categorical factor.
    effect_levels : np.array(1d)
        The level of each factor.
    grps : :py:class:`numba.typed.List` (np.array(1d))
        The groups for each factor to initialize.
    coords : :py:class:`numba.typed.List` (np.array(2d))
        The coordinates for each factor to use.
    Zs : np.array(2d)
        Every grouping vector Z stacked vertically.
    Y : np.array(2d)
        The design matrix to be initialized. May contain the
        some fixed settings if not optimizing all groups.
        This matrix should not be encoded.
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.

    Returns
    -------
    Y : np.array(2d)
        The initialized design matrix.
    """
    ##################################################
    # UNCONSTRAINED DESIGN
    ##################################################
    # Loop over all columns
    for col in range(effect_types.size):
        # Extract parameters
        level = effect_levels[col]
        typ = effect_types[col]

        # Generate random values
        lgrps = grps[col]
        n = len(lgrps)

        if complete:
            if typ == 1:
                # Continuous factor
                r = np.random.rand(n) * 2 - 1
            else:
                # Discrete factor
                choices = np.arange(typ, dtype=np.float64)
                if typ >= n:
                    r = np.random.choice(choices, n, replace=False)
                else:
                    n_replicates = n // choices.size
                    r = np.random.permutation(
                        np.concatenate((
                            np.repeat(choices, n_replicates), 
                            np.random.choice(choices, n - choices.size * n_replicates)
                        ))
                    )
        else:
            # Extract the possible coordinates
            if typ > 1:
                # Convert to decoded values for categorical factors
                choices = np.arange(len(coords[col]), dtype=np.float64)
            else:
                choices = coords[col].flatten()

            # Pick from the choices and try to have all of them atleast once
            if choices.size >= n:
                r = np.random.choice(choices, n, replace=False)
            else:
                n_replicates = n // choices.size
                r = np.random.permutation(np.concatenate((
                    np.repeat(choices, n_replicates), 
                    np.random.choice(choices, n - choices.size * n_replicates)
                )))
        
        # Fill design
        if level == 0:
            Y[lgrps, col] = r
        else:
            Z = Zs[level-1]
            for i, grp in enumerate(lgrps):
                Y[Z == grp, col] = r[i]

    return Y

@numba.njit
def __correct_constraints(effect_types, effect_levels, grps, coords, 
                          constraints, Zs, Y, complete=False):
    """
    Corrects a design matrix to be within the `constraints`.
    It alters the factors starting from the most hard to change factors 
    to the most easy to change until all constraints are met.

    .. note::
        The resulting design matrix `Y` is not encoded.

    Parameters
    ----------
    effect_types : np.array(1d)
        The effect types of each factor, representing 
        a 1 for a continuous factor and the number of 
        levels for a categorical factor.
    effect_levels : np.array(1d)
        The level of each factor.
    grps : :py:class:`numba.typed.List` (np.array(1d))
        The groups for each factor to initialize.
    coords : :py:class:`numba.typed.List` (np.array(2d))
        The coordinates for each factor to use.
    constraints : func
        The constraints function, validating the design matrix.
        Should return True if the constraints are violated.
    Zs : np.array(2d)
        Every grouping vector Z stacked vertically.
    Y : np.array(2d)
        The design matrix to be initialized. May contain the
        some fixed settings if not optimizing all groups.
        This matrix should not be encoded.
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.

    Returns
    -------
    Y : np.array(2d)
        The initialized design matrix.
    """
    # Check which runs are invalid
    invalid_run = np.ascontiguousarray(constraints(Y))

    # Order the Zs
    zidx = np.concatenate((
        np.argsort(np.array([len(np.unique(Z)) for Z in Zs])) + 1,
        np.array([0])
    ))

    # Loop over all levels from small number to large number
    for j, level in enumerate(zidx):
        # Initialize permitted to optimize
        if level == 0:
            permitted_to_optimize = np.zeros(len(Y), dtype=np.bool_)
        else:
            permitted_to_optimize = np.zeros(np.max(Zs[level-1])+1, dtype=np.bool_)

        # Check which ones are permitted for this level
        for grp, el in zip(grps, effect_levels):
            if el == level:
                permitted_to_optimize[grp] = True
        permitted_to_optimize = np.flatnonzero(permitted_to_optimize)

        # Loop over all groups
        for i in permitted_to_optimize:
            # Determine which runs belong to that group
            if level == 0:
                runs = np.array([i], dtype=np.int_)
                prev_runs = np.arange(i+1, dtype=np.int_)
            else:
                runs = np.flatnonzero(Zs[level-1] == i)
                prev_runs = np.flatnonzero(Zs[level-1] <= i)

            # Check if all invalid
            if np.all(invalid_run[runs]):
                # Specify which groups to regenerate
                grps_ = [
                    np.array([
                        co 
                        for co in (np.unique(Zs[l-1][runs]) if l > 0 else runs)
                        if co in grps[k]
                    ], dtype=np.int_)
                    if l in zidx[j:] else np.empty((0,), dtype=np.int_) 
                    for k, l in enumerate(effect_levels)
                ]
                grps_ = List(grps_)

                # Regenerate until no longer all invalid
                c = True
                while c:
                    Y = __init_unconstrained(effect_types, effect_levels, grps_, 
                                            coords, Zs, Y, complete)
                    c = np.any(constraints(Y[prev_runs]))

                # Update the runs
                invalid_run = constraints(Y)

    return Y

@profile
def initialize_feasible(params, complete=False, max_tries=1000):
    """
    Generates a random initial design for a generic design.
    `grps` specifies at each level which level-groups should be
    initialized. This is useful when augmenting an existing design.

    .. note::
        The resulting design matrix `Y` is not encoded.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The parameters of the design generation.
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.
    max_tries : int
        The maximum number of tries to generate a feasible design.

    Returns
    -------
    Y : np.array(2d)
        The generated design.
    enc : tuple(np.array(2d), np.array(2d))
        The categorical factor encoded Y and X respectively.
    """
    # Compute design sizes
    ncol = params.effect_types.shape[0]

    # Initiate design matrix
    Y = params.prior
    if Y is None:
        Y = np.zeros((params.nruns, ncol), dtype=np.float64)

    feasible = False
    tries = 0
    while not feasible:
        # Add one try
        tries += 1

        # Initialize unconstrained
        Y = __init_unconstrained(
            params.effect_types, params.effect_levels, params.grps, 
            params.coords, params.Zs, Y, complete
        )

        # Constraint corrections
        Y = __correct_constraints(
            params.effect_types, params.effect_levels, params.grps, 
            params.coords, params.fn.constraintso,
            params.Zs, Y, complete
        )
        
        # Encode the design
        Yenc = encode_design(Y, params.effect_types)

        # Make sure it's feasible
        Xenc = params.fn.Y2X(Yenc)
        feasible = np.linalg.matrix_rank(Xenc) >= Xenc.shape[1]

        # Check if not in infinite loop
        if tries >= max_tries and not feasible:

            # Determine which column causes rank deficiency
            for i in range(1, Xenc.shape[1]+1):
                if np.linalg.matrix_rank(Xenc[:, :i]) < i:
                    break

            # pylint: disable=line-too-long
            raise ValueError(f'Unable to find a feasible design due to the model: component {i} causes rank collinearity with all prior components (note that these are categorically encoded)')

                    
    return Y, (Yenc, Xenc)

def init_random(params, n=1, complete=False):
    """
    Initialize a design with `n` randomly sampled runs. They must
    be within the constraints.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.utils.Parameters>`
        The parameters of the design generation.
    n : int
        The number of runs
    complete : bool
        Whether to use the coordinates for initialization
        or initialize fully randomly.

    Returns
    -------
    design : np.array(2d)
        The resulting design.
    """
    # Initialize
    run = np.zeros((n, params.colstart[-1]), dtype=np.float64)
    invalid = np.ones(n, dtype=np.bool_)

    # Adjust for completeness
    if complete:
        coords = None
    else:
        coords = params.coords

    # Loop until all are valid
    while np.any(invalid):
        run[invalid] = init_single_unconstrained(params.colstart, coords, run[invalid], params.effect_types)
        invalid[invalid] = params.fn.constraints(run[invalid])

    return run
