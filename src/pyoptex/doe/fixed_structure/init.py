"""
Module for all init functions of fixed structure.
"""

import numpy as np
from ..._profile import profile
from ...utils.design import encode_design
from ..utils.init import init_single_unconstrained
from ._init_cy import __init_unconstrained, __correct_constraints

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
