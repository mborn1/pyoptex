"""
Module for all init functions of the split^k-plot algorithm
"""

import numpy as np

from ._init_cy import __init_unconstrained, __correct_constraints
from ...._profile import profile
from ....utils.design import encode_design

@profile
def initialize_feasible(params, complete=False, max_tries=1000):
    """
    Generates a random initial design for a split^k plot design.
    `grps` specifies at each level which level-groups should be
    initialized. This is useful when augmenting an existing design.

    .. note::
        The resulting design matrix `Y` is not encoded.

    Parameters
    ----------
    params : :py:class:`Parameters <pyoptex.doe.fixed_structure.splitk_plot.utils.Parameters>`
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
    n = np.prod(params.plot_sizes)
    ncol = params.effect_types.shape[0]

    # Initiate design matrix
    Y = params.prior
    if Y is None:
        Y = np.zeros((n, ncol), dtype=np.float64)

    feasible = False
    tries = 0
    while not feasible:
        # Add one try
        tries += 1

        # Initialize unconstrained
        Y = __init_unconstrained(
            params.effect_types, params.effect_levels, params.grps, 
            params.thetas, params.coords, Y, complete
        )

        # Constraint corrections
        Y = __correct_constraints(
            params.effect_types, params.effect_levels, params.grps, 
            params.thetas, params.coords, params.plot_sizes, params.fn.constraintso,
            Y, complete
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
