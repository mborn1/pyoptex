# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cnp.import_array()

def x2fx(const double[:, ::1] Yenc not None, const long[:, ::1] modelenc not None):
    """
    Cython implementation to create the model matrix from the design matrix
    and model specification.

    Parameters
    ----------
    Yenc : np.ndarray[float64, ndim=2]
        The encoded design matrix.
    modelenc : np.ndarray[long, ndim=2]
        The encoded model, specified as in MATLAB.

    Returns
    -------
    Xenc : np.ndarray[float64, ndim=2]
        The model matrix
    """
    # Get dimensions
    cdef int n_runs = Yenc.shape[0]
    cdef int n_factors = Yenc.shape[1]
    cdef int n_terms = modelenc.shape[0]

    # Initialize output array
    cdef cnp.ndarray[cnp.double_t, ndim=2] Xenc = np.zeros((n_runs, n_terms), dtype=np.float64)
    cdef double[:, ::1] Xenc_view = Xenc

    # Loop variables
    cdef int i, j, k
    cdef double p_val
    cdef const long[::1] term

    for i in range(n_terms): 
        term = modelenc[i]
        for k in range(n_runs):
            p_val = 1.0
            for j in range(n_factors):
                if term[j] != 0:
                    if term[j] == 1:
                        p_val *= Yenc[k, j]
                    else:
                        p_val *= pow(Yenc[k, j], term[j])
            Xenc_view[k, i] = p_val
    return Xenc

def force_Zi_asc(cnp.ndarray[cnp.long_t, ndim=1] Zi not None):
    """
    Force ascending groups. In other words [0, 0, 2, 1, 1, 1]
    is transformed to [0, 0, 1, 2, 2, 2].

    Parameters
    ----------
    Zi : np.array(1d)
        The current grouping matrix
    
    Returns
    -------
    Zi : np.array(1d)
        The grouping matrix with ascending groups
    """
    cdef long[::1] Zi_view = Zi

    cdef long c_asc = 0
    cdef long c = Zi_view[0]
    cdef int i

    Zi_view[0] = c_asc
    for i in range(1, Zi_view.shape[0]):
        if Zi_view[i] != c:
            c_asc += 1
            c = Zi_view[i]
        Zi_view[i] = c_asc

    return Zi

def encode_design(const double[:, ::1] Y not None, const long[::1] effect_types not None, list coords=None):
    """
    Encode the design according to the effect types.
    Each categorical factor is encoded using
    effect-encoding, unless the coordinates are specified.

    It is the inverse of :py:func:`decode_design <pyoptex.utils.design.decode_design>`

    Parameters
    ----------
    Y : np.array(2d)
        The current design matrix.
    effect_types : np.array(1d) 
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).
    coords : None or list[np.ndarray]
        The possible coordinates for each factor. 

    Returns
    -------
    Yenc : np.array(2d)
        The encoded design-matrix 
    """
    # Extract parameters
    cdef int n_runs = Y.shape[0]
    cdef int n_factors = Y.shape[1]
    cdef long[::1] cols = np.zeros(n_factors, dtype=np.int64)
    cdef long ncols = 0

    cdef int i
    for i in range(n_factors):
        if effect_types[i] > 1:
            cols[i] = effect_types[i] - 1  
        else:
            cols[i] = 1
        ncols += cols[i]

    # Initialize encoding
    cdef cnp.ndarray[cnp.double_t, ndim=2] Yenc = np.zeros((n_runs, ncols), dtype=np.float64)
    cdef double[:, ::1] Yenc_view = Yenc
    
    # Loop over factors
    cdef int j, k, start, val
    cdef double[:, ::1] coord
    for i in range(n_runs):
        start = 0
        for j in range(n_factors):
            if effect_types[j] == 1:
                Yenc_view[i, start] = Y[i, j]
                start += 1
            else:
                val = <int>Y[i, j]
                if coords is None:
                    if val < cols[j]:
                        Yenc_view[i, start:start+cols[j]] = 0
                        Yenc_view[i, start+val] = 1
                    else:
                        Yenc_view[i, start:start+cols[j]] = -1
                else:
                    coord = coords[j]
                    Yenc_view[i, start:start+cols[j]] = coord[val]
                start += cols[j]

    return Yenc

def decode_design(const double[:, ::1] Y not None, const long[::1] effect_types not None, list coords=None):
    """
    Decode the design according to the effect types.
    Each categorical factor is decoded from
    effect-encoding, unless the coordinates are specified.

    It is the inverse of :py:func:`encode_design <pyoptex.utils.design.encode_design>`

    Parameters
    ----------
    Y : np.array(2d)
        The effect-encoded design matrix.
    effect_types : np.array(1d) 
        An array indicating whether the effect is continuous (=1)
        or categorical (with >1 levels).
    coords: None or list[np.ndarray]
        Coordinates to be used for decoding the categorical variables.

    Returns
    -------
    Ydec : np.array(2d)
        The decoded design-matrix 
    """
    # Extract parameters
    cdef int n_runs = Y.shape[0]
    cdef int n_factors = effect_types.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=2] Ydec = np.zeros((n_runs, n_factors), dtype=np.float64)
    cdef double[:, ::1] Ydec_view = Ydec
    
    # Loop variables
    cdef int i, j, k, l, ncols, max_idx, start
    cdef bint all_match
    cdef double max_val
    cdef double[:, ::1] coord

    # Decoding process
    for i in range(n_runs):
        start = 0
        for j in range(n_factors):
            if effect_types[j] == 1:
                # Copy the value
                Ydec_view[i, j] = Y[i, start]
                start += 1
            else:
                # Determine the number of encoded columns
                ncols = effect_types[j] - 1

                # Decode the value
                if coords is None:
                    if Y[i, start] == -1:
                        # Last value in the encoding
                        Ydec_view[i, j] = ncols
                    else:
                        # Search for the one
                        for k in range(ncols):
                            if Y[i, start + k] == 1:
                                Ydec_view[i, j] = k
                                break

                else:
                    # Extract coordinate for this effect                    
                    coord = coords[j]

                    # Search for the index of the coordinate
                    for k in range(coord.shape[0]):
                        # Determine if all elements are a match
                        all_match = True
                        for l in range(ncols):
                            if coord[k, l] != Y[i, start + l]:
                                all_match = False
                                break

                        # Set the value
                        if all_match:
                            Ydec_view[i, j] = k
                            break

                # Increase the counter
                start += ncols

    return Ydec
