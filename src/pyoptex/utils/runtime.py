import os
import signal
import inspect
import multiprocessing
import numpy as np

def set_nb_cores(n=1):
    """
    Set the runtime environment to use a specified number of cores.
    Setting the number of cores to one most often results in better
    performance due to the small dimensions of the matrices, especially
    when used in combination with :py:func:`parallel_generation`.

    .. note::
        This function MUST ALWAYS be called before any imports. For example,

        >>> from pyoptex.utils.runtime import set_nb_cores
        >>> set_nb_cores(1)
        >>> 
        >>> import numba
        >>> import numpy as np

    Parameters
    ----------
    n : int, optional
        The number of cores to use. Default is 1.
    """
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['OMP_THREAD_LIMIT'] = str(n)
    os.environ['OMP_DYNAMIC'] = 'FALSE'

    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['MKL_DYNAMIC'] = 'FALSE'

    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['GOTO_NUM_THREADS'] = str(n)

    os.environ['NUMEXPR_NUM_THREADS'] = str(n)

    os.environ['NUMBA_NUM_THREADS'] = str(n)

    os.environ['PYTHON_CPU_COUNT'] = str(n)

    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)

def parallel_generation(fn, *args, ncores=None, parallel_arg_name=None, **kwargs):
    """
    Allows parallel generation of the design. For example, the generation using the
    CODEX algorithm will parallelize the `nreps` argument over `ncores` cores. The 
    generation using the split-plot or fixed structure algorithms will parallelize
    the `n_tries` argument over `ncores` cores.

    If `ncores` is not specified, the number of cores will be set to the number of
    available cores on the machine.

    Parameters
    ----------
    fn : callable
        The function to parallelize.
    *args : tuple
        The arguments to pass to the function.
    ncores : int, optional
        The number of cores to use. If not specified, the number of available cores
        on the machine will be used.
    parallel_arg_name : str, optional
        The name of the argument to parallelize. If not specified, the function will
        look for an argument named `nreps`, `n_reps`, `ntries`, or `n_tries`.
    **kwargs : dict
        The keyword arguments to pass to the function.

    Returns
    -------
    Y : pd.DataFrame
        The best design found.
    state : :py:class:`State <pyoptex.doe.cost_optimal.utils.State>`
        The state of the best design.

    Examples
    --------
    Instead of calling

    >>> Y, state = create_cost_optimal_codex_design(
    >>>     params, nsims=nsims, nreps=nreps
    >>> )

    You call

    >>> from pyoptex.utils.runtime import parallel_generation
    >>> Y, state = parallel_generation(create_cost_optimal_codex_design, params, nsims=nsims, nreps=nreps)

    which will parallelize the number of repetitions over the specified or available number of cores.
    """
    # Validate the function
    assert callable(fn), 'The function argument must be a callable'
    assert ncores is None or (isinstance(ncores, int) and ncores > 0), 'The ncores argument must be an integer larger than zero, or None'

    # Extract the function arguments for use in starmap
    fn_args = [(k, v.default) 
                if v.default is not inspect.Parameter.empty 
                else (k, None)
                for k, v in inspect.signature(fn).parameters.items()]
    fn_arg_names = [name for name, _ in fn_args]
    
    # Get the parallelizable argument
    if parallel_arg_name is None:
        parallel_arg_idx, parallel_arg_default = None, None
        for name in ['nreps', 'n_reps', 'ntries', 'n_tries']:
            try:
                idx = fn_arg_names.index(name)
                parallel_arg_idx, parallel_arg_name, parallel_arg_default = \
                        idx, fn_args[idx][0], fn_args[idx][1]
                break
            except ValueError:
                pass
    else:
        parallel_arg_idx, parallel_arg_default = None, None
        try:
            idx = fn_arg_names.index(parallel_arg_name)
            parallel_arg_idx, parallel_arg_default = idx, fn_args[idx][1]
        except ValueError:
            pass
    
    # Validate the argument
    if parallel_arg_idx is None:
        raise ValueError('No parallelizable argument found, please provide a function with a '
                         'keyword argument named nreps, n_reps, ntries, or n_tries')

    # Retrieve the value of the parallelizable argument
    if len(args) > parallel_arg_idx:
        nreps = args[parallel_arg_idx]
    elif parallel_arg_name in kwargs:
        nreps = kwargs[parallel_arg_name]
    else:
        nreps = parallel_arg_default

    # Validate the argument
    if not isinstance(nreps, int) or nreps <= 0:
        raise ValueError('The parallelizable argument must be an integer greater than zero')

    # Determine the number of cores to use
    if ncores is None:
        ncores = os.cpu_count()
    ncores = min(ncores, nreps)

    # Compute the number of repetitions per core
    nreps_per_core = nreps // ncores
    nreps_remainder = nreps % ncores

    # Prepare the arguments
    args = list(args) + [kwargs[name] if name in kwargs else value for name, value in fn_args[len(args):]]
    args = [list(args)] * ncores
    for i in range(len(args)):
        if i < nreps_remainder:
            args[i][parallel_arg_idx] = nreps_per_core + 1
        else:
            args[i][parallel_arg_idx] = nreps_per_core

    # Map the results
    with multiprocessing.Pool(ncores) as p:
        print(f'Starting parallel generation with {nreps} repetitions on {ncores} cores')
        try:
            # Generate the designs in parallel
            mapresults = p.starmap_async(fn, args)
            results = mapresults.get()
        except KeyboardInterrupt:
            # Send stop signal to subprocesses and collect the results
            for proc in p._pool:
                os.kill(proc.pid, signal.SIGINT)
            results = mapresults.get()

    # Select the best result
    best_idx = np.argmax([state.metric for _, state in results])
    Y, state = results[best_idx]

    return Y, state
