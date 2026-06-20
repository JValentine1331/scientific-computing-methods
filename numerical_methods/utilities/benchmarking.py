"""
Utilities for timing and benchmarking numerical methods.
"""

import time


def time_function(func, *args, **kwargs):
    """
    Measure execution time of a function.

    Parameters
    ----------
    func : callable
        Function to evaluate.

    *args
        Positional arguments passed to func.

    **kwargs
        Keyword arguments passed to func.

    Returns
    -------
    result
        Return value of func.

    elapsed_time : float
        Execution time in seconds.
    """

    start = time.perf_counter()

    result = func(*args, **kwargs)

    end = time.perf_counter()

    elapsed_time = end - start

    return result, elapsed_time