import numpy as np

"""
Utilities for convergence analysis.
"""

def compute_error_sequence(history, true_value):
    """
    Compute the error at each iteration.

    Parameters
    ----------
    history : list
        Sequence of approximations.

    true_value : float
        Exact or reference value.

    Returns
    -------
    list
        Absolute error at each iteration.
    """

    return np.array([abs(x - true_value) for x in history])

def estimate_order_of_convergence(errors):
    """
    Estimate the observed order of convergence.

    Parameters
    ----------
    errors : list
        Sequence of errors.

    Returns
    -------
    list
        Estimated convergence orders.
    """
    
    orders = []
    
    for i in range(2, len(errors)):
        
        e_nm1 = errors[i - 2]
        e_n = errors[i - 1]
        e_np1 = errors[i]
        
        # Skip estimates once errors approach machine precision
        if min(e_nm1, e_n, e_np1) < 1e-14:
            continue
        
        p = np.log(e_np1 / e_n) / np.log(e_n / e_nm1)

        orders.append(p)


    return orders