"""
Error metrics used throughout the numerical methods library.
"""


def absolute_error(true_value, approximate_value):
    """
    Compute the absolute error.

    Parameters
    ----------
    true_value : float
        Exact or reference value.

    approximate_value : float
        Numerical approximation.

    Returns
    -------
    float
        Absolute error.
    """
    return abs(true_value - approximate_value)


def relative_error(true_value, approximate_value):
    """
    Compute the relative error.

    Parameters
    ----------
    true_value : float
        Exact or reference value.

    approximate_value : float
        Numerical approximation.

    Returns
    -------
    float
        Relative error.
    """
    if true_value == 0:
        raise ZeroDivisionError(
            "Relative error is undefined when true_value = 0."
        )

    return abs(true_value - approximate_value) / abs(true_value)


def percent_error(true_value, approximate_value):
    """
    Compute percent error.
    """
    return 100 * relative_error(true_value, approximate_value)