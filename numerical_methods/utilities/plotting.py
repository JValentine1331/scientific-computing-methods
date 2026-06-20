"""
Plotting utilities for numerical methods.
"""

import matplotlib.pyplot as plt


def plot_error_history(errors,
                       method_name="Newton"):
    """
    Plot error versus iteration number on a semilog scale.

    Parameters
    ----------
    errors : list or array-like
        Error sequence.
    """

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        range(len(errors)),
        errors,
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.title("Error History")
    plt.grid(True)

    plt.show()