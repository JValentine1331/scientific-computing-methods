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
        marker="o",
        label=method_name
    )

    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")

    if method_name:
        plt.title(f"{method_name} Error History")
        plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()