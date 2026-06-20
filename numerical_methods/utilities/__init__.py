from .error_metrics import (
    absolute_error,
    relative_error,
    percent_error,
)

from .convergence import (
    compute_error_sequence,
    estimate_order_of_convergence,
)

from .benchmarking import (
    time_function,
)

from .plotting import (
    plot_error_history,
)

__all__ = [
    "absolute_error",
    "relative_error",
    "percent_error",
    "compute_error_sequence",
    "estimate_order_of_convergence",
    "time_function",
    "plot_error_history",
]