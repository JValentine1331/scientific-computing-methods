# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:26:05 2026

@author: jessv
"""

from collections.abc import Callable

"""
Secant Method

Implementation of the secant method for approximating a root of the
equation f(x) = 0.

Starting from two initial approximations, the method generates a
sequence of iterates by replacing the derivative in Newton's method
with a finite-difference approximation. Each new approximation is
obtained from the intersection of the secant line through two
successive points on the graph of f with the x-axis.

The secant method avoids the need to evaluate derivatives while
typically converging more rapidly than bisection. Under suitable
conditions, the method exhibits superlinear convergence.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""


def secant(
    f: Callable[[float], float],
    p0: float,
    p1: float,
    tol: float = 1e-8,
    max_iter: int = 500,
    ) -> dict:
    
    """
    Approximate a root of f(x) = 0 using the secant method.
    
    Parameters
    ----------
    
    f : Callable[[float], float]
    Function whose root is sought.
    
    p0 : float
    First initial approximation to the root.
    
    p1 : float
    Second initial approximation to the root.
    
    tol : float, optional
    Desired tolerance for convergence. Iteration terminates
    when the absolute difference between successive iterates
    is less than tol. Default is 1e-8.
    
    max_iter : int, optional
    Maximum number of iterations permitted. Default is 500.
    
    Returns
    -------
    
    dict
    Dictionary containing:
    
    root : float
        Final root approximation.
    
    iterations : int
        Number of iterations performed.
    
    converged : bool
        True if the stopping criterion was satisfied.
    
    history : list[float]
        Sequence of iterates generated during the iteration.
    
    Raises
    ------
    
    ValueError
    If the secant denominator becomes zero during iteration.
    
    Notes
    -----
    
    The secant method generates a sequence according to
    
    p_n = p_(n-1)
          - f(p_(n-1))
            (p_(n-1) - p_(n-2))
            / (f(p_(n-1)) - f(p_(n-2)))
    
    and may be viewed as a derivative-free approximation to Newton's
    method. Although convergence is not guaranteed, the method often
    converges rapidly when the initial approximations are chosen near a
    simple root.
    """

    
    #perform first step calculation
    i = 2
    q0 = f(p0)
    q1 = f(p1)
    history = [p0, p1]
    
    while i <= max_iter:
        
        #check for divide by zero error
        if abs(q1 - q0) < 1e-14:
            raise ValueError("Secant denominator evaluated to zero.")
        else:
            pass
        
        #compute next step
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        history.append(p)
        
        if abs(p-p1) < tol:
            return {
                    "root": p,
                    "iterations": i,
                    "converged": True,
                    "history": history,
                    }
        else:
            i += 1
            p0 = p1
            q0 = q1
            p1 = p
            q1 = f(p)
            
    #If we get here, the method failed to converge
    return {
            "root": p,
            "iterations": max_iter,
            "converged": False,
            "history": history,
            }

# #Quick test
# def f(x):
#     return x**3 + 4*x**2 - 10

# p0 = 1.0
# p1 = 2.0

# results = secant(f, p0, p1)
# print(results)
