# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:26:05 2026

@author: jessv
"""

from collections.abc import Callable

"""
Newton's Method

Implementation of Newton's method for approximating a root of the
equation f(x) = 0.

Starting from an initial guess p0, the method generates a sequence of
approximations by replacing the function locally with its tangent line
and computing the x-intercept of that tangent. The iteration is given by

p_(n+1) = p_n - f(p_n)/f'(p_n)

When the initial guess is sufficiently close to a simple root and the
derivative is nonzero, Newton's method exhibits quadratic convergence,
making it one of the most efficient root-finding algorithms.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""


def newton(
        f: Callable[[float], float],
        df: Callable[[float], float],
        p0: float,
        tol: float = 1e-8,
        max_iter: int = 500,
        ) -> dict:
    
    """
    Approximate a root of f(x) = 0 using Newton's method.
    
    Parameters
    ----------
    
    f : Callable[[float], float]
    Function whose root is sought.
    
    df : Callable[[float], float]
    Derivative of f.
    
    p0 : float
    Initial approximation to the root.
    
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
    If the derivative evaluates to zero during iteration.
    
    Notes
    -----
    
    Newton's method generates a sequence according to
    
    p_(n+1) = p_n - f(p_n)/f'(p_n)
    
    and converges quadratically near a simple root when the function is
    sufficiently smooth and the initial approximation is chosen close to
    the solution. Poor initial guesses or vanishing derivatives may cause
    the method to fail to converge.
    """

    
    #initialize count and storage
    i = 1
    history = [p0]
    
    #perform root finding
    while i <= max_iter:
        
        if abs(df(p0)) < 1e-14:
            raise ValueError("Derivative evaluated to zero.")
        
        #Compute next approx
        p = p0 - f(p0) / df(p0)
        history.append(p)
        
        if abs(p - p0) < tol:
            return {
                    "root": p,
                    "iterations": i,
                    "converged": True,
                    "history": history,
                    }
        else:
            i += 1
            p0 = p
    
    #If we get here, the method failed to converge
    return {
            "root": p,
            "iterations": max_iter,
            "converged": False,
            "history": history,
            }

#Quick Test
def f(x):
    return x**3 + 4*x**2 - 10

def df(x):
    return 3*x**2 + 8*x

p0 = 1.5
results = newton(f, df, p0)
print(results)

def f(x):
    return x**2 - 2

def df(x):
    return 2*x

p0 = 1.0
results = newton(f, df, p0)
print(results)