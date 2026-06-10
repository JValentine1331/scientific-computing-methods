# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:12:51 2026

@author: jessv
"""

"""
Modified Newton's Method

Implementation of the modified Newton method for approximating a root
of the equation f(x) = 0 when the multiplicity of the root is known.

For roots of multiplicity greater than one, the standard Newton method
loses its characteristic quadratic convergence and converges only
linearly. The modified Newton method restores quadratic convergence by
incorporating the root multiplicity into the iteration.

Given a root of multiplicity m, the iteration is

    p_(n+1) = p_n - m * f(p_n) / f'(p_n)

where m is the known multiplicity of the root.

The method retains the rapid local convergence of Newton's method while
improving performance for multiple roots. As with Newton's method,
convergence depends on the quality of the initial approximation and the
behavior of the function near the root.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""

from collections.abc import Callable

def modified_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    p0: float,
    multiplicity: int,
    tol: float = 1e-8,
    max_iter: int = 500,
    ) -> dict:
    
    """
    Approximate a root of f(x) = 0 using the modified Newton method.
    
    Parameters
    ----------
    f : Callable[[float], float]
        Function whose root is sought.
    
    df : Callable[[float], float]
        Derivative of the function f.
    
    p0 : float
        Initial approximation to the root.
    
    multiplicity : int
        Known multiplicity of the root.
    
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
    
    ValueError
        If multiplicity is less than one.
    
    Notes
    -----
    The modified Newton method generates a sequence according to
    
        p_(n+1) = p_n - m * f(p_n) / f'(p_n)
    
    where m is the known multiplicity of the root.
    
    For simple roots (m = 1), the method reduces to the standard Newton
    method. For multiple roots, the modification restores quadratic
    convergence that is otherwise lost by the standard Newton iteration.
    
    Like Newton's method, the modified Newton method is locally convergent
    and requires both a sufficiently accurate initial approximation and a
    nonzero derivative near the root.
    """
    #initialize count and storage
    i = 1
    history = [p0]
    
    #multiplicity validation
    if multiplicity < 1:
        raise ValueError("Multiplicity must be a positive integer.")
    
    #perform root finding
    while i <= max_iter:
        
        if abs(df(p0)) < 1e-14:
            raise ValueError("Derivative evaluated to zero.")
        
        #Compute next approx
        p = p0 - multiplicity * f(p0) / df(p0)
        history.append(p)
        
        if abs(f(p)) < tol:
            return {
                    "root": p,
                    "iterations": i,
                    "converged": True,
                    "history": history,
                    }
        
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

# def f(x):
#     return (x - 1)**3

# def df(x):
#     return 3*(x - 1)**2

# p0 = 2.0
# multiplicity = 3
# results = modified_newton(f, df, p0, multiplicity)
# print(results)