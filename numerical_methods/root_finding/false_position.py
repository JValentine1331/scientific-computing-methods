# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:26:05 2026

@author: jessv
"""


from collections.abc import Callable

def signum(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    
"""
False Position Method

Implementation of the False Position (Regula Falsi) method for
approximating a root of the equation f(x) = 0.

Like the bisection method, False Position maintains a bracketing
interval whose endpoints have function values of opposite sign,
thereby guaranteeing that a root remains within the interval.
Instead of selecting the midpoint, the method computes the
x-intercept of the secant line joining the interval endpoints
and uses this value as the next approximation.

The False Position method combines the robustness of bracketing
methods with a secant-line approximation and often converges more
rapidly than bisection. However, convergence may become slow when
one endpoint remains fixed for many iterations.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""


def false_position(
    f: Callable[[float], float],
    p0: float,
    p1: float,
    tol: float = 1e-8,
    max_iter: int = 500,
    ) -> dict:
    
    """
    Approximate a root of f(x) = 0 using the False Position method.
    
    Parameters
    ----------
    
    f : Callable[[float], float]
    Continuous function whose root is sought.
    
    p0 : float
    Left endpoint of the initial bracketing interval.
    
    p1 : float
    Right endpoint of the initial bracketing interval.
    
    tol : float, optional
    Desired tolerance for convergence. Iteration terminates
    when the absolute difference between successive
    approximations is less than tol. Default is 1e-8.
    
    max_iter : int, optional
    Maximum number of iterations permitted. Default is 500.
    
    Returns
    -------
    
    dict
    Dictionary containing:
    
    ```
    root : float
        Final root approximation.
    
    iterations : int
        Number of iterations performed.
    
    converged : bool
        True if the stopping criterion was satisfied.
    
    history : list[float]
        Sequence of approximations generated during
        iteration.
    
    Raises
    ------
    
    ValueError
    If f(p0) and f(p1) do not have opposite signs and
    therefore do not bracket a root.
    
    Notes
    -----
    
    The False Position method generates a sequence of approximations
    by computing the x-intercept of the secant line joining the
    points (p0, f(p0)) and (p1, f(p1)).
    
    Unlike the secant method, False Position preserves a bracketing
    interval throughout the iteration, providing a guarantee of
    convergence for continuous functions whose initial interval
    contains a root. The method generally converges faster than
    bisection but may converge slowly when one endpoint of the
    interval remains unchanged for many iterations.
    """

    
    #First step
    i = 2
    q0 = f(p0)
    q1 = f(p1)
    
    if signum(q0) * signum(q1) > 0:
        raise ValueError("f(p0) and f(p1) must have opposite signs." )
    
    history = [p0, p1]
    
    #compute next steps
    while i <= max_iter:
      
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        history.append(p)
        
        if abs(p - p1) < tol:
            return {
                    "root": p,
                    "iterations": i,
                    "converged": True,
                    "history": history,
                    }
        
        #if convergence is not met
        else:
            i += 1 
            q = f(p)
            
            #check bracketing
            if signum(q) * signum(q1) < 0:
                p0 = p1
                q0 = q1
            else:
                pass
            p1 = p
            q1 = q
    
    #If we get here, the method failed to converge
    return {
            "root": p,
            "iterations": max_iter,
            "converged": False,
            "history": history,
            }

#quick test
# def f(x):
#     return x**3 + 4*x**2 - 10

# p0 = 1.
# p1 = 2.
# results = false_position(f, p0, p1)
# print(results)

# def f(x):
#     return x**2 - 2

# results = false_position(f, p0, p1)
# print(results)

# import numpy as np
# def f(x):
#     return np.cos(x) - x

# results = false_position(f, 0, 1)
# print(results)

# import numpy as np

# def f(x):
#     return np.exp(x) - 3*x

# print(f(0))
# print(f(1))
# print(f(0) * f(1))

# from bisection import bisection
# results = bisection(f, 0, 1)
# print(results)
# results = false_position(f, 0, 1)
# print(results)

# def f(x):
#     return x**2 + 1

# results = false_position(f, -1, 1)
# print(results)
