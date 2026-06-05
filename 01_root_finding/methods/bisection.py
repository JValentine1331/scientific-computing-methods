# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:20:38 2026

@author: jessv
"""

import numpy as np
from collections.abc import Callable

"""
Bisection Method

Implementation of the classical bisection method for approximating a root
of a continuous function on an interval [a, b].

The method requires that f(a) and f(b) have opposite signs, ensuring
that at least one root exists in the interval by the Intermediate Value
Theorem. At each iteration, the interval is halved and the subinterval
containing the root is retained.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""

def signum(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 500,
    ) -> dict:
    
   
    """
    Approximate a root of f(x) = 0 using the bisection method.
    
    Parameters
    ----------
    f : Callable[[float], float]
        Continuous function whose root is sought.
    
    a : float
        Left endpoint of the initial interval.
    
    b : float
        Right endpoint of the initial interval.
    
    tol : float, optional
        Desired tolerance for the interval width. Iteration terminates
        when (b - a) / 2 < tol. Default is 1e-8.
    
    max_iter : int, optional
        Maximum number of iterations permitted. Default is 100.
    
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
            Sequence of midpoint approximations generated during
            iteration.
    
    Raises
    ------
    ValueError
        If f(a) and f(b) do not have opposite signs and therefore do
        not bracket a root.
    
    Notes
    -----
    The bisection method converges linearly and guarantees convergence
    provided that the function is continuous on [a, b] and the initial
    interval brackets a root.
    """

    #begin count and evaluate function at endpoint
    i = 1
    FA = f(a)
    FB = f(b)
    
    #perform a sign check
    if signum(FA) * signum(FB) == 1:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    #initlize storage
    history = []
    
    while i <= max_iter:
        
        #compute midpoint
        p = a + (b - a) / 2.0
        FP = f(p)
        history.append(p)
        
        #establish convergence criteria
        if np.abs(FP) < tol or (b - a) / 2.0 < tol:
            
            return {
                    "root": p,
                    "iterations": i,
                    "converged": True,
                    "history": history,
                    }
        
        #if root isn't found, proceed
        else:
            i += 1
            if signum(FA) * signum(FP) == 1:
                a = p
                FA = FP
            else:
                b = p
     
    # If we get here, the method failed to converge
    return {
            "root": p,
            "iterations": max_iter,
            "converged": False,
            "history": history,
            }

#Quick Test 
def f(x):
    return x**3 + 4*x**2 - 10

result = bisection(f, 1, 2)

print(result)

def g(x):
    return x**2 + 1

bisection(g, -1, 1)