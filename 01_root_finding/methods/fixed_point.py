# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:20:38 2026

@author: jessv
"""

import numpy as np
from collections.abc import Callable

"""
Fixed Point Iteration

Implementation of the fixed-point iteration method for approximating a
solution to the equation

g(x) = x.

Starting from an initial guess p0, the method generates a sequence of
approximations according to

p_(n+1) = g(p_n).

Under appropriate conditions, such as when g is a contraction mapping
near the fixed point, the sequence converges to a solution. The method
forms the foundation for several important iterative techniques in
numerical analysis and scientific computing.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""

def fixed_point(
                g: Callable[[float], float],
                p0: float,
                tol: float = 1e-8,
                max_iter: int = 500,
                ) -> dict:
    
    """
    Approximate a fixed point of g(x) using fixed-point iteration.
    
    Parameters
    ----------
    
    g : Callable[[float], float]
    Iteration function defining the fixed-point problem
    x = g(x).
    
    p0 : float
    Initial approximation to the fixed point.
    
    tol : float, optional
    Desired tolerance for convergence. Iteration terminates
    when the relative difference between successive iterates
    is less than tol. Default is 1e-8.
    
    max_iter : int, optional
    Maximum number of iterations permitted. Default is 500.
    
    Returns
    -------
    
    dict
    Dictionary containing:
    
    root : float
        Final fixed-point approximation.
    
    iterations : int
        Number of iterations performed.
    
    converged : bool
        True if the stopping criterion was satisfied.
    
    history : list[float]
        Sequence of iterates generated during the iteration.
    
    Notes
    -----
    
    The fixed-point iteration method generates a sequence according to
    
    p_(n+1) = g(p_n)
    
    and converges when successive iterates approach a fixed point.
    Convergence is not guaranteed and depends on the properties of
    the iteration function g. In particular, convergence is typically
    expected when g is a contraction mapping in a neighborhood of
    the fixed point.
    """

    #start iteration and initilize storage
    i = 1
    history = [p0]
    
    #perform root finding
    while i <= max_iter:
        
        #calculate next value
        p = g(p0)
        history.append(p)
        
        #test convergence
        if abs(p-p0) < tol:
            return  {
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
            
#Quick test

def g(x):
    return np.cos(x)

result = fixed_point(g, 1.0)

print(result)

p0 = 1.0

results = fixed_point(g, p0)
print(results)

def g(x):
    return (4 - x)**(1/3)

p0 = 1.5

results = fixed_point(g, p0)
print(results)

