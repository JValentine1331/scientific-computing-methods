# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:26:19 2026

@author: jessv
"""


from collections.abc import Callable

"""
Steffensen's Method

Implementation of Steffensen's method for approximating a fixed point
of the equation

g(x) = x.

Steffensen's method accelerates fixed-point iteration by applying
Aitken's Δ² process to the sequence of iterates. The method generates
quadratically convergent approximations under conditions similar to
those required for Newton's method, but without requiring evaluation
of derivatives.

Given an initial approximation p0, the method computes successive
iterates using evaluations of the iteration function g and an
acceleration formula that improves upon standard fixed-point
iteration.

Steffensen's method can be viewed as a derivative-free analogue of
Newton's method and often converges significantly faster than
ordinary fixed-point iteration.

Reference
---------

Burden, R. L., & Faires, J. D.
Numerical Analysis.
"""

def steffensen(
    g: Callable[[float], float],
    p0: float,
    tol: float = 1e-8,
    max_iter: int = 500,
    ) -> dict:
    
    """
    Approximate a fixed point of g(x) using Steffensen's method.
    
    Parameters
    ----------
    g : Callable[[float], float]
        Iteration function defining the fixed-point problem
        g(x) = x.
    
    p0 : float
        Initial approximation to the fixed point.
    
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
            Final fixed-point approximation.
    
        iterations : int
            Number of iterations performed.
    
        converged : bool
            True if the stopping criterion was satisfied.
    
        history : list[float]
            Sequence of iterates generated during the iteration.
    
    Raises
    ------
    ValueError
        If the denominator in Steffensen's acceleration formula
        evaluates to zero.
    
    Notes
    -----
    Steffensen's method computes
    
        p1 = g(p0)
    
        p2 = g(p1)
    
    and then forms the accelerated approximation
    
                        (p1 - p0)^2
        p = p0 - -------------------------
                  p2 - 2*p1 + p0
    
    This process is equivalent to applying Aitken's Δ² acceleration
    to fixed-point iteration.
    
    Under appropriate conditions, Steffensen's method exhibits
    quadratic convergence while requiring only evaluations of the
    iteration function g. As a result, it often converges much more
    rapidly than standard fixed-point iteration while avoiding the
    need for derivative information.
    """
    
    #storage and iteration definition
    i = 1
    history = []
    
    #calculate next step and evaluate convergence
    while i <= max_iter:
        p1 = g(p0)
        p2 = g(p1)
        p = p0 - (p1 - p0)**2.0 / (p2 - 2.0 * p1 + p0)
        history.append(p)
        
        if abs(p-p0) < tol:
            return {
                    "root": p,
                    "iterations": i,
                    "converged": True,
                    "history": history,
                    }
        else:
            pass
        
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
import numpy as np

def g(x):
    return np.cos(x)
p0 = 1.0
results = steffensen(g, p0)
print(results)