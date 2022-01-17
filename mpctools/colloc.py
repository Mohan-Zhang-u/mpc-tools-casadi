import numpy as np
from scipy import special

"""
Python version of Octave's colloc function.

Adapted from code by John Eaton, which was itself translated from Fortran
routines from "Solution of Differential Equation Models by Polynomial
Approximation," by J. Villadsen and M. L. Michelsen.

Code makes use of numpy/scipy's polynomial and rootfinding capabilities so that
we don't have to do any rootfinding. We do, however, use recursive functions
to evaluate polynomials because numpy/scipy's built-in evaluation is
numerically unstable for high-order polynomials.

- Michael Risbeck
  risbeck@wisc.edu
  March 2015
"""

def weights(n,include0=True,include1=True):
    """
    Returns collocation weights for order n.
    
    Two optional arguments decide whether to include left or right endpoints.
    
    Returns [r,A,B,q] with r the roots (on the interval [0,1]), A the first
    derivative weights, B the second derivative weights, and q the quadrature
    weights. r and q are numpy rank-1 arrays, while A and B are numpy rank-2
    arrays.
    """
    
    alpha = 0
    beta = 0
    [d1,d2,d3,r] = jacobi(n,alpha,beta,include0,include1)
    A = dfopr(n,d1,d2,d3,r,"first")
    B = dfopr(n,d1,d2,d3,r,"second")
    q = dfopr(n,d1,d2,d3,r,"weights")
    
    return [r,A,B,q]

def jacobi(n,alpha,beta,include0=True,include1=True):
    """
    Returns roots and derivatives of jacobi polynomials.
    
    [d1, d2, d3, r] = jacobi(n, alpha, beta, include0, include1)
    
    d1, d2, and d3 are derivatives at the roots which are given in r.
    """
    # Suppress some SciPy warnings that may occur.
    oldNpInvalidSetting = np.seterr(invalid="ignore")["invalid"]    
    
    # Scipy uses different parameters for shifted Jacobi polynomials.
    q = beta + 1
    p = alpha + q
    
    # Get polynomial and find roots.
    Pn = special.sh_jacobi(n,p,q)
    r = list(np.real(Pn.weights[:,0]))    
    
    # Decide what to do about endpoints.
    if include0:
        r = [0] + r
    if include1:
        r = r + [1]
        
    # Now calculate derivatives.    
    N = len(r) # Total number of points.
    d1 = [1]*N
    d2 = [0]*N
    d3 = [0]*N
    
    # Use recursive formulas. Could probably be vectorized for speed.
    for i in range(N):
        x = r[i]
        for j in range(N):
            if j != i:
                y = x - r[j]
                d3[i] = y*d3[i] + 3*d2[i]
                d2[i] = y*d2[i] + 2*d1[i]
                d1[i] = y*d1[i]
    
    # Change back the settings.
    np.seterr(invalid=oldNpInvalidSetting)
    
    return [d1, d2, d3, np.array(r)]
        
def dfopr(n,d1,d2,d3,r,mode="weights"):
    """    
    Gets weighting for derivatives or quadrature.
    
    n is the order of the polynomail.
    
    d1, d2, d3, and r are the first through third derivatives at the roots r
    These are the outputs, e.g., of jacobi, and they should be lists.
    
    mode must be one of "weights", "first", or "second".
    """
    # Check mode.    
    if mode not in ["weights","first","second"]:
        raise ValueError("dfopr: Invalid choice for mode!")
    
    # Cast to numpy arrays.
    d1 = np.array(d1)
    d2 = np.array(d2)
    d3 = np.array(d3)
    r = np.array(r)
    
    # Now do stuff.    
    if mode == "weights":
        ax = r*(1-r)
        # Check if 0 and 1 are endpoints.
        if r[0] != 0:
            ax /= r**2
        if r[-1] != 1:
            ax /= (1 - r)**2
        M = ax/d1**2
        M /= np.sum(M)
    else:
        # First handle diagonal elements.
        if mode == "first":
            m = d2/(2*d1)
        else:
            m = d3/(3*d1)
        
        # Now do off-diagonal stuff.
        [ri, rj] = ijify(r)
        [d1i, d1j] = ijify(d1)
        [d2i, d2j] = ijify(d2)
        
        y = ri - rj
        badInds = (y == 0)
        y[badInds] = 1 # We will fix this later.
        M = d1i/(d1j*y)
        if mode == "second":
            M *= (d2i/d1i - 2/y)
        M[badInds] = 0 # Get rid of bad elements.
        
        # Add back in diagonal.
        M += np.diag(m)
        
    return M
    
def ijify(v):
    """
    Takes a rank 1 array and returns two castable rank 2 arrays.
    
    Example:
        v = np.array([1,2,3])
        [i,j] = ijify(v)
        print i - j
        
        [[ 0 -1 -2]
         [ 1  0 -1]
         [ 2  1  0]]

    This allows for numpy's broadcasting to eliminate for loops.           
    """
    v = np.array(v).flatten()
    i = v.reshape(v.shape + (1,))
    j = v.reshape((1,) + v.shape)

    return [i,j]    
   