import warnings
import numpy as np
import scipy as sp
from statsmodels.stats.moment_helpers import cov2corr

def chol2inv(X):
    """ Uses cholesky decomp to get inverse of matrix """
    triang = np.linalg.inv(np.linalg.cholesky(X))
    return np.dot(triang.T,triang)

def force_positive_definite(X, tol = 1e-3):
    """Forces X to be positive semidefinite with minimum eigenvalue of tol"""
    
    # Find minimum eigenvalue
    min_eig = np.linalg.eig(X)[0].min()
    imag_part = np.imag(min_eig)
    if imag_part != 0:
        warnings.warn(f'Uh oh: the minimum eigenvalue is not real (imag part = {imag_part})')
    min_eig = np.real(min_eig)
    
    # Check within tolerance
    if min_eig < tol:
        p = X.shape[0]
        X += (tol - min_eig) * sp.sparse.eye(p)
    return X

def calc_group_sizes(groups):
    """
    :param groups: p-length array of integers between 1 and m, 
    where m <= p
    returns: m-length array of group sizes """
    m = groups.max()
    group_sizes = np.zeros(m)
    for j in groups:
        group_sizes[j-1] += 1
    return group_sizes

def random_permutation(p):
    """ Returns a random permutation of length p and its inverse.
    Both the permutation and its inverse are denoted as arrays of 
    dim p, taking unique values from 0 to p-1"""
    permutation = np.random.choice(np.arange(0, p, 1), size = p, replace = False)
    inv_permutation = np.zeros(p)
    for i, j in enumerate(permutation):
        inv_permutation[j] = i
    inv_permutation = inv_permutation.astype('int32')
    return permutation, inv_permutation