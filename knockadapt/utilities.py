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
    min_eig = np.linalg.eigh(X)[0].min()
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

def random_permutation_inds(length):
    """ Returns indexes which correspond to a random permutation,
    as well as indexes which undo the permutation. Is truly random
    (calls np.random.seed()) but does not change random state."""

    # Save random state and change it
    st0 = np.random.get_state()
    np.random.seed()

    # Create inds and rev inds
    inds = np.arange(0, length, 1)
    np.random.shuffle(inds)
    rev_inds = [0 for _ in range(length)]
    for (i, j) in enumerate(inds):
        rev_inds[j] = i

    # Reset random state and return
    np.random.set_state(st0)
    return inds, rev_inds

# def calc_ccorr(Q11, sigma12, Q22):
#     """ Calculates canonical correlation between
#     two sets of variables of dimensions p and q.
#     :param Q11: precision matrix of first set of variables.
#     p x p numpy array. 
#     :param sigma12: covariance matrix between sets. 
#     p x q numpy array.
#     :param Q22: precision matrix of second set of variables.
#     q x q numpy array.

#     returns: canonical correlation 
#     """
#     pass
    
