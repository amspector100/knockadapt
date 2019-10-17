import warnings
import numpy as np
import cvxpy as cp
import scipy as sp
from scipy import stats

from .utilities import chol2inv


def equicorrelated_block_matrix(Sigma, groups, tol = 1e-5):
    """ Calculates the block diagonal matrix S using
    the equicorrelated method described by Dai and Barber 2016.
    :param Sigma: true precision matrix of X, of dimension p x p
    :param groups: numpy array of length p, list of groups of variables
    :param tol: Tolerance for error allowed in eigenvalues computations
    """
        
    # Get eigenvalues and decomposition
    D = np.zeros((p, p))
    for j in np.unique(groups):
        
        #Select subset of cov matrix 
        inds = np.where,mv (groups == j)[0]
        full_inds = np.ix_(inds, inds)
        group_sigma = Sigma[full_inds]
        
        #Take square root of inverse
        inv_group_sigma = chol2inv(group_sigma)
        sqrt_inv_group_sigma = sp.linalg.sqrtm(inv_group_sigma)
        
        # Fill in D
        D[full_inds] = sqrt_inv_group_sigma
        
    min_d_eig = np.linalg.eig(D)[0].min()
    print(f'D min eig val is: {min_d_eig}')
        
    # Find minimum eigenvalue
    DSigD = np.einsum('pk,kj,jl->pl', D, Sigma, D)
    gamma = min(2 * np.linalg.eig(DSigD)[0].min(), 1)
    print(f'Gamma is: {gamma}')
    if np.imag(gamma) > tol:
        warnings.warn('The minimum eigenvalue is not real, is the cov matrix pos definite?')
    gamma = np.real(gamma)

    # Start to fill up S
    S = np.zeros((p, p))
    for j in np.unique(groups):
        
        # Select subset of cov matrix 
        inds = np.where(groups == j)[0]
        full_inds = np.ix_(inds, inds)
        group_sigma = Sigma[full_inds]
        
        # fill up S
        S[full_inds] = gamma * group_sigma
        
    return S



def solve_group_SDP(Sigma, groups, verbose = False):
    """ Solves the group SDP problem: extends the
    formulation from Barber and Candes 2015/
    Candes et al 2018 (MX Knockoffs)"""
    
    # Figure out sizes of groups
    p = Sigma.shape[0]
    m = groups.max()
    group_sizes = np.zeros(m)
    for j in groups:
        group_sizes[j-1] += 1

    # Sort Sigma by the groups for convenient block diagonalization of S
    inds_and_groups = [(i, group) for i, group in enumerate(groups)]
    inds_and_groups = sorted(inds_and_groups, key = lambda x: x[1])
    inds = [i for (i,j) in inds_and_groups]

    # Make sure we can unsort
    inv_inds = np.zeros(p)
    for i, j in enumerate(inds):
        inv_inds[j] = i
    inv_inds = inv_inds.astype('int32')

    # Sort the covariance matrix according to the groups
    sortedSigma = Sigma[inds][:, inds]

    # Create blocks of semidefinite matrix S,
    # as well as the whole matrix S
    variables = []
    constraints = []
    S_rows = []
    shift = 0
    for j in range(m):

        # Create block variable
        gj = int(group_sizes[j]) 
        Sj = cp.Variable((gj,gj), symmetric=True)
        constraints += [Sj >> 0]
        variables.append(Sj)

        # Create row of S
        if shift == 0 and shift + gj < p:
            rowj = cp.hstack([
                Sj, cp.Constant(np.zeros((gj, p - gj)))
            ])
        elif shift + gj < p:
            rowj = cp.hstack([
                cp.Constant(np.zeros((gj, shift))), 
                Sj, 
                cp.Constant(np.zeros((gj, p - gj - shift)))
            ])
        elif shift + gj == p and shift > 0:
            rowj = cp.hstack([
                cp.Constant(np.zeros((gj, shift))),
                Sj
            ])
        elif gj == p and shift == 0:
            rowj = cp.hstack([
                Sj
            ])

        else:
            raise ValueError(f'shift ({shift}) and gj ({gj}) add up to more than p ({p})')  
        S_rows.append(rowj)

        shift += gj

    # Construct S and Grahm Matrix
    S = cp.vstack(S_rows)
    sortedSigma = cp.Constant(sortedSigma)
    X = sortedSigma - S
    G = cp.bmat(
        [[sortedSigma, sortedSigma - S], [sortedSigma - S, sortedSigma]]
    )
    constraints += [G >> 0]

    # Construct optimization constraint:
    # This is equivalent to minimizing 
    # the L1 norm of Sigma_m - S_m 
    objective = cp.Minimize(cp.sum(cp.abs(sortedSigma - S)))
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(verbose = verbose)
    if verbose:
        print('Finished solving SDP!')
        
    # Return unsorted S value
    return S.value[inv_inds][:, inv_inds]


def group_gaussian_knockoffs(X, Sigma, groups, 
                             copies = 1, 
                             tol = 1e-5, 
                             method = 'sdp', 
                             return_S = False,
                             verbose = True,
                             sdp_verbose = True):
    """ Constructs group Gaussian MX knockoffs:
    This is not particularly efficient yet...
    :param X: numpy array of dimension n x p 
    :param Sigma: true covariance matrix of X, of dimension p x p
    :param method: 'equicorrelated' or 'sdp', how to construct
     the true Cov(X, tilde(X)), where tilde(X) is the knockoffs.
    :param groups: numpy array of length p, list of groups of X
    :param copies: integer number of knockoff copies of each observation to draw
    :param tol: Minimum eigenvalue allowed for cov matrix of knockoffs
    :param bool verbose: If true, will print stuff as it goes
    :param bool sdp_verbose: If true, will tell the SDP solver to be verbose.
    
    returns: copies x n x p numpy array of knockoffs"""
    
    # I follow the notation of Katsevich et al. 2019
    n = X.shape[0]
    p = X.shape[1]
    
    if groups.shape[0] != p:
        raise ValueError(f'Groups dimension ({groups.shape[0]}) and data dimension ({p}) do not match')
        
    # Get precision matrix
    invSigma = chol2inv(Sigma)
        
    # Calculate group-block diagonal matrix S 
    # using SDP, equicorrelated, or ASDP
    method = str(method).lower()
    if method == 'sdp':
        if verbose:
            print(f'Solving SDP for S with p = {p}')
        S = solve_group_SDP(Sigma, groups)
    elif method == 'equicorrelated':
        S = EquicorrelatedCovMatrix(Sigma, groups)
    else:
        raise ValueError(f'Method must be one of "equicorrelated" or "sdp", not {method}')
        
    # Check to make sure the methods worked
    #G = np.array([[Sigma, Sigma - S], [Sigma - S, Sigma]])
    min_eig1 = np.linalg.eig(2*Sigma - S)[0].min()
    if verbose:
        print(f'Minimum eigenvalue of 2Sigma - S is {min_eig1}')
        
    # Calculate MX knockoff moments...
    mu = X - np.einsum('pk,kl,ls', X, invSigma, S)
    V = 2*S - np.einsum('pk,kl,ls', S, invSigma, S)
    
    # Account for numerical errors
    min_eig = np.linalg.eig(V)[0].min()
    if verbose:
        print(f'Minimum eigenvalue of V is {min_eig}')
    if min_eig < tol:
        V += (tol - min_eig) * sp.sparse.eye(p)
    
    # ...and sample MX knockoffs! 
    knockoffs = stats.multivariate_normal.rvs(
        mean = np.zeros(p), cov = V, size = copies * n
    )
    knockoffs = knockoffs.reshape(n, p, copies)
    mu = mu.reshape(n, p, 1)
    knockoffs += mu

    # For debugging...
    if return_S:
        return knockoffs, S
    
    return knockoffs