import warnings
import numpy as np
import scipy as sp
from scipy import stats
import scipy.linalg

from .utilities import calc_group_sizes, preprocess_groups
from .utilities import shift_until_PSD, scale_until_PSD
from . import utilities
from . import mrc

# Multiprocessing tools
from functools import partial
from multiprocessing import Pool

# For SDP
import time
import cvxpy as cp
from pydsdp.dsdp5 import dsdp

# Options for SDP solver
OBJECTIVE_OPTIONS = ["abs", "pnorm", "norm"]


def TestIfCorrMatrix(Sigma):
    """ Tests if a square matrix is a correlation matrix """
    p = Sigma.shape[0]
    diag = np.diag(Sigma)
    if np.sum(np.abs(diag - np.ones(p))) > p * 1e-2:
        raise ValueError("Sigma is not a correlation matrix. Scale it properly first.")


def calc_min_group_eigenvalue(Sigma, groups, tol=1e-5, verbose=False):
    """
    Calculates the minimum "group" eigenvalue of a covariance 
    matrix Sigma: see Dai and Barber 2016. This is useful for
    constructing equicorrelated knockoffs.
    :param Sigma: true precision matrix of X, of dimension p x p
    :param groups: numpy array of length p, list of groups of variables
    :param tol: Tolerance for error allowed in eigenvalues computations
    """

    # Test corr matrix
    TestIfCorrMatrix(Sigma)

    # Construct group block matrix apprx of Sigma
    p = Sigma.shape[0]
    D = np.zeros((p, p))
    for j in np.unique(groups):

        # Select subset of cov matrix
        inds = np.where(groups == j)[0]
        full_inds = np.ix_(inds, inds)
        group_sigma = Sigma[full_inds]

        # Take square root of inverse
        inv_group_sigma = utilities.chol2inv(group_sigma)
        sqrt_inv_group_sigma = sp.linalg.sqrtm(inv_group_sigma)

        # Fill in D
        D[full_inds] = sqrt_inv_group_sigma

    # Test to make sure this is positive definite
    min_d_eig = np.linalg.eigh(D)[0].min()
    if min_d_eig < -1 * tol:
        raise ValueError(f"Minimum eigenvalue of block matrix D is {min_d_eig}")

    # Find minimum eigenvalue
    DSig = np.dot(D, Sigma)
    DSigD = np.dot(DSig, D)
    gamma = min(2 * np.linalg.eigh(DSigD)[0].min(), 1)

    # Warn if imaginary
    if np.imag(gamma) > tol:
        warnings.warn(
            "The minimum eigenvalue is not real, is the cov matrix pos definite?"
        )
    gamma = np.real(gamma)

    return gamma


def equicorrelated_block_matrix(Sigma, groups, tol=1e-5, verbose=False, num_iter=10):
    """ Calculates the block diagonal matrix S using
    the equicorrelated method described by Dai and Barber 2016.
    :param Sigma: true precision matrix of X, of dimension p x p
    :param groups: numpy array of length p, list of groups of variables
    :param tol: Tolerance for error allowed in eigenvalues computations
    """

    # Get eigenvalues and decomposition
    p = Sigma.shape[0]
    if groups is None:
        groups = np.arange(1, p+1, 1)
    if np.all(groups == np.arange(1, p+1, 1)):
        gamma = min(2*np.linalg.eigh(Sigma)[0].min(), 1)
        S = gamma * np.eye(p)
    else:
        gamma = calc_min_group_eigenvalue(Sigma, groups, tol=tol, verbose=verbose)

        # Start to fill up S
        S = np.zeros((p, p))
        for j in np.unique(groups):

            # Select subset of cov matrix
            inds = np.where(groups == j)[0]
            full_inds = np.ix_(inds, inds)
            group_sigma = Sigma[full_inds]

            # fill up S
            S[full_inds] = gamma * group_sigma

    # Scale to make this PSD using binary search
    S, _ = scale_until_PSD(Sigma, S, tol, num_iter)

    return S

def solve_SDP(
    Sigma,
    verbose=False,
    num_iter=10,
    tol=1e-2,
    **kwargs
):
    """ 
    Much faster solution to SDP without grouping
    """

    # The code here does not make any intuitive sense,
    # The DSDP solver is super fast but its input format is nonsensical.
    # This basically solves:
    # minimize c^T y s.t.
    # Ay <= b
    # F0 + y1 F1 + ... + yp Fp > 0 where F0,...Fp are PSD matrices
    # However the variables here do NOT correspond to the variables
    # in the equations because the Sedumi format is strange - 
    # see https://www.ece.uvic.ca/~wslu/Talk/SeDuMi-Remarks.pdf
    # Also, the "l" argument in the K options dictionary 
    # in the SDP package may not work.
    # TODO: make this work for group SDP. 
    # Idea: basically, add more variables for the off-diagonal elements
    # and maximize their sum subject to the constraint that they can't
    # be larger than the corresponding off-diagonal elements of Sigma
    # (I.e. make the linear constraints larger...)

    # Constants 
    p = Sigma.shape[0]
    maxtol = np.linalg.eigh(Sigma)[0].min() / 10
    if tol > maxtol and verbose:
        warnings.warn(
            f"Reducing SDP tol from {tol} to {maxtol}, otherwise SDP would be infeasible"
        )
    tol = min(maxtol, tol)

    # Construct C (-b + vec(F0) from above)
    # Note the tolerance here prevents the min. val
    # of S from being too small.
    Cl1 = np.diag(-1*tol*np.ones(p)).reshape(1, p**2)
    Cl2 = np.diag(np.ones(p)).reshape(1, p**2)
    Cs = np.reshape(2*Sigma,[1,p*p])
    C = np.concatenate([Cl1,Cl2,Cs],axis=1)

    # Construct A 
    rows = []
    cols = []
    data = []
    for j in range(p):
        rows.append(j)
        cols.append((p+1)*j)
        data.append(-1) 
    Al1 = sp.sparse.csr_matrix((data, (rows, cols)))
    Al2 = -1*Al1.copy()
    As = Al2.copy()
    A = sp.sparse.hstack([Al1, Al2, As])

    # Construct b
    b = np.ones([p,1])

    # Options
    K = {}
    K['s'] = [p,p,p]
    OPTIONS = {
        'gaptol':1e-6,
        'maxit':1000,
        'logsummary':1 if verbose else 0,
        'outputstats':1 if verbose else 0,
        'print':1 if verbose else 0
    }

    # Solve
    warnings.filterwarnings("ignore")
    result = dsdp(A, b, C, K, OPTIONS=OPTIONS)
    warnings.resetwarnings()

    # Raise an error if unsolvable
    status = result['STATS']['stype']
    if status != 'PDFeasible':
        raise ValueError(
            f"DSDP solver returned status {status}, should be PDFeasible"
        )
    S = np.diag(result['y'])

    # Scale to make this PSD using binary search
    S, gamma = scale_until_PSD(Sigma, S, tol, num_iter)
    if verbose:
        mineig = np.linalg.eigh(2 * Sigma - S)[0].min()
        print(
            f"After SDP, mineig is {mineig} after {num_iter} line search iters. Gamma is {gamma}"
        )

    return S

def solve_group_SDP(
    Sigma,
    groups=None,
    verbose=False,
    objective="abs",
    norm_type=2,
    num_iter=10,
    tol=1e-2,
    **kwargs,
):
    """ Solves the group SDP problem: extends the
    formulation from Barber and Candes 2015/
    Candes et al 2018 (MX Knockoffs). Note this will be 
    much faster with equal-sized groups and objective="abs."
    :param Sigma: true covariance (correlation) matrix, 
    p by p numpy array.
    :param groups: numpy array of length p with
    integer values between 1 and m. 
    :param verbose: if True, print progress of solver
    :param objective: How to optimize the S matrix for 
    group knockoffs. (For ungrouped knockoffs, using the 
    objective = 'abs' is strongly recommended.)
    There are several options:
        - 'abs': minimize sum(abs(Sigma - S))
        between groups and the group knockoffs.
        - 'pnorm': minimize Lp-th matrix norm.
        Equivalent to abs when p = 1.
        - 'norm': minimize different type of matrix norm
        (see norm_type below).
    :param norm_type: Means different things depending on objective.
        - When objective == 'pnorm', i.e. objective is Lp-th matrix norm, 
          which p to use. Can be any float >= 1. 
        - When objective == 'norm', can be 'fro', 'nuc', np.inf, or 1
          (i.e. which other norm to use).
    Defaults to 2.
    :param num_iter: We do a line search and scale S at the end to make 
    absolutely sure there are no numerical errors. Defaults to 10.
    :param tol: Minimum eigenvalue of S must be greater than this.
    """

    # By default we lower the convergence epsilon a bit for drastic speedup.
    if "eps" not in kwargs:
        kwargs["eps"] = 5e-3

    # Default groups
    p = Sigma.shape[0]
    if groups is None:
        groups = np.arange(1, p+1, 1)

    # Test corr matrix
    TestIfCorrMatrix(Sigma)

    # Check to make sure the objective is valid
    objective = str(objective).lower()
    if objective not in OBJECTIVE_OPTIONS:
        raise ValueError(f"Objective ({objective}) must be one of {OBJECTIVE_OPTIONS}")
    # Warn user if they're using a weird norm...
    if objective == "norm" and norm_type == 2:
        warnings.warn(
            "Using norm objective and norm_type = 2 can lead to strange behavior: consider using Frobenius norm"
        )
    # Find minimum tolerance, possibly warn user if lower than they specified
    maxtol = np.linalg.eigh(Sigma)[0].min() / 1.1
    if tol > maxtol and verbose:
        warnings.warn(
            f"Reducing SDP tol from {tol} to {maxtol}, otherwise SDP would be infeasible"
        )
    tol = min(maxtol, tol)

    # Figure out sizes of groups
    m = groups.max()
    group_sizes = utilities.calc_group_sizes(groups)

    # Possibly solve non-grouped SDP
    if m == p:
        return solve_SDP(
            Sigma=Sigma,
            verbose=verbose,
            num_iter=num_iter,
            tol=tol,
        )

    # Sort the covariance matrix according to the groups
    inds, inv_inds = utilities.permute_matrix_by_groups(groups)
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
        Sj = cp.Variable((gj, gj), symmetric=True)
        constraints += [Sj >> 0]
        variables.append(Sj)

        # Create row of S
        if shift == 0 and shift + gj < p:
            rowj = cp.hstack([Sj, cp.Constant(np.zeros((gj, p - gj)))])
        elif shift + gj < p:
            rowj = cp.hstack(
                [
                    cp.Constant(np.zeros((gj, shift))),
                    Sj,
                    cp.Constant(np.zeros((gj, p - gj - shift))),
                ]
            )
        elif shift + gj == p and shift > 0:
            rowj = cp.hstack([cp.Constant(np.zeros((gj, shift))), Sj])
        elif gj == p and shift == 0:
            rowj = cp.hstack([Sj])

        else:
            raise ValueError(
                f"shift ({shift}) and gj ({gj}) add up to more than p ({p})"
            )
        S_rows.append(rowj)

        # Incremenet shift
        shift += gj

    # Construct S and Grahm Matrix
    S = cp.vstack(S_rows)
    sortedSigma = cp.Constant(sortedSigma)
    constraints += [2 * sortedSigma - S >> 0]

    # Construct optimization objective
    if objective == "abs":
        objective = cp.Minimize(cp.sum(cp.abs(sortedSigma - S)))
    elif objective == "pnorm":
        objective = cp.Minimize(cp.pnorm(sortedSigma - S, norm_type))
    elif objective == "norm":
        objective = cp.Minimize(cp.norm(sortedSigma - S, norm_type))
    # Note we already checked objective is one of these values earlier

    # Construct, solve the problem.
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=verbose, **kwargs)
    if verbose:
        print("Finished solving SDP!")

    # Unsort and get numpy
    S = S.value
    if S is None:
        raise ValueError(
            "SDP formulation is infeasible. Try decreasing the tol parameter."
        )
    S = S[inv_inds][:, inv_inds]

    # Clip 0 and 1 values
    for i in range(p):
        S[i, i] = max(tol, min(1 - tol, S[i, i]))

    # Scale to make this PSD using binary search
    S, gamma = scale_until_PSD(Sigma, S, tol, num_iter)
    if verbose:
        mineig = np.linalg.eigh(2 * Sigma - S)[0].min()
        print(
            f"After SDP, mineig is {mineig} after {num_iter} line search iters. Gamma is {gamma}"
        )

    # Return unsorted S value
    return S


def solve_group_ASDP(
    Sigma,
    groups,
    Sigma_groups=None,
    alpha=None,
    verbose=True,
    num_iter=10,
    max_block=100,
    numprocesses=1,
    tol=1e-2,
    **kwargs,
):
    """
    :param Sigma: covariance (correlation) matrix
    :param groups: numpy array of length p with
    integer values between 1 and m. 
    :param Sigma_groups: array of groups for block approximation
    OR dictionary mapping groups to integers between 1 and l, 
    with 1 < l: used to construct an l-block approximation of Sigma. 
    :param alpha: If Sigma_groups is none, combines sets 
    of alpha groups to define the blocks used in the block
    approximation of Sigma. E.g. if alpha = 2, will combine
    pairs of groups in order to define blocks.
    (How does it choose which pairs of groups to combine?
    It's basically random right now.) Defaults to using
    blocks of sizes of about 100 
    """

    # Test corr matrix
    TestIfCorrMatrix(Sigma)

    # Shape of Sigma
    p = Sigma.shape[0]
    m = np.unique(groups).shape[0]

    # Possibly automatically choose alpha
    # TO DO - pick these groups better (hier clustering)
    if alpha is None and Sigma_groups is None:
        group_sizes = calc_group_sizes(groups)
        max_group = group_sizes.max()

        # Trivially, if max_block < p
        if max_block <= p:
            alpha = m
            Sigma_groups = {x: 0 for x in np.unique(groups)}

        # If the max group size is small enough:
        elif max_block >= max_group:
            alpha = int(max_block / max_group)
            Sigma_groups = {x: int(x / alpha) for x in np.unique(groups)}

        # Else we have to split up groups to make comp. tractable
        else:

            # How much to split up groups by
            inv_alpha = int(max_group / max_block)

            # Split them up
            unique_groups = np.unique(groups)
            group_counter = {j: 0 for j in unique_groups}
            Sigma_groups = []
            for i, group_id in enumerate(groups):
                group_counter[group_id] += 1
                count = group_counter[group_id]
                Sigma_groups.append(inv_alpha * group_id + count % inv_alpha)

            # Preprocess
            Sigma_groups = preprocess_groups(Sigma_groups)

    # Possibly infer Sigma_groups
    if isinstance(Sigma_groups, dict):
        Sigma_groups = np.array([Sigma_groups[i] for i in groups])

    # If verbose, report on block sizes
    if verbose:
        sizes = calc_group_sizes(Sigma_groups + 1)
        max_block = sizes.max()
        mean_block = int(sizes.mean())
        num_blocks = sizes.shape[0]
        print(
            f"ASDP has split Sigma into {num_blocks} blocks, of mean size {mean_block} and max size {max_block}"
        )

    # Make sure this is zero indexed
    Sigma_groups = preprocess_groups(Sigma_groups) - 1

    # Construct block approximation
    num_blocks = Sigma_groups.max() + 1
    Sigma_blocks = [
        Sigma[Sigma_groups == i][:, Sigma_groups == i] for i in range(num_blocks)
    ]
    group_blocks = [groups[Sigma_groups == i] for i in range(num_blocks)]
    group_blocks = [preprocess_groups(x) for x in group_blocks]

    # Feed approximation to SDP, possibly using multiprocessing
    if numprocesses == 1:
        S_blocks = []
        for i in range(num_blocks):
            S_block = solve_group_SDP(
                Sigma=Sigma_blocks[i],
                groups=group_blocks[i],
                tol=tol,
                **kwargs
            )
            S_blocks.append(S_block)
    else:
        # Partial function for proper mapping
        partial_group_SDP = partial(solve_group_SDP, **kwargs)

        # Construct arguments for pool
        all_arguments = []
        for i in range(num_blocks):
            args = (Sigma_blocks[i], group_blocks[i])
            all_arguments.append(args)

        # And solve (this is trivially parallelizable)
        with Pool(numprocesses) as thepool:
            S_blocks = thepool.starmap(partial_group_SDP, all_arguments)

    S_sorted = sp.linalg.block_diag(*S_blocks)


    # Create indexes to unsort S
    inds_and_groups = [(i, group) for i, group in enumerate(Sigma_groups)]
    inds_and_groups = sorted(inds_and_groups, key=lambda x: x[1])
    inds = [i for (i, j) in inds_and_groups]

    # Make sure we can unsort
    inv_inds = np.zeros(p)
    for i, j in enumerate(inds):
        inv_inds[j] = i
    inv_inds = inv_inds.astype("int32")

    # Unsort
    S = S_sorted[inv_inds]
    S = S[:, inv_inds]

    # Scale to make this PSD using binary search
    S, gamma = scale_until_PSD(Sigma, S, tol, num_iter)
    if verbose:
        mineig = np.linalg.eigh(2 * Sigma - S)[0].min()
        print(
            f"After ASDP, mineig is {mineig} after {num_iter} line search iters. Gamma is {gamma}"
        )

    return S


def parse_method(method, groups, p):
    """ Decides which method to use to create the 
    knockoff S matrix """
    if method is not None:
        return method
    if np.all(groups == np.arange(1, p + 1, 1)):
        method = "mvr"
    else:
        if p > 1000:
            method = "asdp"
        else:
            method = "sdp"
    return method

def compute_S_matrix(
    Sigma,
    groups=None,
    method=None,
    solver='cd',
    **kwargs
):
    """
    Wraps a variety of S-matrix generation
    functions.
    :param Sigma: covariance matrix
    :param groups: groups for group knockoffs
    :param method: Method for constructing
    S-matrix. One of mvr, maxent, sdp, asdp,
    equicorrelated.
    :param solver: Method for solving mrc knockoffs.
    One of 'cd' (coordinate descent) or 'psgd'
    (projected gradient descent).
    :param **kwargs: kwargs to one of the downstream
    functions.
    """
    # If S in kwargs, just return S (important
    # for chaining methods in metro sampling)
    kwargs = kwargs.copy()
    if 'S' in kwargs:
        if kwargs['S'] is not None:
            return kwargs['S']
        else:
            kwargs.pop('S')

    # Initial params
    p = Sigma.shape[0]
    if method is not None:
        method = str(method).lower()
    method = parse_method(method, groups, p)

    # Scale to correlation matrix
    scale = np.sqrt(np.diag(Sigma))
    scale_matrix = np.outer(scale, scale)
    Sigma = Sigma / scale_matrix

    # Currently cd solvers cannot do group knockoffs
    # (this is todo)
    if groups is not None:
        if not np.all(groups == np.arange(1, p + 1, 1)):
            solver = 'psgd'
    if (method == 'mvr' or method == 'maxent') and solver == 'psgd':
        S = mrc.solve_mrc_psgd(
            Sigma=Sigma, groups=groups, **kwargs
        )
    elif method == 'mvr':
        S = mrc.solve_mvr(
            Sigma=Sigma, **kwargs
        )
    elif method == 'maxent':
        S = mrc.solve_maxent(
            Sigma=Sigma, **kwargs
        )
    elif method == "sdp":
        S = solve_group_SDP(
            Sigma,
            groups,
            **kwargs,
        )
    elif method == "asdp":
        S = solve_group_ASDP(
            Sigma,
            groups,
            **kwargs,
        )
    elif method == "equicorrelated" or method == 'eq':
        S = equicorrelated_block_matrix(
            Sigma, groups, **kwargs
        )
    else:
        raise ValueError(f"Unrecognized method {method}")

    # Rescale and return
    return S * scale_matrix

def gaussian_knockoffs(
    X,
    fixedX=False,
    mu=None,
    Sigma=None,
    groups=None,
    invSigma=None,
    copies=1,
    sample_tol=1e-5,
    S=None,
    init_S=None,
    method=None,
    return_S=False,
    verbose=False,
    **kwargs,
):
    """ Constructs group Gaussian MX knockoffs.
    :param X: numpy array of dimension n x p 
    :param fixedX: If true, calculate fixedX knockoffs. Otherwise
    compute model-X (MX) knockoffs.
    :param mu: true mean of X, of dimension p.
    :param Sigma: true covariance matrix of X, of dimension p x p
    :param groups: numpy array of length p, list of groups of X
    :param copies: integer number of knockoff copies of each observation to draw
    :param S: the S matrix defined s.t. Cov(X, tilde(X)) = Sigma - S. Defaults to None
    and will be constructed by knockoff generator.
    :param init_S: An initial guess for the S matrix. This is helpful when using
    the mvr method (defined below).
    :param method: How to construct S matrix. There are several options:
        - 'equicorrelated': In this construction, the correlation between 
        each feature and its knockoff is the same (gamma). Minimizes this 
        gamma while preserving validity. See Dai and Barber 2016 for the
        group equicorrelated construction.
        - 'sdp': solves a convex semidefinite program to minimize the 
        (absolute) correlation between the features and their knockoffs
        while keeping knockoffs valid.
        - 'asdp': Same as SDP, but to increase speed, approximates correlation
        matrix as a block-diagonal matrix.
        - 'mvr': minimizes trace of feature-knockoff precision matrix. Solves this
        problem with projected gradient/coord descent, using either SDP or ASDP 
        to initialize. 
    The default is to use mvr for non-group knockoffs, and to use the group-SDP
    for grouped knockoffs. In both cases we use the ASDP if p > 1000.
    :param objective: How to optimize the S matrix if using SDP.
    There are several options:
        - 'abs': minimize sum(abs(Sigma - S))
        between groups and the group knockoffs.
        - 'pnorm': minimize Lp-th matrix norm.
        Equivalent to abs when p = 1.
        - 'norm': minimize different type of matrix norm
        (see norm_type below).
    :param sample_tol: Minimum eigenvalue allowed for cov matrix of knockoffs.
    Keep this extremely small (1e-5): it's just to prevent linalg errors downstream.
    :param bool verbose: If true, will print stuff as it goes
    :param rec_prop: The proportion of knockoffs you are planning to recycle
    (see Barber and Candes 2018, https://arxiv.org/abs/1602.03574). If 
    method = 'mvr',then the method takes this into account and should 
    dramatically increase the power of recycled knockoffs, especially in
    sparsely-correlated, high-dimensional settings.
    :param max_epochs: number of epochs (gradient steps) for mvr solver.
    :param smoothing: Smoothing parameter for mvr solver.
    :param kwargs: other kwargs for either equicorrelated/SDP/ASDP solvers.

    returns: n x p x copies numpy array of knockoffs"""

    # I follow the notation of Katsevich et al. 2019
    n = X.shape[0]
    p = X.shape[1]
    if groups is None:
        groups = np.arange(1, p + 1, 1)
    if groups.shape[0] != p:
        raise ValueError(
            f"Groups dimension ({groups.shape[0]}) and data dimension ({p}) do not match"
        )

    # For FX knockoffs, check dimensionality and also scale X
    if fixedX:
        if n < 2*p:
            raise np.linalg.LinAlgError(
                f"FX knockoffs can't be generated with n ({n}) < 2p ({2*p})"
            )

    # Scale X and (possibly) infer covariance matrix
    if fixedX:
        scale = np.sqrt(np.diag(np.dot(X.T, X)))
        X = X / scale.reshape(1, -1)
        Sigma = np.dot(X.T, X)
        if init_S is not None:
            init_S = init_S / np.outer(scale, scale)
        if S is not None:
            S = S / np.outer(scale, scale)

    # Possibly estimate mu, Sigma for MX 
    if Sigma is None and not fixedX: 
        Sigma, invSigma = utilities.estimate_covariance(X, tol=1e-2)

    # Scale for MX case (no shifting required)
    if not fixedX:
        # Scale X
        scale = np.sqrt(np.diag(Sigma))
        X = X / scale.reshape(1, -1)

        # Scale Sigma / invSigma 
        scale_matrix = np.outer(scale, scale)
        Sigma = Sigma / scale_matrix
        if invSigma is not None:
            invSigma = invSigma * scale_matrix

        if init_S is not None:
            init_S = init_S / scale_matrix
        if S is not None:
            S = S / scale_matrix

    # Infer (and adjust) mu parameter
    if mu is None:
        mu = X.mean(axis=0)
    else:
        mu = mu / scale

    # Get precision matrix
    if invSigma is None:
        invSigma = utilities.chol2inv(Sigma)
    else:
        product = np.dot(Sigma, invSigma)
        max_error = np.abs(product - np.eye(p)).max()
        if max_error > sample_tol:
            raise ValueError(
                f"invSigma provided was not the inverse of Sigma (max_error {max_error} > sample_tol {sample_tol})"
            )

    # Calculate group-block diagonal matrix S
    # using SDP, equicorrelated, or (maybe) ASDP
    if S is None:
        S = compute_S_matrix(
            Sigma=Sigma, groups=groups, method=method, **kwargs
        )

    # Check to make sure the methods worked
    min_eig1 = np.linalg.eigh(2 * Sigma - S)[0].min()
    if verbose:
        print(f"Minimum eigenvalue of 2Sigma - S is {min_eig1}")
        print(f"Minimum eigenvalue of S is {np.linalg.eigh(S)[0].min()}")
    if min_eig1 < -1e-6:
        raise np.linalg.LinAlgError(
            f"Minimum eigenvalue of 2Sigma - S is {min_eig1}, meaning FDR control violations are extremely likely"
        )

    if not fixedX:
        knockoffs = produce_MX_gaussian_knockoffs(
            X=X, 
            mu=mu,
            invSigma=invSigma,
            S=S,
            sample_tol=sample_tol,
            copies=copies,
            verbose=verbose
        )
        # Scale back to original dist
        scale = scale.reshape(1, -1, 1)
        knockoffs = scale*knockoffs
    else:
        knockoffs = produce_FX_gaussian_knockoffs(
            X=X,
            invSigma=invSigma,
            S=S,
            copies=copies,
            scale=scale
        )

    # For caching/debugging
    if return_S:
        # Rescale S for original cov matrix
        S = S * np.outer(scale, scale)
        return knockoffs, S

    return knockoffs

def produce_FX_gaussian_knockoffs(X, invSigma, S, scale, copies=1):
    """
    See equation (1.4) of https://arxiv.org/pdf/1404.5609.pdf
    """

    # Calculate C matrix
    n, p = X.shape
    #invSigma_S = np.dot(invSigma, S)
    CTC = 2*S - np.dot(S, np.dot(invSigma, S))
    C = scipy.linalg.cholesky(CTC)

    # Calculate U matrix
    Q,_ = scipy.linalg.qr(
        np.concatenate([X, np.zeros((n,p))], axis=1)
    )
    U = Q[:,p:2*p]

    # Randomize if copies > 1
    knockoff_base = np.dot(X, np.eye(p) - np.dot(invSigma, S))
    if copies > 1:
        knockoffs = []
        for j in range(copies):

            # Multiply U by random orthonormal matrix
            Qj, _ = scipy.linalg.qr(np.random.randn(p,p))
            Uj = np.dot(U, Qj)

            # Calculate knockoffs
            knockoff_j = knockoff_base + np.dot(Uj, C)
            knockoffs.append(knockoff_j * scale)
    else:
        # Calculate knockoffs and return
        knockoffs = [(knockoff_base + np.dot(U, C)) * scale]

    knockoffs = np.stack(knockoffs, axis=-1)
    return knockoffs


def produce_MX_gaussian_knockoffs(X, mu, invSigma, S, sample_tol, copies, verbose):

   # Calculate MX knockoff moments...
    n, p = X.shape
    invSigma_S = np.dot(invSigma, S)
    mu_k = X - np.dot(X - mu, invSigma_S)  # This is a bottleneck??
    Vk = 2 * S - np.dot(S, invSigma_S)

    # Account for numerical errors
    min_eig = np.linalg.eigh(Vk)[0].min()
    if min_eig < sample_tol and verbose:
        warnings.warn(
            f"Minimum eigenvalue of Vk is {min_eig}, under tolerance {sample_tol}"
        )
        Vk = shift_until_PSD(Vk, sample_tol)

    # ...and sample MX knockoffs!
    knockoffs = stats.multivariate_normal.rvs(mean=np.zeros(p), cov=Vk, size=copies * n)

    # Account for case when n * copies == 1
    if n * copies == 1:
        knockoffs = knockoffs.reshape(-1, 1)

    # (Save this for testing later)
    first_row = knockoffs[0, 0:n].copy()

    # Some annoying reshaping...
    knockoffs = knockoffs.flatten(order="C")
    knockoffs = knockoffs.reshape(p, n, copies, order="F")
    knockoffs = np.transpose(knockoffs, [1, 0, 2])

    # (Test we have reshaped correctly)
    new_first_row = knockoffs[0, 0:n, 0]
    np.testing.assert_array_almost_equal(
        first_row,
        new_first_row,
        err_msg="Critical error - reshaping failed in knockoff generator",
    )

    # Add mu
    mu_k = np.expand_dims(mu_k, axis=2)
    knockoffs = knockoffs + mu_k
    return knockoffs