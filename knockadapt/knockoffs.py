import warnings
import numpy as np
import cvxpy as cp
import scipy as sp
from scipy import stats

from .utilities import chol2inv, calc_group_sizes, preprocess_groups

# Multiprocessing tools
from functools import partial
from multiprocessing import Pool

# Options for SDP solver
OBJECTIVE_OPTIONS = ["abs", "pnorm", "norm"]


def TestIfCorrMatrix(Sigma):
    """ Tests if a square matrix is a correlation matrix """
    p = Sigma.shape[0]
    diag = np.diag(Sigma)
    if np.sum(np.abs(diag - np.ones(p))) > p * 1e-3:
        raise ValueError("Sigma is not a correlation matrix. Scale it properly first.")


def permute_matrix_by_groups(groups):
    """
    Permute a (correlation) matrix according to a list of feature groups.
    :param groups: a p-length array of integers.
    returns: inds and inv_inds
    Given a p x p matrix M, Mp = M[inds][:, inds] permutes the matrix according to the group.
    Then, Mp[inv_inds][:, inv_inds] unpermutes the matrix back to its original form. 
    """
    # Create sorting indices
    inds_and_groups = [(i, group) for i, group in enumerate(groups)]
    inds_and_groups = sorted(inds_and_groups, key=lambda x: x[1])
    inds = [i for (i, j) in inds_and_groups]

    # Make sure we can unsort
    p = groups.shape[0]
    inv_inds = np.zeros(p)
    for i, j in enumerate(inds):
        inv_inds[j] = i
    inv_inds = inv_inds.astype("int32")

    return inds, inv_inds


def shift_until_PSD(M, tol):
    """ Add the identity until a p x p matrix M has eigenvalues of at least tol"""
    p = M.shape[0]
    mineig = np.linalg.eigh(M)[0].min()
    if mineig < tol:
        M += (tol - mineig) * np.eye(p)

    return M


def scale_until_PSD(Sigma, S, tol, num_iter):
    """ Takes a PSD matrix S and performs a binary search to 
    find the largest gamma such that 2*V - gamma*S is PSD as well."""

    # Raise value error if S is not PSD
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = shift_until_PSD(S, tol)

    # Binary search to find minimum gamma
    lower_bound = 0
    upper_bound = 1
    for j in range(num_iter):
        gamma = (lower_bound + upper_bound) / 2
        mineig = np.linalg.eigh(2 * Sigma - gamma * S)[0].min()
        if mineig < tol:
            upper_bound = gamma
        else:
            lower_bound = gamma

    # Scale S properly, be a bit conservative
    S = lower_bound * S

    return S, gamma


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
        inv_group_sigma = chol2inv(group_sigma)
        sqrt_inv_group_sigma = sp.linalg.sqrtm(inv_group_sigma)

        # Fill in D
        D[full_inds] = sqrt_inv_group_sigma

    # Test to make sure this is positive definite
    min_d_eig = np.linalg.eigh(D)[0].min()
    if min_d_eig < -1 * tol:
        raise ValueError(f"Minimum eigenvalue of block matrix D is {min_d_eig}")

    # Find minimum eigenvalue
    DSigD = np.einsum("pk,kj,jl->pl", D, Sigma, D)
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


def solve_group_SDP(
    Sigma,
    groups,
    sdp_verbose=False,
    objective="pnorm",
    norm_type=2,
    num_iter=10,
    tol=1e-2,
    **kwargs,
):
    """ Solves the group SDP problem: extends the
    formulation from Barber and Candes 2015/
    Candes et al 2018 (MX Knockoffs)"
    :param Sigma: true covariance (correlation) matrix, 
    p by p numpy array.
    :param groups: numpy array of length p with
    integer values between 1 and m. 
    :param verbose: if True, print progress of solver
    :param objective: How to optimize the S matrix. 
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
    if tol > maxtol and sdp_verbose:
        warnings.warn(
            f"Reducing SDP tol from {tol} to {maxtol}, otherwise SDP would be infeasible"
        )
    tol = min(maxtol, tol)

    # Figure out sizes of groups
    p = Sigma.shape[0]
    m = groups.max()
    group_sizes = np.zeros(m)
    for j in groups:
        group_sizes[j - 1] += 1

    # Sort the covariance matrix according to the groups
    inds, inv_inds = permute_matrix_by_groups(groups)
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
    G = cp.bmat([[sortedSigma, sortedSigma - S], [sortedSigma - S, sortedSigma]])
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
    problem.solve(verbose=sdp_verbose, **kwargs)
    if sdp_verbose:
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
    if sdp_verbose:
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

    # Possibly automatically choose alpha
    if alpha is None and Sigma_groups is None:
        group_sizes = calc_group_sizes(groups)
        max_group = group_sizes.max()

        # If the max group size is small enough:
        if max_block >= max_group:
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
                Sigma=Sigma_blocks[i], groups=group_blocks[i], **kwargs
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


def group_gaussian_knockoffs(
    X,
    Sigma,
    groups,
    invSigma=None,
    copies=1,
    sample_tol=1e-5,
    sdp_tol=1e-2,
    S=None,
    method="sdp",
    objective="pnorm",
    return_S=False,
    verbose=True,
    sdp_verbose=True,
    **kwargs,
):
    """ Constructs group Gaussian MX knockoffs:
    This is not particularly efficient yet...
    :param X: numpy array of dimension n x p 
    :param Sigma: true covariance matrix of X, of dimension p x p
    :param method: 'equicorrelated' or 'sdp', how to construct
     the true Cov(X, tilde(X)), where tilde(X) is the knockoffs.
    :param groups: numpy array of length p, list of groups of X
    :param copies: integer number of knockoff copies of each observation to draw
    :param S: the S matrix defined s.t. Cov(X, tilde(X)) = Sigma - S. Defaults to None
    and will be constructed by knockoff generator.
    :param method: How to constructe S matrix, either 'sdp' or 'equicorrelated'
    :param objective: How to optimize the S matrix if using SDP.
    There are several options:
        - 'abs': minimize sum(abs(Sigma - S))
    :param sample_tol: Minimum eigenvalue allowed for cov matrix of knockoffs.
    Keep this extremely small (1e-5): it's just to prevent linalg errors downstream.
    :param sdp_tol: Minimum eigenvalue allowed for grahm matrix of knockoff 
    generations. This is used as a constraint in the SDP formulation. Keep this
    small but not inconsequential (1e-2) as if this gets too small, the 
    feature-knockoff combinations may become unidentifiable.
    :param bool verbose: If true, will print stuff as it goes
    :param bool sdp_verbose: If true, will tell the SDP solver to be verbose.
    :param kwargs: other kwargs for either equicorrelated/SDP/ASDP solvers.
    
    returns: copies x n x p numpy array of knockoffs"""

    # I follow the notation of Katsevich et al. 2019
    n = X.shape[0]
    p = X.shape[1]

    if groups.shape[0] != p:
        raise ValueError(
            f"Groups dimension ({groups.shape[0]}) and data dimension ({p}) do not match"
        )

    # Get precision matrix
    if invSigma is None:
        invSigma = chol2inv(Sigma)
    else:
        product = np.dot(invSigma.T, Sigma)
        if np.abs(product - np.eye(p)).sum() > sample_tol:
            raise ValueError(
                "Inverse Sigma provided was not actually the inverse of Sigma"
            )

    # Calculate group-block diagonal matrix S
    # using SDP, equicorrelated, or (maybe) ASDP
    method = str(method).lower()
    if S is None:
        if method == "sdp":
            if verbose:
                print(f"Solving SDP for S with p = {p}")
            S = solve_group_SDP(
                Sigma,
                groups,
                objective=objective,
                sdp_verbose=sdp_verbose,
                tol=sdp_tol,
                **kwargs,
            )
        elif method == "equicorrelated":
            S = equicorrelated_block_matrix(Sigma, groups, *kwargs)
        elif method == "asdp":
            S = solve_group_ASDP(
                Sigma,
                groups,
                objective=objective,
                verbose=verbose,
                sdp_verbose=sdp_verbose,
                **kwargs,
            )
        else:
            raise ValueError(
                f'Method must be one of "equicorrelated", "asdp", "sdp", not {method}'
            )
    else:
        pass

    # Check to make sure the methods worked
    min_eig1 = np.linalg.eigh(2 * Sigma - S)[0].min()
    if verbose:
        print(f"Minimum eigenvalue of 2Sigma - S is {min_eig1}")
    if min_eig1 < -1e-3:
        raise np.linalg.LinAlgError(
            f"Minimum eigenvalue of 2Sigma - S is {min_eig1}, meaning FDR control violations are extremely likely"
        )

    # Calculate MX knockoff moments...
    invSigma_S = np.dot(invSigma, S)
    mu = X - np.dot(X, invSigma_S)  # This is a bottleneck??
    # TODO: if we replace "X" inside the dot with "X - true mu"
    # then we can add a population mu parameter
    Vk = 2 * S - np.dot(S, invSigma_S)

    # Account for numerical errors
    min_eig = np.linalg.eigh(Vk)[0].min()
    if verbose:
        print(f"Minimum eigenvalue of Vk is {min_eig}")
    if min_eig < sample_tol:
        if verbose:
            warnings.warn(
                f"Minimum eigenvalue of Vk is {min_eig}, under tolerance {sample_tol}"
            )
        Vk = shift_until_PSD(Vk, sample_tol)

    # ...and sample MX knockoffs!
    knockoffs = stats.multivariate_normal.rvs(mean=np.zeros(p), cov=Vk, size=copies * n)

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
    mu = np.expand_dims(mu, axis=2)
    knockoffs = knockoffs + mu

    # For caching/debugging
    if return_S:
        return knockoffs, S

    return knockoffs
