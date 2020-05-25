import warnings
import numpy as np
import scipy as sp

### Group helpers


def preprocess_groups(groups):
    """ Turns a p-dimensional numpy array with m unique elements
    into a list of integers from 1 to m"""
    unique_vals = np.unique(groups)
    conversion = {unique_vals[i]: i for i in range(unique_vals.shape[0])}
    return np.array([conversion[x] + 1 for x in groups])


def fetch_group_nonnulls(non_nulls, groups):
    """ 
    :param non_nulls: a p-length array of coefficients where 
    a 0 indicates that a variable is null
    :param groups: a p-length array indicating group membership,
    with m groups (corresponding to ints ranging from 1 to m)
    :returns: a m-length array of 1s and 0s where 0s correspond
    to nulls.
    """

    if not isinstance(non_nulls, np.ndarray):
        non_nulls = np.array(non_nulls)
    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)

    # Initialize
    m = np.unique(groups).shape[0]
    group_nonnulls = np.zeros(m)

    # Calculate and return
    for j in range(m):
        flag = np.abs(non_nulls[groups == j + 1]).sum() > 0
        group_nonnulls[j] = float(flag)
    return group_nonnulls


def calc_group_sizes(groups):
    """
    :param groups: p-length array of integers between 1 and m, 
    where m <= p
    returns: m-length array of group sizes 
    """
    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)
    if np.all(groups.astype("int32") != groups):
        raise TypeError(
            "groups cannot contain non-integer values: apply preprocess_groups first"
        )
    else:
        groups = groups.astype("int32")

    if np.min(groups) == 0:
        raise ValueError(
            "groups cannot contain 0: add one or apply preprocess_groups first"
        )

    m = groups.max()
    group_sizes = np.zeros(m)
    for j in groups:
        group_sizes[j - 1] += 1
    group_sizes = group_sizes.astype("int32")
    return group_sizes


### Matrix helpers for S-matrix computation


def chol2inv(X):
    """ Uses cholesky decomp to get inverse of matrix """
    triang = np.linalg.inv(np.linalg.cholesky(X))
    return np.dot(triang.T, triang)


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


### Feature-statistic helpers


def random_permutation_inds(length):
    """ Returns indexes which correspond to a random permutation,
    as well as indexes which undo the permutation. Is truly random
    (calls np.random.seed()) but does not change random state."""

    # Save random state and change it
    st0 = np.random.get_state()
    # np.random.seed()

    # Create inds and rev inds
    inds = np.arange(0, length, 1)
    np.random.shuffle(inds)
    rev_inds = [0 for _ in range(length)]
    for (i, j) in enumerate(inds):
        rev_inds[j] = i

    # Reset random state and return
    np.random.set_state(st0)
    return inds, rev_inds
