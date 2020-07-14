import warnings
import numpy as np
import scipy as sp
import sklearn.covariance

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

def cov2corr(M):
    scale = np.sqrt(np.diag(M))
    return M / np.outer(scale, scale)

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
        V = 2 * Sigma - gamma * S
        mineig = np.linalg.eigh(V)[0].min()
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

### Helper for MX knockoffs when we infer Sigma, and also when
### X comes from the gibbs model
def estimate_covariance(X, tol=1e-4, shrinkage = 'ledoitwolf'):
    """ Estimates covariance matrix of X. 
    :param X: n x p data matrix
    :param tol: threshhold for minimum eigenvalue
    :shrinkage: The type of shrinkage to apply.
     One of "ledoitwolf," "graphicallasso," or 
     None (no shrinkage). Note even if shrinkage is None,
    if the minim eigenvalue of the empirical cov matrix
    is below a certain tolerance, this will apply shrinkage
    anyway.
    :returns: Sigma, invSigma
    """
    Sigma = np.cov(X.T)
    mineig = np.linalg.eigh(Sigma)[0].min()

    # Parse none strng
    if str(shrinkage).lower() == 'none':
        shrinkage = None

    # Possibly shrink Sigma
    if mineig < tol or shrinkage is not None:
        # Which shrinkage to use
        if str(shrinkage).lower() == 'ledoitwolf' or shrinkage is None: 
            ShrinkEst = sklearn.covariance.LedoitWolf()
        elif str(shrinkage).lower() == 'graphicallasso':
            ShrinkEst = sklearn.covariance.GraphicalLasso(alpha=0.1)
        else:
            raise ValueError(f"Shrinkage arg must be one of None, 'ledoitwolf', 'graphicallasso', not {shrinkage}")

        # Fit shrinkage. Sometimes the Graphical Lasso raises errors
        # so we handle these here.
        try:
            warnings.filterwarnings("ignore")
            ShrinkEst.fit(X)
            warnings.resetwarnings()
        except FloatingPointError:
            warnings.resetwarnings()
            warnings.warn(f"Graphical lasso failed, LedoitWolf matrix")
            ShrinkEst = sklearn.covariance.LedoitWolf()
            ShrinkEst.fit(X)

        # Return
        Sigma = ShrinkEst.covariance_
        invSigma = ShrinkEst.precision_
        return Sigma, invSigma

    # Else return empirical estimate
    return Sigma, chol2inv(Sigma)


def sparse_estimate_covariance(X, sparsity, alpha=0.1):
    """
    Estimates a covariance and precision matrix for a data matrix
    X given a known sparsity patter. To solve this problem,
    I have naively modified the algorith from the
    Friedman 2008 Biostatistics paper.

    :param X: n x p design matrix
    :param sparsity: known p x p sparsity pattern of the precision matrix
    Zeros indicate sparsity.
    :alpha: alpha for graphical lasso

    This code is adapted from the sklearn graphical
    lasso estimator. Sklearn is distributed under the 3-clause 
    BSD license. This requires me to copy the following below:

    Redistribution and use in source and binary forms, with or 
    without modification, are permitted provided that the following 
    conditions are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    if alpha == 0:
        raise ValueError(f"Not implemented for alpha == 0")

    # Default covariance
    _, n_features = X.shape
    covest = sklearn.covariance.EmpiricalCovariance()
    covest.fit(X)
    emp_cov = covest.covariance_

    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[::n_features + 1]
    covariance_.flat[::n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    # Initialize some helpers
    indices = np.arange(n_features)
    costs = list()
    errors = dict(over='raise', invalid='ignore')

    # Loop through GLASSO algorithm. I found
    # https://web.stanford.edu/~hastie/Papers/glassoinsights.pdf
    # to be helpful.
    d_gap = np.inf
    sub_covariance = np.copy(covariance_[1:, 1:], order='C')
    for i in range(max_iter):
        for idx in range(n_features):
            # To keep the contiguous matrix `sub_covariance` equal to
            # covariance_[indices != idx].T[indices != idx]
            # we only need to update 1 column and 1 line when idx changes
            if idx > 0:
                di = idx - 1
                sub_covariance[di] = covariance_[di][indices != idx]
                sub_covariance[:, di] = covariance_[:, di][indices != idx]
            else:
                sub_covariance[:] = covariance_[1:, 1:]
            row = emp_cov[idx, indices != idx]
            with np.errstate(**errors):
                # Use coordinate descent
                coefs = -(precision_[indices != idx, idx]
                          / (precision_[idx, idx] + 1000 * eps))
                coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                    coefs, alpha, 0, sub_covariance,
                    row, row, max_iter, enet_tol,
                    check_random_state(None), False)

            # Update the precision matrix
            precision_[idx, idx] = (
                1. / (covariance_[idx, idx]
                      - np.dot(covariance_[indices != idx, idx], coefs)))
            precision_[indices != idx, idx] = (- precision_[idx, idx]
                                               * coefs)
            precision_[idx, indices != idx] = (- precision_[idx, idx]
                                               * coefs)
            coefs = np.dot(sub_covariance, coefs)
            covariance_[idx, indices != idx] = coefs
            covariance_[indices != idx, idx] = coefs