import warnings
import numpy as np
from sklearn import linear_model
from group_lasso import GroupLasso, LogisticGroupLasso
from pyglmnet import GLMCV

from .utilities import calc_group_sizes, random_permutation_inds

DEFAULT_REG_VALS = np.logspace(-4, 4, base=10, num=20)


def calc_mse(model, X, y):
    """ Gets MSE of a model """
    preds = model.predict(X)
    resids = (preds - y) / y.std()
    return np.sum((resids) ** 2)


def use_reg_lasso(groups):
    """ Parses whether or not to use group lasso """
    # See if we are using regular lasso...
    if groups is not None:
        p = groups.shape[0]
        m = np.unique(groups).shape[0]
        if p == m:
            return True
        else:
            return False
    else:
        return True

def parse_y_dist(y):
    n = y.shape[0]
    if np.unique(y).shape[0] == 2:
        return 'binomial'
    elif np.unique(y).shape[0] == n:
        return 'gaussian'
    else:
        raise ValueError(
            "Please supply 'y_dist' arg (type of GLM to fit), e.g. gaussian, binomial"
        )

def parse_logistic_flag(kwargs):
    """ Checks whether y_dist is binomial """
    if "y_dist" in kwargs:
        if kwargs["y_dist"] == "binomial":
            return True
    return False


def combine_Z_stats(Z, groups, pair_agg="cd", group_agg="sum"):
    """
    Given a "Z" statistic for each feature AND each knockoff, returns
    grouped W statistics. First combines each Z statistic and its 
    knockoff, then aggregates this by group into group W statistics.
    :param Z: p length numpy array of Z statistics. The first p
    values correspond to true features, and the last p correspond
    to knockoffs (in the same order as the true features).
    :param groups: p length numpy array of groups. 
    :param str pair_agg: Specifies how to create pairwise W 
    statistics. Two options: 
        - "CD" (Difference of absolute vals of coefficients),
        - "SM" (signed maximum).
        - "SCD" (Simple difference of coefficients - NOT recommended)
    :param str group_agg: Specifies how to combine pairwise W
    statistics into grouped W statistics. Two options: "sum" (default)
    and "avg".
    """

    # Step 1: Pairwise W statistics.
    p = int(Z.shape[0] / 2)
    if Z.shape[0] != 2 * p:
        raise ValueError(
            f"Unexpected shape {Z.shape} for Z statistics (expected ({2*p},))"
        )
    if groups is None:
        groups = np.arange(1, p + 1, 1)

    pair_agg = str(pair_agg).lower()
    # Absolute coefficient differences
    if pair_agg == "cd":
        pair_W = np.abs(Z[0:p]) - np.abs(Z[p:])
    # Signed maxes
    elif pair_agg == "sm":
        inds = np.arange(0, p, 1)
        pair_W = np.maximum(np.abs(Z[inds]), np.abs(Z[inds + p]))
        pair_W = pair_W * np.sign(np.abs(Z[inds]) - np.abs(Z[inds + p]))
    # Simple coefficient differences
    elif pair_agg == 'scd':
        pair_W = Z[0:p] - Z[p:]
    else:
        raise ValueError(f'pair_agg ({pair_agg}) must be one of "cd", "sm", "scd"')

    # Step 2: Group statistics
    m = np.unique(groups).shape[0]
    W_group = np.zeros(m)
    for j in range(p):
        W_group[groups[j] - 1] += pair_W[j]

    # If averaging...
    if group_agg == "sum":
        pass
    elif group_agg == "avg":
        group_sizes = calc_group_sizes(groups)
        W_group = W_group / group_sizes
    else:
        raise ValueError(f'group_agg ({group_agg}) must be one of "sum", "avg"')

    # Return
    return W_group


# ------------------------------ Lasso Stuff ---------------------------------------#


def calc_lars_path(X, knockoffs, y, groups=None, **kwargs):
    """ Calculates locations at which X/knockoffs enter lasso 
    model when regressed on y.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :param kwargs: kwargs for sklearn Lasso class 
     """

    # Ignore y_dist kwargs (residual)
    if 'y_dist' in kwargs:
        kwargs.pop('y_dist')

    # Bind data
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis=1)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = random_permutation_inds(2 * p)
    features = features[:, inds]

    # By default, all variables are their own group
    if groups is None:
        groups = np.arange(0, p, 1)

    # Fit
    alphas, _, coefs = linear_model.lars_path(features, y, method="lasso", **kwargs,)

    # Calculate places where features enter the model
    Z = np.zeros(2 * p)
    for i in range(2 * p):
        if (coefs[i] != 0).sum() == 0:
            Z[i] = 0
        else:
            Z[i] = alphas[np.where(coefs[i] != 0)[0][0]]

    return Z[rev_inds]


def fit_lasso(
    X, 
    knockoffs, 
    y, 
    y_dist=None, 
    use_lars=False,
    **kwargs
):

    # Parse some kwargs/defaults
    if "max_iter" in kwargs:
        max_iter = kwargs["max_iter"]
        kwargs.pop("max_iter")
    else:
        max_iter = 500
    if "tol" in kwargs:
        tol = kwargs["tol"]
        kwargs.pop("tol")
    else:
        tol = 1e-3
    if "cv" in kwargs:
        cv = kwargs["cv"]
        kwargs.pop("cv")
    else:
        cv = 5
    if y_dist is None:
        y_dist = parse_y_dist(y)

    # Bind data
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis=1)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = random_permutation_inds(2 * p)
    features = features[:, inds]

    # Fit lasso
    warnings.filterwarnings("ignore")
    if y_dist == "gaussian":
        if not use_lars:
            gl = linear_model.LassoCV(
                alphas=DEFAULT_REG_VALS,
                cv=cv,
                verbose=False,
                max_iter=max_iter,
                tol=tol,
                **kwargs,
            ).fit(features, y)
        elif use_lars:
            gl = linear_model.LassoLarsCV(
                cv=cv,
                verbose=False,
                max_iter=max_iter,
                **kwargs,
            ).fit(features, y)
    elif y_dist == "binomial":
        gl = linear_model.LogisticRegressionCV(
            Cs=1 / DEFAULT_REG_VALS,
            penalty="l1",
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=False,
            solver="liblinear",
            **kwargs,
        ).fit(features, y)
    else:
        raise ValueError(f"y_dist must be one of gaussian, binomial, not {y_dist}")
    warnings.resetwarnings()

    return gl, rev_inds


def fit_group_lasso(
    X,
    knockoffs,
    y,
    groups,
    use_pyglm=True,
    y_dist=None,
    group_lasso=True,
    **kwargs,
):
    """ Fits a group lasso model.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :param use_pyglm: If true, use the pyglmnet grouplasso
    Else use the regular one
    :param y_dist: Either "gaussian" or "binomial" (for logistic regression)
    :param group_lasso: If False, do not use group regularization.
    :param kwargs: kwargs for group-lasso GroupLasso class.
    In particular includes reg_vals, a list of regularizations
    (lambda values) which defaults to [(0.05, 0.05)]. In each
    tuple of the list, the first value is the group regularization,
    the second value is the individual regularization.
    """

    warnings.filterwarnings("ignore")

    # Parse some kwargs/defaults
    if "max_iter" in kwargs:
        max_iter = kwargs["max_iter"]
        kwargs.pop("max_iter")
    else:
        max_iter = 100
    if "tol" in kwargs:
        tol = kwargs["tol"]
        kwargs.pop("tol")
    else:
        tol = 1e-2
    if "cv" in kwargs:
        cv = kwargs["cv"]
        kwargs.pop("cv")
    else:
        cv = 5
    if "learning_rate" in kwargs:
        learning_rate = kwargs["learning_rate"]
        kwargs.pop("learning_rate")
    else:
        learning_rate = 2
    if y_dist is None:
        y_dist = parse_y_dist(y)

    # Bind data
    n = X.shape[0]
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis=1)

    # By default, all variables are their own group
    if groups is None:
        groups = np.arange(1, p + 1, 1)
        m = p
    else:
        m = np.unique(groups).shape[0]

    # If m == p, meaning each variable is their own group,
    # just fit a regular lasso
    if m == p or not group_lasso:
        return fit_lasso(X, knockoffs, y, y_dist, **kwargs)

    # Make sure variables and their knockoffs are in the same group
    # This is necessary for antisymmetry
    doubled_groups = np.concatenate([groups, groups], axis=0)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = random_permutation_inds(2 * p)
    features = features[:, inds]
    doubled_groups = doubled_groups[inds]

    # Standardize - important for pyglmnet performance,
    # highly detrimental for group_lasso performance
    if use_pyglm:
        features = (features - features.mean()) / features.std()
        if y_dist == "gaussian":
            y = (y - y.mean()) / y.std()

    # Get regularizations
    if "reg_vals" in kwargs:
        reg_vals = kwargs["reg_vals"]
        kwargs.pop("reg_vals")
    else:
        reg_vals = [(x, x) for x in DEFAULT_REG_VALS]

    # Fit pyglm model using warm starts
    if use_pyglm:

        l1_regs = [x[0] for x in reg_vals]

        gl = GLMCV(
            distr=y_dist,
            tol=tol,
            group=doubled_groups,
            alpha=1.0,
            learning_rate=learning_rate,
            max_iter=max_iter,
            reg_lambda=l1_regs,
            cv=cv,
            solver="cdfast",
        )
        gl.fit(features, y)

        # Pull score, rename
        best_score = -1 * calc_mse(gl, features, y)
        best_gl = gl

    # Fit model
    if not use_pyglm:
        best_gl = None
        best_score = -1 * np.inf
        for group_reg, l1_reg in reg_vals:

            # Fit logistic/gaussian group lasso
            if not use_pyglm:
                if y_dist.lower() == "gaussian":
                    gl = GroupLasso(
                        groups=doubled_groups,
                        tol=tol,
                        group_reg=group_reg,
                        l1_reg=l1_reg,
                        **kwargs,
                    )
                elif y_dist.lower() == "binomial":
                    gl = LogisticGroupLasso(
                        groups=doubled_groups,
                        tol=tol,
                        group_reg=group_reg,
                        l1_reg=l1_reg,
                        **kwargs,
                    )
                else:
                    raise ValueError(
                        f"y_dist must be one of gaussian, binomial, not {y_dist}"
                    )

                gl.fit(features, y.reshape(n, 1))
                score = -1 * calc_mse(gl, features, y.reshape(n, 1))

            # Score, possibly select
            if score > best_score:
                best_score = score
                best_gl = gl

    warnings.resetwarnings()

    return best_gl, rev_inds


def lasso_statistic(
    X,
    knockoffs,
    y,
    groups=None,
    zstat="coef",
    use_pyglm=True,
    group_lasso=False,
    pair_agg="cd",
    group_agg="avg",
    return_Z=False,
    **kwargs,
):
    """
    Calculates group lasso statistics in one of several ways.
    The procedure is as follows:
        - First, uses the lasso to calculate a "Z" statistic 
        for each feature AND each knockoff.
        - Second, calculates a "W" statistic pairwise 
        between each feature and its knockoff.
        - Third, sums or averages the "W" statistics for each
        group to obtain group W statistics.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param y: p length response numpy array
    :param groups: p length numpy array of groups. If None,
    defaults to giving each feature its own group.
    :param zstat: Two options:
        - If 'coef', uses to cross-validated (group) lasso coefficients.
        - If 'lars_path', uses the lambda value where each feature/knockoff
        enters the lasso path (meaning becomes nonzero).
    This defaults to coef.
    :param use_pyglm: If true, use the pyglmnet grouplasso.
    Else use the group-lasso one.
    :param bool group_lasso: If True and zstat='coef', then runs
    group lasso. Defaults to False (recommended). 
    :param str pair_agg: Specifies how to create pairwise W 
    statistics. Two options: 
        - "CD" (Difference of absolute vals of coefficients),
        - "SM" (signed maximum).
        - "SCD" (Simple difference of coefficients - NOT recommended)
    :param str group_agg: Specifies how to combine pairwise W
    statistics into grouped W statistics. Two options: "sum" (default)
    and "avg".
    :param kwargs: kwargs to lasso or lars_path solver. 
    """

    # Possibly set default groups
    n = X.shape[0]
    p = X.shape[1]
    if groups is None:
        groups = np.arange(1, p + 1, 1)

    # Check if y_dist is gaussian, binomial, poisson
    if 'y_dist' not in kwargs:
        kwargs['y_dist'] = parse_y_dist(y)
    elif kwargs['y_dist'] is None:
        kwargs['y_dist'] = parse_y_dist(y)

    # Step 1: Calculate Z statistics
    zstat = str(zstat).lower()
    if zstat == "coef":

        # Fit (possibly group) lasso
        gl, rev_inds = fit_group_lasso(
            X,
            knockoffs,
            y,
            groups=groups,
            use_pyglm=use_pyglm,
            group_lasso=group_lasso,
            **kwargs,
        )

        # Parse the expected output format based on which
        # lasso package we are using
        reg_lasso_flag = use_reg_lasso(groups) or (not group_lasso)
        logistic_flag = parse_logistic_flag(kwargs)

        # Retrieve Z statistics
        if use_pyglm and not reg_lasso_flag:
            Z = gl.beta_[rev_inds]
        elif reg_lasso_flag and logistic_flag:
            if gl.coef_.shape[0] != 1:
                raise ValueError(
                    "Unexpected shape for logistic lasso coefficients (sklearn)"
                )
            Z = gl.coef_[0, rev_inds]
        else:
            Z = gl.coef_[rev_inds]

    elif zstat == "lars_path":
        Z = calc_lars_path(X, knockoffs, y, groups, **kwargs)

    else:
        raise ValueError(f'zstat ({zstat}) must be one of "coef", "lars_path"')

    # Combine Z statistics
    W_group = combine_Z_stats(Z, groups, pair_agg=pair_agg, group_agg=group_agg)
    # Possibly return both Z and W
    if return_Z:
        return W_group, Z    
    return W_group


def marg_corr_diff(
    X,
    knockoffs,
    y,
    groups=None,
    return_Z=False,
    **kwargs,
):
    """
    Marginal correlations used as Z statistics. 
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param y: p length response numpy array
    :param groups: p length numpy array of groups. If None,
    defaults to giving each feature its own group.
    :param **kwargs: kwargs to combine_Z_stats
    """

    # Calc correlations
    features = np.concatenate([X, knockoffs], axis=1)
    correlations = np.corrcoef(features, y.reshape(-1, 1), rowvar=False)[-1][0:-1]

    # Combine
    W = combine_Z_stats(correlations, groups, **kwargs)
    # Possibly return both Z and W
    if return_Z:
        return W, correlations
    return W


def linear_coef_diff(
    X, 
    knockoffs,
    y,
    groups=None,
    return_Z=False,
    **kwargs
):
    """
    Linear regression coefficients used as Z statistics.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param y: p length response numpy array
    :param groups: p length numpy array of groups. If None,
    defaults to giving each feature its own group.
    :param **kwargs: kwargs to combine_Z_stats
    """

    # Run linear regression, permute indexes to prevent FDR violations
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis=1)
    inds, rev_inds = random_permutation_inds(2 * p)
    features = features[:, inds]

    lm = linear_model.LinearRegression(fit_intercept=False).fit(features, y)
    Z = lm.coef_

    # Play with shape, take abs
    Z = np.abs(Z.reshape(-1))
    if Z.shape[0] != 2 * p:
        raise ValueError(
            f"Unexpected shape {Z.shape} for sklearn LinearRegression coefs (expected ({2*p},))"
        )

    # Undo random permutation
    Z = Z[rev_inds]

    # Combine with groups to create W-statistic
    W = combine_Z_stats(Z, groups, **kwargs)
    # Possibly return both Z and W
    if return_Z:
        return W, Z
    return W


def data_dependent_threshhold(W, fdr=0.10, offset=1):
    """
    Follows https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R
    :param W: p-length numpy array of feature statistics OR p x batch-length numpy array
    of feature stats. If batched, the last dimension must be the batch dimension.
    :param fdr: desired FDR level (referred to as q in the literature)
    :param offset: if offset = 0, use knockoffs (which control modified FDR).
    Else, if offset = 1, use knockoff+ (controls exact FDR).
    """

    # Add dummy batch axis if necessary
    if len(W.shape) == 1:
        W = W.reshape(-1, 1)
    p = W.shape[0]
    batch = W.shape[1]

    # Sort W by absolute values
    ind = np.argsort(-1 * np.abs(W), axis=0)
    sorted_W = np.take_along_axis(W, ind, axis=0)

    # Calculate ratios
    negatives = np.cumsum(sorted_W <= 0, axis=0)
    positives = np.cumsum(sorted_W > 0, axis=0)
    positives[positives == 0] = 1  # Don't divide by 0
    ratios = (negatives + offset) / positives

    # Add zero as an option to prevent index errors
    # (zero means select everything strictly > 0)
    sorted_W = np.concatenate([sorted_W, np.zeros((1, batch))], axis=0)

    # Find maximum indexes satisfying FDR control
    # Adding np.arange is just a batching trick
    helper = (ratios <= fdr) + np.arange(0, p, 1).reshape(-1, 1) / p
    sorted_W[1:][helper < 1] = np.inf  # Never select values where the ratio > fdr
    T_inds = np.argmax(helper, axis=0) + 1
    more_inds = np.indices(T_inds.shape)

    # Find Ts
    acceptable = np.abs(sorted_W)[T_inds, more_inds][0]

    # Replace 0s with a very small value to ensure that
    # downstream you don't select W statistics == 0.
    # This value is the smallest abs value of nonzero W
    if np.sum(acceptable == 0) != 0:
        W_new = W.copy() 
        W_new[W_new==0] = np.abs(W).max()
        zero_replacement = np.abs(W_new).min(axis=0)
        acceptable[acceptable==0] = zero_replacement[acceptable==0]

    if batch == 1:
        acceptable = acceptable[0]

    return acceptable


def calc_epowers(Ws, group_sizes, non_nulls=None, **kwargs):
    """
    :param Ws: b x p dimension np array. 
    :param group_sizes: b x p dimension np array.
    (May be zero-padded).
    :param non_nulls: b x p dimension boolean np array.
    True indicates non-nulls. Used only to normalize powers.
    :param kwargs: kwargs to data_dependent_threshhold 
    (fdr and offset).
    returns: batch length array of empirical powers.
    """

    # Find discoveries
    b = Ws.shape[0]
    p = Ws.shape[1]
    Ts = data_dependent_threshhold(Ws.T, **kwargs)
    discoveries = Ws >= Ts.reshape(b, 1)

    # Weight by inverse group sizes
    group_sizes[group_sizes == 0] = -1
    inv_group_sizes = 1 / group_sizes
    inv_group_sizes[inv_group_sizes < 0] = 0

    # Multiply
    epowers = (discoveries * inv_group_sizes).sum(axis=1)

    # Possibly normalize
    if non_nulls is not None:
        num_non_nulls = non_nulls.sum(axis=1)
        num_non_nulls[num_non_nulls == 0] = p
        epowers = epowers / num_non_nulls

    return epowers
