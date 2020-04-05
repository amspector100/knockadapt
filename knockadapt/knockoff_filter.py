import numpy as np
from . import knockoff_stats
from .knockoffs import group_gaussian_knockoffs

def mx_knockoff_filter(
    X,
    y,
    Sigma,
    groups=None,
    knockoffs=None,
    feature_stat_fn='lasso',
    fdr=0.10,
    feature_stat_kwargs={},
    knockoff_kwargs={"sdp_verbose": False},
    recycle_up_to=None,
):
    """
    :param X: n x p design matrix
    :param y: p-length response array
    :param Sigma: p x p covariance matrix of X
    :param groups: Grouping of features, p-length
    array of integers from 1 to m (with m <= p).
    :param knockoffs: n x p array of knockoffs.
    If None, will construct second-order group MX knockoffs.
    Defaults to group gaussian knockoff constructor.
    :param feature_stat: Function used to
    calculate W-statistics in knockoffs. 
    Defaults to group lasso coefficient difference.
    :param fdr: Desired fdr.
    :param feature_stat_fn: A function which takes X,
    knockoffs, y, and groups, and returns a set of 
    p anti-symmetric knockoff statistics. Can also
    be one of "lasso", "ols", or "margcorr." 
    :param feature_stat_kwargs: Kwargs to pass to 
    the feature statistic.
    :param knockoff_kwargs: Kwargs to pass to the 
    knockoffs constructor.
    :param recycle_up_to: Three options:
        - if None, does nothing.
        - if an integer > 1, uses the first "recycle_up_to"
        rows of X as the the first "recycle_up_to" rows of knockoffs.
        - if a float between 0 and 1 (inclusive), interpreted
        as the proportion of knockoffs to recycle. 
    For more on recycling, see https://arxiv.org/abs/1602.03574
    """

    # Preliminaries
    n = X.shape[0]
    p = Sigma.shape[0]
    if groups is None:
        groups = np.arange(1, p + 1, 1)

    # Parse recycle_up_to
    if recycle_up_to is None:
        pass
    elif recycle_up_to <= 1:
        recycle_up_to = int(recycle_up_to*n)
    else:
        recycle_up_to = int(recycle_up_to)

    # Parse feature statistic function
    if feature_stat_fn == 'lasso':
        feature_stat_fn = knockoff_stats.lasso_statistic
    elif feature_stat_fn == 'ols':
        feature_stat_fn = knockoff_stats.linear_coef_diff
    elif feature_stat_fn == 'margcorr':
        feature_stat_fn = knockoff_stats.marg_corr_diff

    # Sample knockoffs
    if knockoffs is None:
        knockoffs = group_gaussian_knockoffs(
            X=X, groups=groups, Sigma=Sigma, **knockoff_kwargs,
        )[:, :, 0]

        # Possibly use recycling
        if recycle_up_to is not None:

            # Split
            rec_knockoffs = X[:recycle_up_to]
            new_knockoffs = knockoffs[recycle_up_to:]

            # Combine
            knockoffs = np.concatenate((rec_knockoffs, new_knockoffs), axis=0)

    # Feature statistics
    W = feature_stat_fn(
        X=X, knockoffs=knockoffs, y=y, groups=groups, **feature_stat_kwargs
    )

    # Data dependent threshhold and group selections
    T = knockoff_stats.data_dependent_threshhold(W=W, fdr=fdr)
    selected_flags = (W >= T).astype("float32")

    # Return
    return selected_flags
