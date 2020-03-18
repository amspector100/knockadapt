import numpy as np
from . import knockoff_stats, utilities
from .knockoffs import group_gaussian_knockoffs


def mx_knockoff_filter(
    X,
    y,
    Sigma,
    groups=None,
    knockoffs=None,
    feature_stat_fn=knockoff_stats.lasso_statistic,
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
	:param feature_stat_kwargs: Kwargs to pass to 
	the feature statistic.
	:param knockoff_kwargs: Kwargs to pass to the 
	knockoffs constructor.
    :param recycle_up_to: If not None, use the first int(recycle_up_to)
     rows of X as the first int(recycle_up_to) rows of knockoffs.
	"""

    if groups is None:
        groups = np.arange(1, p + 1, 1)

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
