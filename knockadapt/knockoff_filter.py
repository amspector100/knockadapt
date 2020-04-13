import numpy as np
from . import knockoff_stats
from .knockoffs import group_gaussian_knockoffs


class MXKnockoffFilter():
    def __init__(self):
        pass

    def sample_knockoffs(
        self,
        X,
        Sigma,
        groups,
        knockoff_kwargs,
        recycle_up_to,
    ):

        # SDP degen flag (for internal use)
        if '_sdp_degen' in knockoff_kwargs:
            _sdp_degen = knockoff_kwargs.pop('_sdp_degen')
        else:
            _sdp_degen = False

        # Initial sample
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

        # For high precision simulations of degenerate knockoffs,
        # ensure degeneracy
        if _sdp_degen:
            sumcols = X[:, 0] + knockoffs[:, 0]
            knockoffs = sumcols.reshape(-1, 1) - X

        self.knockoffs = knockoffs
        return knockoffs

    def make_selections(self, W, fdr):
        """" Calculate data dependent threshhold and selections """
        T = knockoff_stats.data_dependent_threshhold(W=W, fdr=fdr)
        selected_flags = (W >= T).astype("float32")
        return selected_flags

    def forward(
        self,
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
            knockoffs = self.sample_knockoffs(
                X=X,
                Sigma=Sigma,
                groups=groups,
                knockoff_kwargs=knockoff_kwargs,
                recycle_up_to=recycle_up_to,
            )

        # Feature statistics
        self.W, self.Z = feature_stat_fn(
            X=X, 
            knockoffs=knockoffs,
            y=y, 
            groups=groups,
            return_Z=True, 
            **feature_stat_kwargs
        )

        selected_flags = self.make_selections(self.W, fdr)

        # Return
        return selected_flags