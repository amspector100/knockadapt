import copy
import numpy as np
import scipy.cluster.hierarchy as hierarchy

from . import utilities
from . import knockoff_stats
from .knockoffs import gaussian_knockoffs
from .knockoff_stats import LassoStatistic, data_dependent_threshhold
from .knockoff_filter import KnockoffFilter

class KnockoffBicycle(KnockoffFilter):
    """
    A class which implements pre/recycling
    for adaptive knockoff selection.
    TODO: could also call this the bicycle
    """
    def __init__(
        self,
        fixedX=False,
        knockoff_kwargs={},
        knockoff_type = 'gaussian',
    ):
        self.fixedX = fixedX
        self.knockoff_kwargs = knockoff_kwargs
        if knockoff_type != 'gaussian':
            raise ValueError(
                f"Non-gaussian knockoff types not implemented yet"
            )

    def forward(
        self,
        X,
        y,
        groups=None,
        mu=None,
        Sigma=None,
        rec_prop=0.5,
        **filter_kwargs
    ):

        # Save params
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.mu = mu
        self.p = Sigma.shape[0]
        self.Sigma = Sigma
        self.groups = groups

        # Generate knockoffs
        if 'knockoffs' in filter_kwargs:
            self.knockoffs = filter_kwargs['knockoffs']
        else:
            self.knockoffs = gaussian_knockoffs(
                X=self.X, 
                groups=self.groups,
                mu=self.mu,
                Sigma=self.Sigma,
                **self.knockoff_kwargs,
            )[:, :, 0]

        # Pre-cycled features and knockfoffs
        nrec = int(rec_prop*self.n)
        self.preknockoffs = self.knockoffs.copy()
        self.preknockoffs[nrec:] = (self.knockoffs[nrec:] + self.X[nrec:])/2
        self.preX = self.X.copy()
        self.preX[nrec:] = self.preknockoffs[nrec:]
        self.prefilter = KnockoffFilter(fixedX=self.fixedX)
        self.prefilter.forward(
            X=self.preX,
            y=self.y,
            mu=self.mu,
            Sigma=self.Sigma,
            knockoffs=self.preknockoffs,
        )

        # Re-cycled features and knockoffs
        self.reknockoffs = self.knockoffs.copy()
        self.reknockoffs[0:nrec] = self.X[0:nrec].copy()
        self.refilter = KnockoffFilter(fixedX=self.fixedX)
        self.refilter.forward(
            X=self.preX,
            y=self.y,
            mu=self.mu,
            Sigma=self.Sigma,
            knockoffs=self.preknockoffs,
        )


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


def create_cutoffs(link, reduction, max_size):

    # Only consider partitions w groups less than max size
    # - Computational justification (easier to do SDP)
    # - Practical justification (groups > 100 not so useful)
    # - Power justification (only can try so many cutoffs,
    # the super high ones usually perform badly)
    max_group_sizes = np.maximum.accumulate(link[:, 3])
    subset = link[max_group_sizes <= max_size]

    # Create cutoffs
    spacing = int(subset.shape[0] / reduction)
    cutoffs = subset[:, 2]

    # Add 0 to beginning (this is our baseline - no groups)
    # and then add spacing
    cutoffs = np.insert(cutoffs, 0, 0)
    cutoffs = cutoffs[::spacing]

    # If cutoffs aren't unique, only consider some
    # (This occurs only in simulated data)
    if np.unique(cutoffs).shape[0] != cutoffs.shape[0]:
        cutoffs = np.unique(cutoffs)

    return cutoffs


def resample_tau_conditionally(Ws, group_sizes, non_nulls, fdr=0.2, reps=10000):
    """
    :param Ws: R x p matrix of W statistics
    :param non_nulls: R x p matrix of binary flags, where
    1 indicates that this variable is a non-nulls
    :param group_sizes: R x p matrix of group sizes
    returns: R length array of empirical power samples.
    """

    # Reflip null coins
    R = Ws.shape[0]
    p = Ws.shape[1]
    signs = np.random.binomial(1, 0.5, (R, p, reps))
    signs = (1 - np.expand_dims(non_nulls, axis=-1)) * signs
    signs = 1 - 2 * signs

    # Recalculate Ws and empirical powers
    new_Ws = np.expand_dims(Ws, axis=-1) * signs

    # Batch
    new_Ws = new_Ws.reshape(R * reps, p, order="F")
    group_sizes = np.concatenate([group_sizes for _ in range(reps)], axis=0)
    non_nulls = np.concatenate([non_nulls for _ in range(reps)], axis=0)
    epowers = calc_epowers(new_Ws, group_sizes, non_nulls, fdr=fdr)

    # Reshape and return
    epowers = epowers.reshape(R, reps, order="F")
    return epowers


def apprx_epower(
    Ws,
    group_sizes,
    q=0.25,
    eps=0.05,
    delta=0.05,
    num_partitions=1,
    reduce_var=1,
    use_signs=True,
):
    """
    Let p be the number of features, R be the 
    number of partitions/groupiings.
    :param Ws: an R x p matrix of W-statistics.
    All groupings must be padded to length p.
    :param group_sizes: An R x p matrix of 
    group sizes. 
    """

    # Sort Ws and group sizes by absolute value
    sorting_inds = np.argsort(-1 * np.abs(Ws), axis=1)
    sorted_Ws = np.take_along_axis(Ws, sorting_inds, axis=1)
    sorted_group_sizes = np.take_along_axis(group_sizes, sorting_inds, axis=1)

    # Invert group_sizes
    sorted_group_sizes[sorted_group_sizes == 0] = -1
    inv_group_sizes = 1 / sorted_group_sizes
    inv_group_sizes[inv_group_sizes < 0] = 0

    # Cumulative groupsizes, positives, negatives
    cum_epower = np.cumsum(
        inv_group_sizes, axis=1
    )  # ((sorted_Ws > 0)*sorted_group_sizes), axis=1)
    npos = np.cumsum((sorted_Ws > 0), axis=1).astype("float32")
    nneg = np.cumsum((sorted_Ws <= 0), axis=1).astype("float32")

    # Prevent divide by 0 errors
    npos[npos == 0] = 1

    # These cutoffs control the FDR
    flags = ((nneg + 1) / npos <= q) * (sorted_Ws != 0)
    estimates = (cum_epower * flags).sum(axis=1)

    return estimates


def weighted_binomial_feature_stat(
    Ws, group_sizes, eps=0.05, delta=0.05, num_partitions=1, reduce_var=1,
):
    """
    Let p be the number of features, R be the 
    number of partitions/groupiings.
    :param Ws: an R x p matrix of W-statistics.
    All groupings must be padded to length p.
    :param group_sizes: An R x p matrix of 
    group sizes. 
    """

    # Constants to experiment with
    c = 0  # 3/8
    k = 0  # 2

    # Sort Ws and group sizes by absolute value
    sorting_inds = np.argsort(-1 * np.abs(Ws), axis=1)
    sorted_Ws = np.take_along_axis(Ws, sorting_inds, axis=1)
    sorted_group_sizes = np.take_along_axis(group_sizes, sorting_inds, axis=1)

    # Invert group_sizes
    sorted_group_sizes[sorted_group_sizes == 0] = -1
    inv_group_sizes = 1 / sorted_group_sizes
    inv_group_sizes[inv_group_sizes < 0] = 0

    # How much you weight each positive
    R = Ws.shape[0]
    p = Ws.shape[1]
    weights = (1 / np.arange(1, p + 1, 1) ** 0.5).reshape(1, p)

    # Otherwise cond. variance bounds, tight under global null
    # var_bounds = ((weights *inv_group_sizes)**2).sum(axis=1)/4

    # Nonzero counts
    nonzero_counts = (Ws != 0).sum(axis=1)

    # Actual estimates
    estimates = ((sorted_Ws > 0) * (inv_group_sizes * weights)).sum(axis=1)
    estimates -= ((sorted_Ws < 0) * (inv_group_sizes * weights)).sum(axis=1)
    estimates = estimates / (nonzero_counts ** (c))

    # Construct adaptive variance estimates
    adaptive_var = (2 * (sorted_Ws < 0) * ((inv_group_sizes * weights) ** 2)).sum(
        axis=1
    )
    adaptive_var = adaptive_var / (nonzero_counts ** (2 * c))
    adaptive_var[adaptive_var == 0] = 1

    # "Lower confidence bound"
    estimates = estimates - k * np.sqrt(adaptive_var)

    # Apply the Hoeffding bound
    sigmaN2 = (
        (1 / eps) * np.sqrt(-1 * np.log(delta)) * np.sqrt(adaptive_var) / reduce_var
    )

    # Sample and return
    return sigmaN2 * np.random.randn(R) + estimates


class GroupKnockoffEval:
    """ Evaluates power, fdr, empirical power
    of different groupings of knockoffs. 
    :param corr_matrix: True correlation matrix of X, which is
    assumed to be centered and scaled.
    :param non_nulls: p-length array where 0's indicate null variables.
    :param q: Desiredd Type 1 error level.
    :param feature_stat: A class with a fit method. The fit method
    should create and return the feature statistics (W)
    given the design matrix X, the response y, the knockofs, and the groups.
    Defaults to a LassoStatistic.
    :param feature_stat_kwargs: kwargs to the feature stat function
    :param kwargs: kwargs to the Gaussian Knockoffs constructor
    """

    def __init__(
        self,
        corr_matrix,
        q,
        non_nulls=None,
        feature_stat=LassoStatistic,
        feature_stat_kwargs={},
        **kwargs
    ):

        # Save values
        self.p = corr_matrix.shape[0]
        self.sigma = corr_matrix
        self.q = q
        self.non_nulls = non_nulls
        self.feature_stat = feature_stat()
        self.feature_stat_kwargs = feature_stat_kwargs
        self.knockoff_kwargs = kwargs

        # Defaults for knockoff_kwargs
        if "verbose" not in self.knockoff_kwargs:
            self.knockoff_kwargs["verbose"] = False
        if "sdp_verbose" not in self.knockoff_kwargs:
            self.knockoff_kwargs["sdp_verbose"] = False

    def combine_S_kwargs(self, new_kwargs):
        """ Combines default knockoff kwargs with new ones (new ones 
        override old ones) """

        knockoff_kwargs = copy.copy(self.knockoff_kwargs)
        for key in new_kwargs:
            knockoff_kwargs[key] = new_kwargs[key]

        return knockoff_kwargs

    def sample_knockoffs(self, X, groups, recycle_up_to=None, copies=20, **kwargs):
        """ Samples copies knockoffs of X. 
        :param int recycle_up_to: If recycle_up_to = k, then the 
        knockoff generator will use the data itself as the knockoffs
        for the first k observations.
        :param **kwargs: kwargs to pass to the knockoff sampler. Will override
        the default kwargs built into the class.
        """

        knockoff_kwargs = self.combine_S_kwargs(kwargs)

        # Possibly recycle to some extent:
        if recycle_up_to is not None:

            # Split
            trainX = X[:recycle_up_to]
            testX = X[recycle_up_to:]

            # Generate second half knockoffs
            test_knockoffs = gaussian_knockoffs(
                X=testX, 
                Sigma=self.sigma,
                groups=groups,
                copies=copies,
                **knockoff_kwargs
            )

            # Recycle first half and combine
            recycled_knockoffs = np.repeat(trainX, copies).reshape(-1, self.p, copies)
            all_knockoffs = np.concatenate((recycled_knockoffs, test_knockoffs), axis=0)

        # Else, vanilla Knockoff generation
        else:
            all_knockoffs = gaussian_knockoffs(
                X=X,
                Sigma=self.sigma,
                groups=groups,
                copies=copies,
                **knockoff_kwargs
            )

        return all_knockoffs

    def eval_knockoff_instance(
        self, X, knockoffs, y, groups, group_sizes=None, group_selections=None
    ):
        """ Calculates empirical power and possibly true power/fdp
        for a SINGLE knockoff instance, which must have exactly the same 
        shape as X """

        # Possibly compute group sizes, group selections
        if group_sizes is None:
            group_sizes = utilities.calc_group_sizes(groups)
        if group_selections is None and self.non_nulls is not None:
            group_selections = utilities.fetch_group_nonnulls(self.non_nulls, groups)

        # Calculate number of non-nulls (if we're under the global null,
        # or aren't the oracle, we let the denominator be p)
        num_non_nulls = np.sum(group_selections)
        if num_non_nulls == 0:
            num_non_nulls = self.p

        # W statistics
        W = self.feature_stat.fit(
            X=X, knockoffs=knockoffs, y=y, groups=groups, **self.feature_stat_kwargs
        )

        # Data dependent threshhold and group selections
        T = data_dependent_threshhold(W, fdr=self.q)
        selected_flags = (W >= T).astype("float32")

        # Empirical power
        hat_power = (
            np.einsum("m,m->", (1 / group_sizes), selected_flags) / num_non_nulls
        )

        # Possibly, calculate oracle FDP and power
        if self.non_nulls is not None:

            # True power
            power = np.einsum(
                "m,m,m->", (1 / group_sizes), selected_flags, group_selections
            )
            power = power / num_non_nulls

            # FDP
            FDP = np.einsum("p,p->", selected_flags, 1 - group_selections)
            FDP = FDP / max(1, selected_flags.sum())

        else:
            power = None
            FDP = None

        return FDP, power, hat_power, W

    def eval_grouping(self, X, y, groups, recycle_up_to=None, copies=20, **kwargs):
        """ 
        Calculates empirical power, power, and FDP by
        running knockoffs. Does this "copies" times.
        :param X: n x p design matrix
        :param y: n-length response array
        :param groups: p-length array of the groups of each feature
        :param int recycle_up_to: If recycle_up_to = k, then the 
        knockoff generator will use the data itself as the knockoffs
        for the first k observations. This is necessary to prevent 
        "double dipping," i.e. looking at the data before runnign
        knockoffs, from violating FDR control. Defaults to None. 
        :param kwargs: kwargs to pass to knockoff sampler
        """

        # Group sizes
        group_sizes = utilities.calc_group_sizes(groups)

        # Get knockoffs
        all_knockoffs = self.sample_knockoffs(
            X=X, groups=groups, recycle_up_to=recycle_up_to, copies=copies, **kwargs
        )

        # Possibly calculate true selections for this grouping
        if self.non_nulls is not None:
            group_selections = utilities.fetch_group_nonnulls(self.non_nulls, groups)

        # For each knockoff, calculate FDP, empirical power, power.
        fdps = []
        powers = []
        hat_powers = []
        Ws = []

        # Loop through each knockoff - TODO: paralellize properly
        for j in range(copies):

            # Calculate one FDP, power, hatpower for each copy
            knockoffs = all_knockoffs[:, :, j]
            fdp, power, hat_power, W = self.eval_knockoff_instance(
                X=X,
                y=y,
                groups=groups,
                knockoffs=knockoffs,
                group_sizes=group_sizes,
                group_selections=group_selections,
            )

            # Append
            fdps.append(fdp)
            powers.append(power)
            hat_powers.append(hat_power)
            Ws.append(W)

        # Return
        hat_powers = np.array(hat_powers)
        Ws = np.array(Ws)
        if self.non_nulls is None:
            return hat_powers, Ws
        else:
            fdps = np.array(fdps)
            powers = np.array(powers)
            return fdps, powers, hat_powers, Ws

    def eval_many_cutoffs(
        self,
        X,
        y,
        link,
        cutoffs=None,
        reduction=10,
        max_group_size=100,
        S_matrices=None,
        **kwargs
    ):
        """
        :param X: n x p design matrix
        :param y: n-length response array
        :param groups: p-length array of the groups of each feature
        :param link: Link object returned by scipy hierarchical cluster, 
        or the create_correlation_tree function.
        :param cutoffs: List of cutoffs to consider (makes link and reduction
        parameters redundant). Defaults to None
        :param reduction: How many cutoffs to try. E.g. if reduction = 10,
        try 10 different grouping. These will be evenly spaced cutoffs 
        on the link.
        :param S_matrices: dictionary mapping cutoffs to S matrix for knockoffs
        :param kwargs: kwargs to eval_grouping, may contain kwargs to 
        gaussian group knockoffs constructor.

        returns: the list of cutoffs, associated FDR estimates, 
        associated power estimates, and associated empirical powers. 
        These middle two lists will contain elements "None" if non_nulls
        is None (since then we don't have oracle knowledge of power/FDR
        """

        # Create cutoffs and groups - for effieicny,
        # only currently going to look at every 10th cutoff
        if cutoffs is None:
            cutoffs = create_cutoffs(link, reduction, max_group_size)

        # Possibly create S_matrices dictionary
        # if not already supplied
        if S_matrices is None:
            S_matrices = {cutoff: None for cutoff in cutoffs}

        # Initialize
        cutoff_hat_powers = []
        cutoff_fdps = []
        cutoff_powers = []
        cutoff_Ws = []

        # This is inefficient but it's not a bottleneck
        # - is at worst O(p^2)
        for cutoff in cutoffs:

            # Create groups
            groups = hierarchy.fcluster(link, cutoff, criterion="distance")

            # Get S matrix
            S = S_matrices[cutoff]

            # Possible just get empirical powers if there's no ground truth
            outputs = self.eval_grouping(X, y, groups, S=S, **kwargs)

            # Return differently based on whether non_nulls supplied
            if self.non_nulls is None:
                hat_powers, Ws = outputs
                cutoff_hat_powers.append(np.array(hat_powers).mean())
                cutoff_fdps.append(None)
                cutoff_powers.append(None)
                cutoff_Ws.append(Ws)
            else:
                fdps, powers, hat_powers, Ws = outputs
                cutoff_hat_powers.append(np.array(hat_powers).mean())
                cutoff_fdps.append(np.array(fdps).mean())
                cutoff_powers.append(np.array(powers).mean())
                cutoff_Ws.append(Ws)

        # Return arrays
        return cutoffs, cutoff_fdps, cutoff_powers, cutoff_hat_powers, cutoff_Ws
