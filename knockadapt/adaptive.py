import copy
import numpy as np
import scipy.cluster.hierarchy as hierarchy

from . import utilities
from .graphs import sample_data, create_correlation_tree
from .knockoffs import group_gaussian_knockoffs
from .knockoff_stats import group_lasso_LCD, calc_data_dependent_threshhold

def create_cutoffs(link, reduction, max_size):

    # Only consider partitions w groups less than max size
    # - Computational justification (easier to do SDP)
    # - Practical justification (groups > 100 not so useful)
    # - Power justification (only can try so many cutoffs,
    # the super high ones usually perform badly)
    max_group_sizes = np.maximum.accumulate(link[:, 3]) 
    subset = link[max_group_sizes <= max_size]

    # Create cutoffs
    spacing = int(subset.shape[0]/reduction)
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

class GroupKnockoffEval():
    """ Evaluates power, fdr, empirical power
    of different groupings of knockoffs. 
    :param corr_matrix: True correlation matrix of X, which is
    assumed to be centered and scaled.
    :param non_nulls: p-length array where 0's indicate null variables.
    :param q: Desiredd Type 1 error level.
    :param feature_stat_fn: A function which creates the feature statistics (W)
    given the design matrix X, the response y, the knockofs, and the groups.
    Defaults to calc_nongroup_LSM.
    :param feature_stat_kwargs: kwargs to the feature stat function
    :param kwargs: kwargs to the Gaussian Knockoffs constructor
    """ 

    def __init__(self, corr_matrix, q,
                 non_nulls = None, 
                 feature_stat_fn = group_lasso_LCD,
                 feature_stat_kwargs = {},
                 **kwargs):

        # Save values
        self.p = corr_matrix.shape[0]
        self.sigma = corr_matrix
        self.q = q
        self.non_nulls = non_nulls
        self.feature_stat_fn = group_lasso_LCD
        self.feature_stat_kwargs = feature_stat_kwargs
        self.knockoff_kwargs = kwargs

        # Defaults for knockoff_kwargs
        if 'verbose' not in self.knockoff_kwargs:
            self.knockoff_kwargs['verbose'] = False
        if 'sdp_verbose' not in self.knockoff_kwargs:
            self.knockoff_kwargs['sdp_verbose'] = False


    def combine_S_kwargs(self, new_kwargs):
        """ Combines default knockoff kwargs with new ones (new ones 
        override old ones) """

        knockoff_kwargs = copy.copy(self.knockoff_kwargs)
        for key in new_kwargs:
            knockoff_kwargs[key] = new_kwargs[key]

        return knockoff_kwargs


    def sample_knockoffs(self, X, groups, recycle_up_to = None, copies = 20, **kwargs):
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
            test_knockoffs = group_gaussian_knockoffs(
                testX, self.sigma, groups, copies = copies, tol = 1e-2, 
                **self.knockoff_kwargs
            )

            # Recycle first half and combine
            recycled_knockoffs = np.repeat(trainX, copies).reshape(-1, self.p, copies)
            all_knockoffs = np.concatenate(
                (recycled_knockoffs, test_knockoffs), axis = 0
            )

        # Else, vanilla Knockoff generation
        else:
            all_knockoffs = group_gaussian_knockoffs(
                X, self.sigma, groups, copies = copies, tol = 1e-2,
                **self.knockoff_kwargs
            )

            # # Delete all this plz
            # all_knockoffs = all_knockoffs[150:]
            # rec = np.repeat(X[0:150], copies).reshape(-1, self.p, copies)
            # all_knockoffs = np.concatenate(
            #     (rec, all_knockoffs), axis = 0
            # )

        return all_knockoffs


    def eval_knockoff_instance(self, X, knockoffs, y, groups, 
                               group_sizes = None, group_selections = None):
        """ Calculates empirical power and possibly true power/fdp
        for a SINGLE knockoff instance, which must have exactly the same 
        shape as X """

        # Possibly compute group sizes, group selections
        if group_sizes is None:
            group_sizes = utilities.calc_group_sizes(groups)
        if group_selections is None and self.non_nulls is not None:
            group_selections = utilities.fetch_group_nonnulls(
                self.non_nulls, groups
            )

            # Calculate number of non-nulls (if we're under the global null,
            # or aren't the oracle, we let the denominator be p)
            num_non_nulls = np.sum(group_selections)
            if num_non_nulls == 0:
                num_non_nulls = self.p
        else:
            num_non_nulls = self.p


        # W statistics
        W = self.feature_stat_fn(
            X = X, knockoffs = knockoffs, y = y, groups = groups,
            **self.feature_stat_kwargs
        )
        if np.max(group_sizes) == 1:
            tups = [(Wv, flag) for Wv, flag in zip(W, group_selections)]
            sorted_W = sorted(tups, key = lambda x: -1*abs(x[0]))

        # Data dependent threshhold and group selections
        T = calc_data_dependent_threshhold(W, fdr = self.q)
        selected_flags = (W >= T).astype('float32')

        # Empirical power
        hat_power = np.einsum('m,m->',(1/group_sizes), selected_flags) / num_non_nulls

        # Possibly, calculate oracle FDP and power
        if self.non_nulls is not None:

            # True power
            power = np.einsum(
                'm,m,m->', (1/group_sizes), selected_flags, group_selections
            )
            power = power / num_non_nulls

            # FDP
            FDP = np.einsum(
                'p,p->', selected_flags, 1-group_selections
            )
            FDP = FDP / max(1, selected_flags.sum())

        else:
            power = None
            FDP = None


        return FDP, power, hat_power

    
    def eval_grouping(self, X, y, groups, 
                      recycle_up_to = None, 
                      copies = 20,
                      **kwargs):
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

        # n, m, X
        n = X.shape[0]
        m = np.unique(groups).shape[0]
        group_sizes = utilities.calc_group_sizes(groups)

        # Get knockoffs
        all_knockoffs = self.sample_knockoffs(
            X=X, groups=groups, recycle_up_to=recycle_up_to, copies=copies,
            **kwargs
        )

        # Possibly calculate true selections for this grouping
        if self.non_nulls is not None:
            group_selections = utilities.fetch_group_nonnulls(
                self.non_nulls, groups
            )

        # For each knockoff, calculate FDP, empirical power, power.
        fdps = []
        powers = []
        hat_powers = []

        # Loop through each knockoff - TODO: paralellize properly
        for j in range(copies):

            # Calculate one FDP, power, hatpower for each copy
            knockoffs = all_knockoffs[:, :, j]
            fdp, power, hat_power = self.eval_knockoff_instance(
                X = X, y = y, groups = groups, 
                knockoffs = knockoffs,
                group_sizes = group_sizes, 
                group_selections = group_selections
            )

            # Append
            fdps.append(fdp)
            powers.append(power)
            hat_powers.append(hat_power)

        # Return
        hat_powers = np.array(hat_powers)
        if self.non_nulls is None:
            return hat_powers
        else:
            fdps = np.array(fdps)
            powers = np.array(powers)
            return fdps, powers, hat_powers


    def eval_many_cutoffs(self,
                          X, y, 
                          link, 
                          cutoffs = None,
                          reduction = 10, 
                          max_group_size = 100,
                          S_matrices = None,
                          **kwargs):
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
            S_matrices = {cutoff:None for cutoff in cutoffs}

        # Initialize
        cutoff_hat_powers = []
        cutoff_fdps = []
        cutoff_powers = []
        
        # This is inefficient but it's not a bottleneck
        # - is at worst O(p^2)
        for cutoff in cutoffs:
            
            # Create groups
            groups = hierarchy.fcluster(link, cutoff, criterion = "distance")

            # Get S matrix
            S = S_matrices[cutoff]

            # Possible just get empirical powers if there's no ground truth
            outputs = self.eval_grouping(
                X, y, groups, S = S, **kwargs
            )

            # Return differently based on whether non_nulls supplied
            if self.non_nulls is None:
                hat_powers = outputs
                cutoff_hat_powers.append(np.array(hat_powers).mean())
                cutoff_fdps.append(None)
                cutoff_powers.append(None)
            else:
                fdps, powers, hat_powers = outputs
                cutoff_hat_powers.append(np.array(hat_powers).mean())
                cutoff_fdps.append(np.array(fdps).mean())
                cutoff_powers.append(np.array(powers).mean())

        
        # Return arrays
        return cutoffs, cutoff_fdps, cutoff_powers, cutoff_hat_powers


    def sample_split(self, X, y):

        pass