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

def evaluate_grouping(X, y, 
                      corr_matrix, groups, q, 
                      recycle_up_to = None,
                      non_nulls = None,
                      copies = 20, 
                      feature_stat_fn = group_lasso_LCD,
                      feature_stat_kwargs = {},
                      **kwargs):
    """ Calculates empirical power, power, and FDP by
    running knockoffs. Does this "copies" times.
    :param X: n x p design matrix
    :param y: n-length response array
    :param corr_matrix: True correlation matrix of X, which is
    assumed to be centered and scaled.
    :param groups: p-length array of the groups of each feature
    :param int recycle_up_to: If recycle_up_to = k, then the 
    knockoff generator will use the data itself as the knockoffs
    for the first k observations. This is necessary to prevent 
    "double dipping," i.e. looking at the data before runnign
    knockoffs, from violating FDR control. Defaults to None. 
    :param non_nulls: p-length array where 0's indicate null variables.
    :param q: Desiredd Type 1 error level.
    :param feature_stat_fn: A function which creates the feature statistics (W)
     given the design matrix X, the response y, the knockofs, and the groups.
     Defaults to calc_nongroup_LSM.
    :param verbose: Whether the knockoff constructor should give progress reports.
    :param feature_stat_kwargs: kwargs to the feature stat function
    :param kwargs: kwargs to the Gaussian Knockoffs constructor."""
    
    n = X.shape[0]
    p = X.shape[1]
    m = np.unique(groups).shape[0]
    group_sizes = utilities.calc_group_sizes(groups)

    # Possibly recycle to some extent:
    if recycle_up_to is not None:

        # Split
        trainX = X[:recycle_up_to]
        testX = X[recycle_up_to:]

        # Generate second half knockoffs
        test_knockoffs = group_gaussian_knockoffs(
            testX, corr_matrix, groups, copies = copies, tol = 1e-2, 
            **kwargs
        )

        # Recycle first half and combine
        recycled_knockoffs = np.repeat(trainX, copies).reshape(-1, p, copies)
        all_knockoffs = np.concatenate(
            (recycled_knockoffs, test_knockoffs), axis = 0
        )

    # Else, vanilla Knockoff generation
    else:
        all_knockoffs = group_gaussian_knockoffs(
            X, corr_matrix, groups, copies = copies, tol = 1e-2,
            **kwargs
        )
    
    # Possibly calculate true selections for this grouping
    if non_nulls is not None:
        true_selection = np.zeros(m)
        for j in range(m):
            flag = np.abs(non_nulls[groups == j+1]).sum() > 0
            true_selection[j] = flag

    # For each knockoff, calculate FDP, empirical power, power
    fdps = []
    powers = []
    hat_powers = []
    for j in range(copies):
        knockoffs = all_knockoffs[:, :, j]
    
        # Statistics
        W = feature_stat_fn(
            X = X, knockoffs = knockoffs, y = y, groups = groups,
            **feature_stat_kwargs
        )

        # Calculate data-dependent threshhold
        T = calc_data_dependent_threshhold(W, fdr = q)
        selected_flags = (W >= T)

        # Calculate empirical power
        hat_power = np.einsum('m,m->',(1/group_sizes), selected_flags.astype('float32'))
        hat_powers.append(hat_power)

        # Possibly, calculate oracle FDP and power
        if non_nulls is not None:

            # True power
            power = np.einsum(
                'm,m,m->', (1/group_sizes), selected_flags, true_selection
            )

            # FDP
            FDP = np.einsum(
                'p,p->', selected_flags.astype('float32'), 1-true_selection.astype('float32')
            )
            FDP = FDP / max(1, selected_flags.sum())
            
            # Append
            fdps.append(FDP)
            powers.append(power)
    
    # Return depending on whether we have oracle info
    if non_nulls is not None:
        num_non_nulls = float(np.sum(true_selection.astype('float32')))
        fdps = np.array(fdps)
        if num_non_nulls != 0:
            hat_powers = np.array(hat_powers)/num_non_nulls
            powers = np.array(powers)/num_non_nulls
        # If the global null holds, need a different denominator
        else:
            hat_powers = np.array(hat_powers)/float(p)
            powers = np.array(powers)/float(p)
        return fdps, powers, hat_powers
    else:
        return np.array(hat_powers)
    
def select_highest_power(X, y, corr_matrix, 
                         link, 
                         cutoffs = None,
                         reduction = 10, 
                         max_group_size = 100,
                         non_nulls = None,
                         S_matrices = None,
                         **kwargs):
    """
    :param link: Link object returned by scipy hierarchical cluster, 
    or the create_correlation_tree function.
    :param cutoffs: List of cutoffs to consider (makes link and reduction
    parameters redundant). Defaults to None
    :param reduction: How many cutoffs to try. E.g. if reduction = 10,
    try 10 different grouping. These will be evenly spaced cutoffs 
    on the link.
    :param non_nulls: p-length array where 0 indicates that that feature is null.
    Defaults to None.
    :param S_matrices: dictionary mapping cutoffs to S matrix for knockoffs
    :param kwargs: kwargs to evaluate_grouping, may contain kwargs to 
    gaussian group knockoffs constructor.

    returns: If non_nulls is None, returns the list of cutoffs and associated
    empirical powers.
    If non_nulls is not None, returns cutoffs, associated empirical powers,
    associated FDR estimates, and associated power estimates.
    """
    
    # Create cutoffs and groups - for effieicny,
    # only currently going to look at every 10th cutoff
    if cutoffs is not None:
        cutoffs = create_cutoffs(link, reduction, max_group_size)

    # Possibly create S_matrices dictionary
    # if not already supplied
    if S_matrices is None:
        S_matrices = {cutoff:None for cutoff in cutoffs}

    cutoff_hat_powers = []
    Ms = []
    if non_nulls is not None:
        cutoff_fdps = []
        cutoff_powers = []
    
    # This is inefficient but it's not a bottleneck
    # - is at worst O(p^2)
    for cutoff in cutoffs:
        
        # Create groups
        groups = hierarchy.fcluster(link, cutoff, criterion = "distance")
        Ms.append(np.unique(groups).shape[0])

        # Get S matrix
        S = S_matrices[cutoff]

        # Possible just get empirical powers if there's no ground truth
        if non_nulls is None:
            hat_powers = evaluate_grouping(
                X, y, corr_matrix, groups, non_nulls = non_nulls, 
                S = S, **kwargs
            )
            cutoff_hat_powers.append(np.array(hat_powers).mean())

        # Else, calculate it all!
        else:
            fdps, powers, hat_powers = evaluate_grouping(
                X, y, corr_matrix, groups, non_nulls = non_nulls,
                S = S, **kwargs
            )
            cutoff_hat_powers.append(np.array(hat_powers).mean())
            cutoff_fdps.append(np.array(fdps).mean())
            cutoff_powers.append(np.array(powers).mean())
    
    # Return arrays
    if non_nulls is None:
        return cutoffs, cutoff_hat_powers, Ms
    else:
        return cutoffs, cutoff_hat_powers, cutoff_fdps, cutoff_powers, Ms