import numpy as np
import scipy.cluster.hierarchy as hierarchy

from . import utilities
from .graphs import sample_data, create_correlation_tree
from .knockoffs import group_gaussian_knockoffs
from .knockoff_stats import calc_nongroup_LSM, calc_data_dependent_threshhold


def evaluate_grouping(X, y, 
                      corr_matrix, groups, q, 
                      non_nulls = None,
                      copies = 20, 
                      feature_stat_fn = calc_nongroup_LSM,
                      **kwargs):
    """ Calculates empirical power, power, and FDP by
    running knockoffs. Does this "copies" times.
    :param X: n x p design matrix
    :param y: n-length response array
    :param corr_matrix: True correlation matrix of X, which is
    assumed to be centered and scaled.
    :param groups: p-length array of the groups of each feature
    :param non_nulls: p-length array where 0's indicate null variables.
    :param q: Desiredd Type 1 error level.
    :param feature_stat_fn: A function which creates the feature statistics (W)
     given the design matrix X, the response y, the knockofs, and the groups.
     Defaults to calc_nongroup_LSM.
    :param verbose: Whether the knockoff constructor should give progress reports.
    :param kwargs: kwargs to the Gaussian Knockoffs constructor."""
    
    m = np.unique(groups).shape[0]
    group_sizes = utilities.calc_group_sizes(groups)

    # Knockoff generation
    all_knockoffs, S = group_gaussian_knockoffs(
        X, corr_matrix, groups, copies = copies, tol = 1e-3, return_S = True, 
        **kwargs
    )
    
    # For each knockoff, calculate FDP, empirical power, power
    fdps = []
    powers = []
    hat_powers = []
    for j in range(copies):
        knockoffs = all_knockoffs[:, :, j]
    
        # Statistics
        W = feature_stat_fn(
            X = X, knockoffs = knockoffs, y = y, groups = groups
        )

        # Calculate data-dependent threshhold
        T = calc_data_dependent_threshhold(W, fdr = q)
        selected_flags = (W >= T)
        #selected = np.where(selected_flags)[0] 

        # Calculate empirical power
        hat_power = np.einsum('m,m->',(1/group_sizes), selected_flags.astype('float32'))
        hat_powers.append(hat_power)

        # Possibly, calculate oracle FDP and power
        if non_nulls is not None:
            true_selection = np.zeros(m)
            for j in range(m):
                flag = np.abs(non_nulls[groups == j+1]).sum() > 0
                true_selection[j] = flag

            # True power
            power = np.einsum(
                'm,m,m->', (1/group_sizes), selected_flags, true_selection
            )#/true_selection.sum()
            power = np.round(power, 2)

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
        num_non_nulls = float(np.sum(non_nulls != 0))
        hat_powers = np.array(hat_powers)/num_non_nulls
        fdps = np.array(fdps)
        powers = np.array(powers)/num_non_nulls
        return fdps, powers, hat_powers
    else:
        return np.array(hat_powers)
    
def select_highest_power(X, y, corr_matrix, 
                         link, 
                         cutoffs = None,
                         reduction = 10, 
                         non_nulls = None,
                         S_matrices = None,
                         **kwargs):
    """
    :param link: Link object returned by scipy hierarchical cluster, 
    or the create_correlation_tree function.
    :param cutoffs: List of cutoffs to consider (makes link and reduction
    parameters redundant). Defaults to None
    :param reduction: If reduction = 10, only look at every 10th cutoff 
    (i.e. only consider cutoffs which occur after each 10 steps in the 
    agglomerative clustering step)
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
        cutoffs = link[:, 2]
        cutoffs = cutoffs[::reduction]

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