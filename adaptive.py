import numpy as np
import scipy.cluster.hierarchy as hierarchy

from . import utilities
from .graphs import sample_data, create_correlation_tree
from .knockoffs import group_gaussian_knockoffs
from .knockoff_stats import calc_group_LSM, calc_data_dependent_threshhold


def evaluate_grouping(X, y, corr_matrix, groups, 
                      non_nulls = None,
                      copies = 10, 
                      q = 0.1,
                      verbose = False,
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
    :param verbose: Whether the knockoff constructor should give progress reports.
    :param kwargs: kwargs to the Gaussian Knockoffs constructor."""
    
    m = np.unique(groups).shape[0]
    group_sizes = utilities.calc_group_sizes(groups)

    # Knockoff generation
    all_knockoffs, S = group_gaussian_knockoffs(
        X, corr_matrix, groups, copies = copies, tol = 1e-3, return_S = True, 
        verbose = verbose, **kwargs
    )
    
    # For each knockoff, calculate FDP, empirical power, power
    fdps = []
    powers = []
    hat_powers = []
    for j in range(copies):
        knockoffs = all_knockoffs[:, :, j]
    
        # Statistics
        W = calc_group_LSM(X, knockoffs, y, groups = groups)

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

            # print(1/group_sizes)
            # print('---------------')
            # print('hi', true_selection.sum())
            # print([x for _,x in sorted(zip(np.abs(W), selected_flags))])
            # print([x for _,x in sorted(zip(np.abs(W), true_selection))])
            # print(sorted(W, key = lambda x: abs(x)))

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
        return fdps, powers, hat_powers
    else:
        return hat_powers
    
def select_highest_power(X, y, corr_matrix, 
                         link, 
                         reduction = 10, 
                         non_nulls = None,
                         **kwargs):
    """
    :param link: Link object returned by scipy hierarchical cluster, 
    or the create_correlation_tree function
    :param reduction: If reduction = 10, only look at every 10th cutoff 
    (i.e. only consider cutoffs which occur after each 10 steps in the 
    agglomerative clustering step)
    :param non_nulls: p-length array where 0 indicates that that feature is null.
    Defaults to None.
    returns: If non_nulls is None, returns the list of cutoffs and associated
    empirical powers.
    If non_nulls is not None, returns cutoffs, associated empirical powers,
    associated FDR estimates, and associated power estimates.
    """
    
    # Create cutoffs and groups - for effieicny,
    # only currently going to look at every 10th cutoff
    cutoffs = link[:, 2]
    cutoffs = cutoffs[::reduction]
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
        if non_nulls is None:
            hat_powers = evaluate_grouping(
                X, y, corr_matrix, groups, non_nulls = non_nulls, **kwargs
            )
            cutoff_hat_powers.append(np.array(hat_powers).mean())
        else:
            fdps, powers, hat_powers = evaluate_grouping(
                X, y, corr_matrix, groups, non_nulls = non_nulls, **kwargs
            )
            cutoff_hat_powers.append(np.array(hat_powers).mean())
            cutoff_fdps.append(np.array(fdps).mean())
            cutoff_powers.append(np.array(powers).mean())
    
    # Return arrays
    if non_nulls is None:
        return cutoffs, cutoff_hat_powers, Ms
    else:
        return cutoffs, cutoff_hat_powers, cutoff_fdps, cutoff_powers, Ms

def test_proposed_method(n = 50,
                         p = 50, 
                         coeff_size = 10, 
                         q = 0.25, 
                         link_method = 'complete',
                         ):

    pass