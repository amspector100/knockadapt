import warnings
import numpy as np
from sklearn import linear_model
from group_lasso import GroupLasso
from statsmodels.stats.moment_helpers import cov2corr

from .utilities import random_permutation

DEFAULT_REG_VALS = np.logspace(-2, 1, base = 10, num = 5)

def calc_LCD(Z, groups):
    """
    Calculates coefficient differences for knockoffs
    :param Z: statistics for each feature (including knockoffs)
    2p dimensional numpy array
    :param groups: p dimensional numpy array of group membership
    with m discrete values
    :returns W: m dimensional array of knockoff LCD statistics
    """

    # Get dims, initialize output
    p = groups.shape[0]
    m = np.unique(groups).shape[0]
    W = np.zeros(m)

    # Separate
    Z_true = Z[0:p]
    Z_knockoff = Z[p:]

    # Create Wjs
    for j in range(m):
        true_coeffs = Z_true[groups == j]
        knock_coeffs = Z_knockoff[groups == j]
        Wj = np.sum(np.abs(true_coeffs)) - np.sum(np.abs(knock_coeffs))
        W[j-1] = Wj

    return W

def calc_LSM(Z, groups):
    """
    Calculates signed maximum statistics for knockoffs
    :param Z: statistics for each feature (including knockoffs)
    2p dimensional numpy array
    :param groups: p dimensional numpy array of group membership
    with m discrete values
    :returns W: m dimensional array of knockoff LSM statistics
    """

    # Get dims, initialize output
    p = groups.shape[0]
    m = np.unique(groups).shape[0]

    # Calculate signed maxes
    inds = np.arange(0, p, 1)
    W = np.maximum(Z[inds], Z[inds + p])
    W = W * np.sign(np.abs(Z[inds]) - np.abs(Z[inds + p]))

    # Combine them for each group
    W_group = np.zeros(m)
    for i in range(p):
        W_group[groups[i]-1] += W[i]
        
    return W_group

def calc_nongroup_LSM(X, knockoffs, y, groups = None, **kwargs):
    """ Calculates difference between average group Lasso signed maxs. 
    Does NOT use a group lasso regression class, unfortunately.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :param kwargs: kwargs for sklearn Lasso class
    """
    
    # Bind data
    n = X.shape[0]
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis = 1)

    # By default, all variables are their own group
    if groups is None:
        groups = np.arange(0, p, 1)
    m = np.unique(groups).shape[0]
    
    # Fit
    alphas, _, coefs = linear_model.lars_path(
        features, y, method='lasso', **kwargs,
    )
    
    # Calculate places where features enter the model
    Z = np.zeros(2*p)
    for i in range(2*p):
        if (coefs[i] != 0).sum() == 0:
            Z[i] = 0
        else:
            Z[i] = alphas[np.where(coefs[i] != 0)[0][0]]

    # Calculate group LSM
    W_group = calc_LSM(Z, groups)
        
    return W_group

def fit_group_lasso(X, knockoffs, y, groups, **kwargs):
    """ Fits a group lasso model.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :param kwargs: kwargs for group-lasso GroupLasso class.
    In particular includes reg_vals, a list of regularizations
    (lambda values) which defaults to [(0.05, 0.05)]. In each
    tuple of the list, the first value is the group regularization,
    the second value is the individual regularization.
    """

    warnings.filterwarnings("ignore")

    # Bind data
    n = X.shape[0]
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis = 1)

    # By default, all variables are their own group
    if groups is None:
        groups = np.arange(0, p, 1)
     
    # Make sure variables and their knockoffs are in the same group
    # This is necessary for antisymmetry
    doubled_groups = np.concatenate([groups, groups], axis = 0)

    # Standardize - dramatically reduces performance for some reason
    # features = (features - features.mean())/features.std()
    # y = (y - y.mean())/y.std()

    # Get regularizations
    if 'reg_vals' in kwargs:
        reg_vals = kwargs['reg_vals']
    else:
        reg_vals = [(x,x) for x in DEFAULT_REG_VALS]
    
    # Fit model
    best_gl = None
    best_score = -1*np.inf
    for group_reg, l1_reg in reg_vals:

        # Fit
        gl = GroupLasso(groups=doubled_groups, tol = 1e-3, 
                        group_reg = group_reg, l1_reg = l1_reg,
                        **kwargs)
        gl.fit(features, y.reshape(n, 1))

        # Score, possibly select
        score = gl.score(features, y.reshape(n, 1))
        if score > best_score:
            best_score = score
            best_gl = gl

    warnings.simplefilter("always")

    return best_gl


def group_lasso_LSM(X, knockoffs, y, groups, **kwargs):
    """ Calculates mean group Lasso signed max. 
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :param kwargs: kwargs for group-lasso GroupLasso class
    """
    gl = fit_group_lasso(X, knockoffs, y, groups = None, **kwargs)

    # Create LSM statistics
    Z = gl.coef_
    W = calc_LSM(Z, groups)
    return W


def group_lasso_LCD(X, knockoffs, y, groups = None, **kwargs):
    """ Calculates group Lasso coefficient difference. 
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :params **kwargs: kwargs to GroupLasso method 
    I.e. if features 2 and 3 are in the same group, 
    we set W(X, knockoffs, y) = 
    sum(abs coeff of X) - sum(abs coeff of knockoffs)
    """
    gl = fit_group_lasso(X, knockoffs, y, groups = None, **kwargs)

    # Create LCD statistics
    Z = gl.coef_
    W = calc_LCD(Z, groups)

    return W

def calc_data_dependent_threshhold(W, fdr=0.10, offset=1):
    """
    This is not efficient but it's definitely not a bottleneck.
    Follows https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R
    :param W: p-length numpy array of feature statistics
    :param fdr: desired FDR level (referred to as q in the literature)
    :param offset: if offset = 0, use knockoffs (which control modified FDR).
    Else, if offset = 1, use knockoff+ (controls exact FDR).
    """

    # Possibly values for Ts
    Ts = sorted(np.abs(W))
    Ts = np.concatenate([np.array((0,)), Ts], axis = 0)
    
    # Calculate ratios
    def hat_fdp(t):
        return ((W <= -t).sum() + offset)/max(1, np.sum(W >= t))
    hat_fdp = np.vectorize(hat_fdp)
    ratios = hat_fdp(Ts)
    
    # Find maximum
    acceptable = Ts[ratios <= fdr]
    if acceptable.shape[0] == 0:
        return np.inf
    
    return acceptable[0]
    