import warnings
import numpy as np
from sklearn import linear_model
from pyglmnet import GLMCV
from group_lasso import GroupLasso
from statsmodels.stats.moment_helpers import cov2corr

from .utilities import random_permutation

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

def calc_group_LSM(X, knockoffs, y, groups = None, **kwargs):
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

    # Calculate LSM for each feature
    inds = np.arange(0, p, 1)
    W = np.maximum(Z[inds], Z[inds + p])
    W = W * np.sign(np.abs(Z[inds]) - np.abs(Z[inds + p]))

    # Combine groups
    W_group = np.zeros(m)
    for i in range(p):
        W_group[groups[i]-1] += W[i]
        
    return W_group

def group_lasso_LCD(X, knockoffs, y, groups, **kwargs):
    """ Calculates difference between average group Lasso signed maxs. 
    Does NOT use a group lasso regression class, unfortunately.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :param kwargs: kwargs for pyglmnet Lasso class
    """

    # Bind data
    n = X.shape[0]
    p = X.shape[1]
    m = np.unique(groups).shape[0]
    features = np.concatenate([X, knockoffs], axis = 1)
    doubled_groups = np.concatenate([groups, groups], axis = 0)

    # Fit the group lasso
    glasso = GLMCV(distr='gaussian', tol=1e-2,
                   group = doubled_groups,
                   alpha = 1.0)

    # Standardize
    y_std = (y - y.mean())/y.std()
    features = (features.copy() - features.mean())/features.std()
    glasso.fit(features, y_std)

    # Get coefficients, calculate LCD
    Z = gl.beta_
    W = calc_LCD(Z, groups)
    return W


def calc_simple_LCD(X, knockoffs, y, groups = None, **kwargs):
    """ Calculates Lasso coefficient difference NOT using group lasso.
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :params **kwargs: kwargs to Lasso method 
    """

    # Bind data
    n = X.shape[0]
    p = X.shape[1]
    features = np.concatenate([X, knockoffs], axis = 1)

    # Lasso
    lasso = linear_model.Lasso(**kwargs)
    lasso.fit(features, y) 
    

def calc_group_LCD(X, knockoffs, y, groups = None, **kwargs):
    """ Calculates group Lasso coefficient difference. 
    :param X: n x p design matrix
    :param knockoffs: n x p knockoff matrix
    :param groups: p length numpy array of groups
    :params **kwargs: kwargs to GroupLasso method 
    I.e. if features 2 and 3 are in the same group, 
    we set W(X, knockoffs, y) = 
    sum(abs coeff of X) - sum(abs coeff of knockoffs)
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

    # Randomly shuffle covariates to 
    # truly ensure antisymmetry (e.g. bc of initialization)
    # perm, inv_perm = random_permutation(2*p)
    # features = features.copy()[:, perm]
    # doubled_groups = doubled_groups.copy()[perm]
    
    # Fit model
    gl = GroupLasso(groups=doubled_groups, **kwargs)
    gl.fit(features, y.reshape(n, 1))
    Z = gl.coef_

    # Create LCD statistics
    W = calc_LCD(Z, groups)
    warnings.simplefilter("always")

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
    