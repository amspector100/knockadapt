import warnings
import numpy as np
import scipy as sp
from scipy import stats

# Utility functions
from statsmodels.stats.moment_helpers import cov2corr
from .utilities import force_positive_definite, chol2inv

# Tree methods
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd

# Graphing
import matplotlib.pyplot as plt

def BandPrecision(p = 500, a = 0.9, rho = 5, c = 1.5):
    """ Generates band precision matrix - DOES NOT work yet
    :param p: number of features
    :param a: decay exponent base
    :param rho: cutoff after which precision matrix entries are zero
    :param c: decay constant"""
    
    vert_dists = np.repeat(np.arange(0, p, 1), p).reshape(p, p)
    dists = np.abs(vert_dists - vert_dists.transpose())
    Q = np.sign(a) * (a**(dists/c)) * (dists <= rho)
    return Q

def AR1(p = 30, a = 1, b = 1):
    """ Generates correlation matrix for AR(1) Gaussian process,
    where $Corr(X_t, X_{t-1})$ are drawn from Beta(a,b),
    independently for each t"""
        
    # Generate rhos, take log to make multiplication easier
    rhos = np.log(stats.beta.rvs(size = p, a = a, b = b))
    rhos[0] = 0 
    
    # Log correlations between x_1 and x_i for each i
    cumrhos = np.cumsum(rhos).reshape(p, 1)
    # Use cumsum tricks to calculate all correlations
    log_corrs = -1 * np.abs(cumrhos - cumrhos.transpose())
    
    return np.exp(log_corrs)

def ErdosRenyi(p = 300, delta = 0.8, 
               values = [-0.8, -0.3, -0.05, 0.05, 0.3, 0.8],
               tol = 1e-3):
    """ Randomly samples bernoulli flags as well as values
    for partial correlations to generate sparse precision
    matrices."""
    
    # Initialization
    values = np.array(values)
    Q = np.zeros((p, p))
    triang_size = int((p**2 + p)/2)
    
    # Sample the upper triangle
    mask = stats.bernoulli.rvs(delta, size = triang_size)
    vals = np.random.choice(values, size = triang_size, replace = True)
    triang = mask * vals
    
    # Set values and add diagonal
    upper_inds = np.triu_indices(p, 0)
    Q[upper_inds] = triang
    Q = np.dot(Q, Q.T)
    
    # Force to be positive definite - 
    Q = force_positive_definite(Q, tol = tol)
    
    return Q

def clearGroups(p = 300, rho = 0.6, gamma = 0.3):
    """
    Construct covariance matrix as in Dai and Barber (2016).
    I.e. within group correlation is clear, """
    pass

def create_correlation_tree(corr_matrix, method = 'single'):
    """ Creates hierarchical clustering (correlation tree)
    from a correlation matrix
    :param corr_matrix: the correlation matrix
    :param method: 'single', 'average', or 'complete
    
    returns: 'link' of the correlation tree, as in scipy"""
    
    p = corr_matrix.shape[0]
    
    # Distance matrix for tree method
    dist_matrix = np.around(1-np.abs(corr_matrix), decimals = 10)
    condensed_dist_matrix = ssd.squareform(dist_matrix)

    # Create linkage
    if method == 'single':
        link = hierarchy.single(condensed_dist_matrix)
    elif method == 'average':
        link = hierarchy.average(condensed_dist_matrix)
    elif method == 'complete':
        link = hierarchy.complete(condensed_dist_matrix)
    else:
        raise ValueError('Only "single", "complete", "average" are valid methods')
        
        
    return link

def sample_data(p = 100, n = 50, coeff_size = 1, 
                sparsity = 0.5, method = 'ErdosRenyi',
               **kwargs):
    """ Creates a random covariance matrix using method
    and then samples Gaussian data from it. It also creates
    a linear response y with sparse coefficients.
    :param p: Dimensionality
    :param n: Sample size
    :param coeff_size: The standard deviation of the
    sparse linear coefficients. (The noise of the 
    response itself is standard normal).
    :param method: How to generate the covariance matrix.
    One of 'ErdosRenyi', 'AR1', 'identity'.
    :param kwargs: kwargs to the graph generator (e.g. AR1 kwargs).
    """
    
    # Create Graph
    method = str(method).lower()
    if method == 'erdosrenyi':
        Q = ErdosRenyi(p = p, **kwargs)
        corr_matrix = cov2corr(chol2inv(Q))
    elif method == 'ar1':
        corr_matrix = AR1(p = p, **kwargs)
        Q = chol2inv(corr_matrix)
    elif method == 'identity':
        corr_matrix = 1e-3 * stats.norm.rvs(size = (p,p))
        corr_matrix = np.dot(corr_matrix.T, corr_matrix)
        corr_matrix -= np.diagflat(np.diag(corr_matrix))
        corr_matrix += np.eye(p)
        Q = corr_matrix
    else:
        raise ValueError("Other methods not implemented yet")

    # Sample design matrix
    mu = np.zeros(p)
    X = stats.multivariate_normal.rvs(mean = mu, cov = corr_matrix, size = n)

    # Create sparse coefficients and y
    num_nonzero = int(np.floor(sparsity * p))
    mask = np.array([0]*num_nonzero + [1]*(p-num_nonzero))
    np.random.shuffle(mask)
    signs = 1 - 2*stats.bernoulli.rvs(sparsity, size = p)
    beta = coeff_size * mask * signs
    y = np.einsum('np,p->n', X, beta) + np.random.standard_normal((n))
    
    return X, y, beta, Q, corr_matrix

def plot_dendrogram(link, title = None):

    # Get title
    if title is None:
        title = 'Hierarchical Clustering Dendrogram'

    # Plot
    plt.figure(figsize=(15, 10))
    plt.title(str(title))
    plt.xlabel('Index')
    plt.ylabel('Correlation Distance')
    hierarchy.dendrogram(
        link,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()
