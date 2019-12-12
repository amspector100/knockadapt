import numpy as np
from scipy import stats

# Utility functions
from statsmodels.stats.moment_helpers import cov2corr
from .utilities import force_positive_definite, chol2inv

# Tree methods
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd

# Graphing
import matplotlib.pyplot as plt


def AR1(p=30, a=1, b=1):
    """ Generates correlation matrix for AR(1) Gaussian process,
    where $Corr(X_t, X_{t-1})$ are drawn from Beta(a,b),
    independently for each t"""

    # Generate rhos, take log to make multiplication easier
    rhos = np.log(stats.beta.rvs(size=p, a=a, b=b))
    rhos[0] = 0

    # Log correlations between x_1 and x_i for each i
    cumrhos = np.cumsum(rhos).reshape(p, 1)
    # Use cumsum tricks to calculate all correlations
    log_corrs = -1 * np.abs(cumrhos - cumrhos.transpose())

    return np.exp(log_corrs)


def ErdosRenyi(p=300, delta=0.8, values=[-0.8, -0.3, -0.05, 0.05, 0.3, 0.8], tol=1e-3):
    """ Randomly samples bernoulli flags as well as values
    for partial correlations to generate sparse precision
    matrices."""

    # Initialization
    values = np.array(values)
    Q = np.zeros((p, p))
    triang_size = int((p ** 2 + p) / 2)

    # Sample the upper triangle
    mask = stats.bernoulli.rvs(delta, size=triang_size)
    vals = np.random.choice(values, size=triang_size, replace=True)
    triang = mask * vals

    # Set values and add diagonal
    upper_inds = np.triu_indices(p, 0)
    Q[upper_inds] = triang
    Q = np.dot(Q, Q.T)

    # Force to be positive definite -
    Q = force_positive_definite(Q, tol=tol)

    return Q


def daibarber2016_graph(n=3000, p=1000, m=None, k=20, rho=0.5, gamma=0, **kwargs):
    """ Same data-generating process as Dai and Barber 2016
    (see https://arxiv.org/abs/1602.03589). **kwargs are passed
    to sample_glm_response function.
    """

    # Set default values
    if m is None:
        m = int(p / 5)

    # Set k
    if k is None and p == 1000:
        k = 20
    else:
        k = int(m / 2)

    # Create groups
    groups = np.array([int(i / (p / m)) for i in range(p)])

    # Helper fn for covariance matrix
    # Add a tinnnyyyy bit of noise to make sure that the
    # cutoff method works properly
    def get_corr(g1, g2):
        if g1 == g2:
            return rho
        else:
            return gamma * rho

    get_corr = np.vectorize(get_corr)

    # Create correlation matrix, invert
    Xcoords, Ycoords = np.meshgrid(groups, groups)
    Sigma = get_corr(Xcoords, Ycoords)
    Sigma += np.eye(p) - np.diagflat(np.diag(Sigma))
    Q = chol2inv(Sigma)

    # Create beta
    chosen_groups = np.random.choice(np.unique(groups), k, replace=False)
    beta = np.array([3.5 if i in chosen_groups else 0 for i in groups])
    signs = 1 - 2 * stats.bernoulli.rvs(0.5, size=p)
    beta = beta * signs

    # Sample design matrix
    mu = np.zeros(p)
    X = stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=n)
    # Sample y
    y = sample_glm_response(X, beta, **kwargs)

    return X, y, beta, Q, Sigma, groups + 1


def create_correlation_tree(corr_matrix, method="single"):
    """ Creates hierarchical clustering (correlation tree)
    from a correlation matrix
    :param corr_matrix: the correlation matrix
    :param method: 'single', 'average', 'fro', or 'complete'
    returns: 'link' of the correlation tree, as in scipy"""

    # Distance matrix for tree method
    if method == "fro":
        dist_matrix = np.around(1 - np.power(corr_matrix, 2), decimals=7)
    else:
        dist_matrix = np.around(1 - np.abs(corr_matrix), decimals=7)
    dist_matrix -= np.diagflat(np.diag(dist_matrix))

    condensed_dist_matrix = ssd.squareform(dist_matrix)

    # Create linkage
    if method == "single":
        link = hierarchy.single(condensed_dist_matrix)
    elif method == "average" or method == "fro":
        link = hierarchy.average(condensed_dist_matrix)
    elif method == "complete":
        link = hierarchy.complete(condensed_dist_matrix)
    else:
        raise ValueError(
            f'Only "single", "complete", "average", "fro" are valid methods, not {method}'
        )

    return link


def create_sparse_coefficients(p, sparsity=0.5, groups=None, k=None, coeff_size=10):
    """ Randomly selects floor(p * sparsity) coefficients to be nonzero,
    which are then plus/minus coeff_size with equal probability.
    Alternatively, if groups and k are supplied, will choose k groups 
    to have nonzero coefficients, where each of the k groups has a 
    coefficient size of plus/minus 10."""

    # First, decide which coefficients are nonzero, one of two methods
    if groups is not None:
        if k is None:
            raise ValueError(
                "To choose group coefficients, must supply 'k' arg, not just groups"
            )
        m = np.unique(groups).shape[0]
        if k > m:
            raise ValueError(
                f"Number of nonzero groups k = {k} is greater than num unique groups {m}"
            )

        # Method one: a certain number of groups are nonzero
        chosen_groups = np.random.choice(np.unique(groups), k, replace=False)
        beta = np.array([coeff_size if i in chosen_groups else 0 for i in groups])

    else:
        if k is not None:
            raise ValueError(
                "To choose group coefficients, must supply 'groups' arg, not just k"
            )

        # Method two (default): a certain percentage of coefficients are nonzero
        num_nonzero = int(np.floor(sparsity * p))
        beta = np.array([coeff_size] * num_nonzero + [0] * (p - num_nonzero))
        np.random.shuffle(beta)

    # Now add random signs
    signs = 1 - 2 * stats.bernoulli.rvs(0.5, size=p)
    beta = beta * signs
    return beta


def sample_glm_response(X, beta, y_dist="gaussian"):
    """ Given a design matrix and coefficients (beta), samples
    a response y """

    n = X.shape[0]

    # Create y, one of two families
    if y_dist == "gaussian":
        y = np.dot(X, beta) + np.random.standard_normal((n))
    elif y_dist == "binomial":
        inner_product = np.dot(X, beta)
        probs = 1 / (1 + np.exp(-1 * inner_product))
        y = stats.bernoulli.rvs(probs)
    else:
        raise ValueError(f"y_dist must be one of 'gaussian', 'binomial', not {y_dist}")

    return y


def sample_data(
    p=100,
    n=50,
    method="ErdosRenyi",
    Q=None,
    corr_matrix=None,
    beta=None,
    coeff_size=1,
    sparsity=0.5,
    k=None,
    groups=None,
    y_dist="gaussian",
    **kwargs,
):
    """ Creates a random covariance matrix using method
    and then samples Gaussian data from it. It also creates
    a linear response y with sparse coefficients.
    :param p: Dimensionality
    :param n: Sample size
    :param coeff_size: The standard deviation of the
    sparse linear coefficients. (The noise of the 
    response itself is standard normal).
    :param method: How to generate the covariance matrix.
    One of 'ErdosRenyi', 'AR1', 'identity', 'daibarber2016'
    :param Q: p x p precision matrix. If supplied, will not generate
    a new covariance matrix.
    :param corr_matrix: p x p correlation matrix. If supplied, will 
    not generate a new correlation matrix.
    :param str y_dist: one of 'gaussian' or 'binomial', used to 
    generate the response. (If 'binomial', uses logistic link fn). 
    :param kwargs: kwargs to the graph generator (e.g. AR1 kwargs).
    """

    # Create Graph
    if Q is None and corr_matrix is None:

        method = str(method).lower()
        if method == "erdosrenyi":
            Q = ErdosRenyi(p=p, **kwargs)
            corr_matrix = cov2corr(chol2inv(Q))
            corr_matrix -= np.diagflat(np.diag(corr_matrix))
            corr_matrix += np.eye(p)
            Q = chol2inv(corr_matrix)
        elif method == "ar1":
            corr_matrix = AR1(p=p, **kwargs)
            Q = chol2inv(corr_matrix)
        elif method == "identity":
            corr_matrix = 1e-3 * stats.norm.rvs(size=(p, p))
            corr_matrix = np.dot(corr_matrix.T, corr_matrix)
            corr_matrix -= np.diagflat(np.diag(corr_matrix))
            corr_matrix += np.eye(p)
            Q = corr_matrix
        elif method == "daibarber2016":
            _, _, beta, Q, corr_matrix, _ = daibarber2016_graph(p=p, n=n, **kwargs)
        else:
            raise ValueError("Other methods not implemented yet")

    elif Q is None:
        Q = chol2inv(corr_matrix)
    elif corr_matrix is None:
        corr_matrix = cov2corr(chol2inv(Q))
    else:
        pass

    # Create sparse coefficients
    if beta is None:
        beta = create_sparse_coefficients(
            p=p, sparsity=sparsity, coeff_size=coeff_size, k=k, groups=groups
        )

    # Sample design matrix
    mu = np.zeros(p)
    X = stats.multivariate_normal.rvs(mean=mu, cov=corr_matrix, size=n)

    y = sample_glm_response(X=X, beta=beta, y_dist=y_dist)

    return X, y, beta, Q, corr_matrix


def plot_dendrogram(link, title=None):

    # Get title
    if title is None:
        title = "Hierarchical Clustering Dendrogram"

    # Plot
    plt.figure(figsize=(15, 10))
    plt.title(str(title))
    plt.xlabel("Index")
    plt.ylabel("Correlation Distance")
    hierarchy.dendrogram(
        link,
        leaf_rotation=90.0,  # rotates the x axis labels
        leaf_font_size=8.0,  # font size for the x axis labels
    )
    plt.show()
