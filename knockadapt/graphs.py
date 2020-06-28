import numpy as np
from scipy import stats

# Utility functions
from . import utilities
from .utilities import shift_until_PSD, chol2inv, cov2corr

# Tree methods
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd

# Graphing
import matplotlib.pyplot as plt

def Wishart(d=100, p=100, tol=1e-2):
    """
    Let W be a random d x p matrix with i.i.d. Gaussian
    entries. Then Sigma = cov2corr(W^T W).
    """

    W = np.random.randn(d, p)
    V = np.dot(W.T, W)
    V = cov2corr(V)
    return cov2corr(shift_until_PSD(V, tol=tol))

def UniformDot(d=100, p=100, tol=1e-2):
    """
    Let U be a random d x p matrix with i.i.d. uniform
    entries. Then Sigma = cov2corr(U^T U)
    """
    U = np.random.uniform(size=(d, p))
    V = np.dot(U.T, U)
    V = cov2corr(V)
    return cov2corr(shift_until_PSD(V, tol=tol))

def DirichletCorr(p=100, temp=1, tol=1e-6):
    """
    Generates a correlation matrix following
    Davies, Philip I; Higham, Nicholas J;
     “Numerically stable generation of correlation matrices and their factors”, 
     BIT 2000, Vol. 40, No. 4, pp. 640 651
     using the scipy implementation.
     We set the eigenvalues using a dirichlet.
     The p dirichlet parameters are i.i.d. uniform [0, temp].
    """
    alpha = np.random.uniform(temp, size=p)
    d = stats.dirichlet(alpha=alpha).rvs().reshape(-1)

    # We have to round to prevent errors from random_correlation,
    # which is supperr sensitive to d.sum() != p even when the 
    # error is a floating point error.
    d = np.around(d+tol, 6)
    d = p * d / d.sum()
    d[0] += p - d.sum() # This is like 1e-10 but otherwise throws an error

    # Create and return matrix
    V = stats.random_correlation.rvs(d)
    return V


def AR1(p=30, a=1, b=1, tol=1e-3, rho=None):
    """ Generates correlation matrix for AR(1) Gaussian process,
    where $Corr(X_t, X_{t-1})$ are drawn from Beta(a,b),
    independently for each t. 
    If rho is specified, then $Corr(X_t, X_{t-1}) = rho
    for all t."""

    # Generate rhos, take log to make multiplication easier
    if rho is None:
        rhos = np.log(stats.beta.rvs(size=p, a=a, b=b))
    else:
        if np.abs(rho) > 1:
            raise ValueError(f"rho {rho} must be a correlation between -1 and 1")
        rhos = np.log(np.array([rho for _ in range(p)]))
    rhos[0] = 0

    # Log correlations between x_1 and x_i for each i
    cumrhos = np.cumsum(rhos).reshape(p, 1)

    # Use cumsum tricks to calculate all correlations
    log_corrs = -1 * np.abs(cumrhos - cumrhos.transpose())
    corr_matrix = np.exp(log_corrs)

    # Ensure PSD-ness
    corr_matrix = cov2corr(shift_until_PSD(corr_matrix, tol))

    return corr_matrix


def ErdosRenyi(p=300, delta=0.8, values=[-0.8, -0.3, -0.05, 0.05, 0.3, 0.8], tol=1e-1):
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
    Q = shift_until_PSD(Q, tol=tol)

    return Q

def TrueErdosRenyi(p=300, delta=0.2, lower=0.1, upper=1, tol=1e-1):
    """ Randomly samples bernoulli flags as well as values
    for partial correlations to generate sparse precision
    matrices. Follows https://arxiv.org/pdf/1908.11611.pdf.
    """

    # Initialization
    V = np.zeros((p, p))
    triang_size = int((p ** 2 + p) / 2)

    # Sample the upper triangle
    mask = stats.bernoulli.rvs(delta, size=triang_size)
    vals = np.random.uniform(lower, upper, size=triang_size)
    flags = 2*np.random.binomial(1, 0.5, triang_size) - 1
    triang = mask * flags * vals

    # Set values and add diagonal
    upper_inds = np.triu_indices(p, 0)
    V[upper_inds] = triang
    V = V + V.T
    V += np.eye(p) - np.diag(np.diag(V))

    # Force to be positive definite -
    V = shift_until_PSD(V, tol=tol)

    return V


def daibarber2016_graph(
    n=3000,
    p=1000,
    group_size=5,
    sparsity=0.1,
    rho=0.5,
    gamma=0,
    coeff_size=3.5,
    coeff_dist=None,
    sign_prob=0.5,
    beta=None,
    mu=None,
    **kwargs,
):
    """ Same data-generating process as Dai and Barber 2016
    (see https://arxiv.org/abs/1602.03589). 
    :param int group_size: The size of groups. Defaults to 5.
    :param int sparsity: The proportion of groups with nonzero effects.
     Defaults to 0.1 (the daibarber2016 default).
    :param rho: Within-group correlation
    :param gamma: The between-group correlation = rho * gamma
    :param beta: If supplied, the linear response
    :param mu: The p-dimensional mean of the covariates. Defaults to 0.
    :param **kwargs: Args passed to sample_response function.
    """

    # Set default values
    num_groups = int(p / group_size)

    # Create groups
    groups = np.array([int(i / (p / num_groups)) for i in range(p)])

    # Helper fn for covariance matrix
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
    if beta is None:
        beta = create_sparse_coefficients(
            p=p,
            sparsity=sparsity,
            coeff_size=coeff_size,
            groups=groups,
            sign_prob=sign_prob,
            coeff_dist=coeff_dist,
        )

    # Sample design matrix
    if mu is None:
        mu = np.zeros(p)
    X = stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=n)
    # Sample y
    y = sample_response(X, beta, **kwargs)

    return X, y, beta, Q, Sigma, groups + 1

def create_sparse_coefficients(
    p, sparsity=0.5, groups=None, coeff_size=1, coeff_dist=None, sign_prob=0.5,
):
    """ Randomly selects floor(p * sparsity) coefficients to be nonzero,
    which are then plus/minus coeff_size with equal probability.
    :param p: Dimensionality of coefficients
    :type p: int
    :param sparsity: Sparsity of selection
    :type sparsity: float
    :param coeff_size: Non-zero coefficients are set to +/i coeff_size
    :type coeff_size: float
    :param groups: Allows the possibility of grouped signals. 
    If not None supplied, will choose 
    floor(sparsity * num_groups) groups, 
    where each feature in the selected groups will 
    have a nonzero coefficient.
    :type groups: np.ndarray
    :param sign_prob: The probability that each nonzero coefficient
    will be positive. (Signs are assigned independently.)
    :param coeff_dist: Three options:
        - If None, all coefficients have absolute value 
    coeff_size.
        - If "normal", all nonzero coefficients are drawn
    from N(coeff_size, 1). 
        - If "uniform", drawn from Unif(coeff_size/2, coeff_size).
    :return: p-length numpy array of sparse coefficients"""

    # First, decide which coefficients are nonzero, one of two methods
    if groups is not None:

        # Method one: a certain number of groups are nonzero
        num_groups = np.unique(groups).shape[0]
        num_nonzero_groups = int(np.floor(sparsity * num_groups))
        chosen_groups = np.random.choice(
            np.unique(groups), num_nonzero_groups, replace=False
        )
        beta = np.array([coeff_size if i in chosen_groups else 0 for i in groups])

    else:

        # Method two (default): a certain percentage of coefficients are nonzero
        num_nonzero = int(np.floor(sparsity * p))
        beta = np.array([coeff_size] * num_nonzero + [0] * (p - num_nonzero))
        np.random.shuffle(beta)

    # Now draw random signs
    signs = 1 - 2 * stats.bernoulli.rvs(sign_prob, size=p)

    # Possibly change the absolute values of beta
    if coeff_dist is not None:
        beta_nonzeros = beta != 0
        if str(coeff_dist).lower() == "normal":
            beta = (beta + np.random.randn(p)) * beta_nonzeros
        elif str(coeff_dist).lower() == "uniform":
            beta = beta * np.random.uniform(size=p) / 2 + beta / 2
        elif str(coeff_dist).lower() == "none":
            pass
        else:
            raise ValueError(
                f"coeff_dist ({coeff_dist}) must be 'none', 'normal', or 'uniform'"
            )

    beta = beta * signs
    return beta


def sample_response(X, beta, cond_mean='linear', y_dist="gaussian"):
    """ Given a design matrix and coefficients (beta), samples
    a response y.
    :param cond_mean: How to calculate the conditional mean. Four options:
        1. Linear (np.dot(X, beta))
        2. Cubic (np.dot(X**3, beta) - np.dot(X, beta))
        3. trunclinear ((X * beta >= 1).sum(axis = 1))
        Stands for truncated linear.
        4. pairint: pairs up non-null coefficients according to the
        order of beta, multiplies them and their beta values, then
        sums. "pairint" stands for pairwise-interactions.
    :param y_dist: If gaussian, y is the conditional mean plus
    gaussian noise. If binomial, Pr(y=1) = softmax(cond_mean). """

    n = X.shape[0]
    p = X.shape[1]

    if cond_mean == 'linear':
        cond_mean = np.dot(X, beta)
    elif cond_mean == 'quadratic':
        cond_mean = np.dot(np.power(X, 2), beta)
    elif cond_mean == 'cubic':
        cond_mean = np.dot(np.power(X, 3), beta) - np.dot(X, beta)
    elif cond_mean == 'trunclinear':
        cond_mean = ((X * beta >= 1) * np.sign(beta)).sum(axis = 1)
    elif cond_mean == 'cos':
        cond_mean = (np.sign(beta)*(beta!=0)*np.cos(X)).sum(axis=1)
    elif cond_mean == 'pairint':
        # Pair up the coefficients
        pairs = [[]]
        for j in range(p):
            if beta[j] != 0:
                if len(pairs[-1]) < 2:
                    pairs[-1].append(j)
                else:
                    pairs.append([j])

        # Multiply pairs and add to cond_mean
        cond_mean = np.zeros(n)
        for pair in pairs:
            interaction_term = np.ones(n)
            for j in pair:
                interaction_term = interaction_term * beta[j]
                interaction_term = interaction_term * X[:, j]
            cond_mean += interaction_term
    else:
        raise ValueError(f"cond_mean must be one of 'linear', 'quadratic', cubic', 'trunclinear', 'cos', 'pairint', not {cond_mean}")

    # Create y, one of two families
    if y_dist == "gaussian":
        y = cond_mean + np.random.standard_normal((n))
    elif y_dist == "binomial":
        probs = 1 / (1 + np.exp(-1 * cond_mean))
        y = stats.bernoulli.rvs(probs)
    else:
        raise ValueError(f"y_dist must be one of 'gaussian', 'binomial', not {y_dist}")

    return y

def sample_ar1t(
    rhos,
    n=50,
    df_t=5, 
):
    """
    Samples t variables according to a Markov chain.
    """
    # Initial t samples
    p = rhos.shape[0] + 1
    tvars = stats.t(df=df_t).rvs(size=(n,p))

    # Initialize X
    X = np.zeros((n, p))
    scale = np.sqrt((df_t - 2) / df_t)
    X[:, 0] = scale * tvars[:, 0]

    # Loop through variables according to markov chain
    conjugates = np.sqrt(1 - rhos**2)
    for j in range(1, p):
        X[:,j] = rhos[j-1]*X[:,j-1] + conjugates[j-1]*scale*tvars[:,j]

    return X

def cov2blocks(V, tol=1e-5):
    """
    Decomposes a PREORDERED block-diagonal matrix V
    into its blocks.
    """
    p = V.shape[0]
    blocks = []
    block_start = 0
    block_inds = []
    for j in range(p+1):
        # Detect if we have exited the block
        if j == p:
            blocks.append(V[block_start:j, block_start:j])
            block_inds.append(list(range(block_start, j)))
        elif np.abs(V[block_start, j]) < tol:
            # If so, reset the block_start
            blocks.append(V[block_start:j, block_start:j])
            block_inds.append(list(range(block_start, j)))
            block_start = j

    return blocks, block_inds

def sample_block_tmvn(
    blocks,
    block_sqrts=None,
    n=50,
    df_t=3,
):
    """
    Samples a block-diagonal multivariate t from a set of 
    blocks. If these blocks have diagonal 1, the marginal
    variances of the t-distribution will be 1 as well.
    :param blocks: A list of square, PSD numpy arrays. These
    are the block covariance matrices.
    :param n: The number of data points to sample
    :param block_sqrts: Possibly pass in the list of sqrts 
    of the blocks for more efficiently sampling. By default
    uses cholesky square roots.
    :param df_t: The degrees of freedom of the t-distribution
    """

    # Possibly calculate sqrt roots
    if block_sqrts is None:
        block_sqrts = [np.linalg.cholesky(block) for block in blocks]
    if len(block_sqrts) != len(blocks):
        raise ValueError(f"Blocks and block_sqrts must have same length")

    # Loop through blocks and sample multivariate t
    X = []
    for i, block_sqrt in enumerate(block_sqrts):
        # Dimensionality and also sample chisquares
        p_block = block_sqrt.shape[0]
        chi_block = stats.chi2.rvs(df=df_t, size=(n,1))

        # Linear transformatino + chi square multiplication
        Z_block = np.random.randn(n,p_block) # n x p 
        t_block = np.dot(Z_block, block_sqrt.T) # n x p
        t_block = np.sqrt(df_t / chi_block) * t_block
        X.append(t_block)

    #  Append back together and scale such that variance is 1
    scale = np.sqrt((df_t - 2) / (df_t))
    X = scale * np.concatenate(X, axis=1)
    return X

def num2coords(i, gridwidth = 10):
    """ Coordinates of variable i in the grid
    :param i: Position of variable in ordering
    :param width: Width of the grid
    :returns: length_coord, width_coord (coordiantes)"""
    length_coord = i % gridwidth
    width_coord = i // gridwidth
    return int(length_coord), int(width_coord)
    
def coords2num(l, w, gridwidth = 10):
    """ Takes coordinates of variable in the grid, returns position"""
    if l < 0 or w < 0:
        return -1
    if l >= gridwidth or w >= gridwidth:
        return -1
    return int(w * gridwidth + l)

def sample_gibbs(
        n, p, method='ising', temp=1, num_iter=15, K=20, max_val=2.5,
    ):
    """ Samples from a Gibbs measure on a square grid
    using a Gibbs sampler."""

    # Create buckets from (approximately) -max_val to max_val
    buckets = np.arange(-K+1, K+1, 2) / (K/max_val)

    # Infer dimensionality
    gridwidth = int(np.sqrt(p))
    variables = set(list(range(p)))
    Q = np.zeros((p, p)) # The UGM

    # Log potentials and cliques
    def log_potential(X1, X2=None, temp=1):
        if X2 is None:
            X2 = X1[:, 1]
            X1 = X1[:, 0]
        return -1*temp*np.power(X1 - X2, 2)

    # Construct cliques
    clique_dict = {}
    for i1 in range(p):
        clique_dict[i1] = []
        # For ising model
        if method=='ising':
            lc, wc = num2coords(i1, gridwidth=gridwidth)
            for ladd in [-1, 1]:
                i2 = coords2num(lc + ladd, wc, gridwidth=gridwidth)
                if i2 != -1:
                    clique_dict[i1].append((i1, i2))
                    sign = 1 - 2*np.random.binomial(1, 0.5)
                    Q[i1, i2] = temp * sign
                    Q[i2, i1] = temp * sign
                else:
            for wadd in [-1, 1]:
                i2 = coords2num(lc, wc + wadd, gridwidth=gridwidth)
                if i2 != -1:
                    print(i1,i2, lc, wc)
                    clique_dict[i1].append((i1, i2))
                    sign = 1 - 2*np.random.binomial(1, 0.5)
                    Q[i1, i2] = temp * sign
                    Q[i2, i1] = temp * sign
        # Otherwise method must be an integer:
        # we randomly connect this variable method others
        else:
            choices = list(variables.difference(set([i1])))
            connections = np.random.choice(choices, method, replace=False)
            for i2 in connections:
                clique_dict[i1].append((i1, i2))
                sign = 1 - 2*np.random.binomial(1, 0.5)
                Q[i1, i2] = temp * sign
                Q[i2, i1] = temp * sign
    print(clique_dict)


    # Initialize
    X = np.random.randn(n, p, 1)
    dists = np.abs(X - buckets.reshape(1, 1, -1))
    indices = dists.argmin(axis=-1)
    X = buckets[indices]

    # Marginalize / gibbs sample
    for _ in range(num_iter):
        for j in range(p):
            # Cliques and marginals for this node
            cliques = clique_dict[j]
            marginals = np.zeros((n, K))
            for clique in cliques:
                # Calculate log-potential from this clique
                X1 = buckets.reshape(1, -1)
                X2 = X[:, clique[-1]].reshape(-1, 1)
                marginals += log_potential(
                    X1=X1, X2=X2, temp=Q[j, clique[-1]]
                )

            # Resample distribution of this value of X
            # Note the exp is a bottleneck here for efficiency
            marginals = np.exp(marginals.astype(np.float32))
            marginals = marginals / marginals.sum(axis=-1, keepdims=True)
            marginals = marginals.cumsum(axis=-1)

            # Batched multinomial sampling using uniforms
            # (faster than for loops + numpy)
            unifs = np.random.uniform(size=(n,1))
            Xnew = buckets[np.argmax(unifs <= marginals, axis=-1)]
            X[:, j] = Xnew

    return X, Q

def sample_data(
    p=100,
    n=50,
    method="AR1",
    mu=None,
    corr_matrix=None,
    Q=None,
    beta=None,
    coeff_size=1,
    coeff_dist=None,
    sparsity=0.5,
    groups=None,
    x_dist="gaussian",
    y_dist="gaussian",
    df_t=3,
    cond_mean='linear',
    sign_prob=0.5,
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
    :param mu: If supplied, a p-dimensional vector of means for
    the covariates. Defaults to zero.
    :param Q: p x p precision matrix. If supplied, will not generate
    a new covariance matrix.
    :param corr_matrix: p x p correlation matrix. If supplied, will 
    not generate a new correlation matrix.
    :param str x_dist: one of 'gaussian' or 'ar1t', used to generate
    the input data. If 'ar1t', method must be 'ar1'.
    :param str y_dist: one of 'gaussian' or 'binomial', used to 
    generate the response. (If 'binomial', uses logistic link fn). 
    :param kwargs: kwargs to the graph generator (e.g. AR1 kwargs).
    returns: X, y, beta, Q, corr_matrix
    Note that Q will be a precision matrix unless x_dist=='ising' 
    or 'gibbs': in this case, Q will be the UGM structure of the graph.
    """

    # Defaults
    if mu is None:
        mu = np.zeros(p)

    # Ising / Gibbs Sampling
    if x_dist == 'gibbs':
        # Sample X, Q
        X, Q = sample_gibbs(n=n, p=p, method=method, **kwargs)

        # Normalize for consistency
        V = np.cov(X.T)
        scale = np.sqrt(np.diag(V))
        X = X / scale
        corr_matrix = utilities.cov2corr(V)

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
        elif method == "daibarber2016":
            _, _, beta, Q, corr_matrix, _ = daibarber2016_graph(
                p=p,
                n=n,
                coeff_size=coeff_size,
                coeff_dist=coeff_dist,
                sparsity=sparsity,
                sign_prob=sign_prob,
                beta=beta,
                **kwargs,
            )
        elif method == 'ver':
            corr_matrix = cov2corr(TrueErdosRenyi(p=p, **kwargs))
            Q = chol2inv(corr_matrix)
        elif method == 'qer':
            Q = TrueErdosRenyi(p=p, **kwargs)
            corr_matrix = cov2corr(chol2inv(Q))
            Q = chol2inv(corr_matrix)
        elif method == 'dirichlet':
            corr_matrix = DirichletCorr(p=p, **kwargs)
            Q = chol2inv(corr_matrix)
        elif method == 'wishart':
            corr_matrix = Wishart(p=p, **kwargs)
            Q = chol2inv(corr_matrix)
        elif method == 'uniformdot':
            corr_matrix = UniformDot(p=p, **kwargs)
            Q = chol2inv(corr_matrix)
        else:
            raise ValueError(f"Other method {method} not implemented yet")

    elif Q is None:
        Q = chol2inv(corr_matrix)
    elif corr_matrix is None:
        corr_matrix = cov2corr(chol2inv(Q))
    else:
        pass

    # Create sparse coefficients
    if beta is None:
        beta = create_sparse_coefficients(
            p=p,
            sparsity=sparsity,
            coeff_size=coeff_size,
            groups=groups,
            sign_prob=sign_prob,
            coeff_dist=coeff_dist,
        )

    # Sample design matrix
    if x_dist == 'gibbs':
        pass
    elif x_dist == 'gaussian':
        X = stats.multivariate_normal.rvs(mean=mu, cov=corr_matrix, size=n)
    elif x_dist == 'ar1t':
        if str(method).lower() != 'ar1':
            raise ValueError(f"For x_dist={x_dist}, method ({method}) should equal 'ar1'")
        X = sample_ar1t(n=n, rhos=np.diag(corr_matrix, 1), df_t=df_t)
    elif x_dist == 'blockt':
        blocks, _ = cov2blocks(corr_matrix)
        X = sample_block_tmvn(blocks, n=n, df_t=df_t)
    else:
        raise ValueError(f"x_dist must be one of 'gaussian', 'gibbs', 'ar1t', 'blockt'")

    # Sample y
    y = sample_response(X=X, beta=beta, y_dist=y_dist, cond_mean=cond_mean)

    return X, y, beta, Q, corr_matrix



def create_correlation_tree(corr_matrix, method="average"):
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
