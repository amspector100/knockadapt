""" 
	The metropolized knockoff sampler for an arbitrary probability density
	and graphical structure using covariance-guided proposals.

	See https://arxiv.org/abs/1903.00434 for a description of the algorithm
	and proof of validity and runtime.

	Initial author: Stephen Bates, October 2019.
	Adapted by: Asher Spector, June 2020. 
"""

# The basics
import sys
import numpy as np
import scipy as sp
import itertools
from scipy import stats
from . import utilities, knockoffs, graphs

# Network and UGM tools
import networkx as nx
from . import tree_processing

# Logging
import warnings
from tqdm import tqdm

def gaussian_log_likelihood(
	X, mu, var
):
	"""
	Somehow this is faster than scipy
	"""
	result = -1*np.power(X - mu, 2) / (2 * var)
	result += np.log(1 / np.sqrt(2 * np.pi * var))
	return result	

def t_log_likelihood(
	X, df_t
):
	"""
	UNNORMALIZED t loglikelihood.
	This is also faster than scipy
	"""
	result = np.log(1 + np.power(X, 2) / df_t)
	result = -1 * result * (df_t + 1) / 2
	return result	


class MetropolizedKnockoffSampler():

	def __init__(
			self,
			lf,
			X,
			mu,
			V,
			undir_graph=None,
			order=None,
			active_frontier=None,
			gamma=0.999,
			metro_verbose=False,
			cliques=None,
			log_potentials=None,
			**kwargs
			):
		"""
		A metropolized knockoff sampler for arbitrary random variables
		using covariance-guided proposals.

		Currently not implemented for group knockoffs.

		:param lf: Log-probability density. This function should take
		an n x p numpy array (n independent samples of a p-dim vector).
		:param X: n x p array of data. (n independent samples of a 
		p-dim vector).
		:param mu: The mean of X. As described in
		https://arxiv.org/abs/1903.00434, exact FDR control is maintained
		even when this vector is incorrect.
		:param V: The covariance matrix of X. As described in
		https://arxiv.org/abs/1903.00434, exact FDR control is maintained
		even when this covariance matrix is incorrect.
		:param undir_graph: A undirected graph specifying the conditional independence
		structure of the data-generating process. This must be specified 
		if either of the ``order`` or ``active_frontier`` params
		are not specified. One of two options:
		- A networkx undirected graph object
		- A p x p numpy array, where nonzero elements represent edges
		:param order: A p-length numpy array specifying the ordering
		to sample the variables. Should be a vector with unique
		entries 0,...,p-1.
		:param active_fontier: A list of lists of length p where
		entry j is the set of entries > j that are in V_j. This specifies
		the conditional independence structure of the distribution given by
		lf. See page 34 of the paper.
		:param gamma: A tuning parameter to increase / decrease the acceptance
		ratio. See appendix F.2.
		:param kwargs: kwargs to pass to a gaussian_knockoffs constructor.
		This is used to create the S matrix which is then used for the
		covariance-guided proposals.
		"""

		# Random params
		self.n = X.shape[0]
		self.p = X.shape[1]
		self.gamma = gamma
		self.metro_verbose = metro_verbose # Controls verbosity
		self.F_queries = 0 # Counts how many queries we make

		# Possibly learn order / active frontier
		if order is None or active_frontier is None:
			if undir_graph is None:
				raise ValueError(
					f"If order OR active_frontier are not provided, you must specify the undir_graph"
				)
			# Convert to nx
			if isinstance(undir_graph, np.ndarray):
				undir_graph = nx.Graph(undir_graph != 0)
			# Run junction tree algorithm
			self.width, self.T = tree_processing.treewidth_decomp(undir_graph)
			order, active_frontier = tree_processing.get_ordering(self.T)

		# Undirected graph must be existent in this case
		Q = utilities.chol2inv(V)
		if undir_graph is not None:
			warnings.filterwarnings('ignore')
			mask = nx.to_numpy_matrix(undir_graph)
			warnings.resetwarnings()
			mask += np.eye(self.p)
			# Handle case where the graph is dense
			if (mask == 0).sum() > 0:
				max_nonedge = np.max(np.abs(Q[mask == 0]))
				if max_nonedge > 1e-3:
					raise ValueError(
						f"Precision matrix Q is not compatible with undirected graph (nonedge has value {max_nonedge})"
					)

		# Save order and inverse order
		self.order = order
		self.inv_order = order.copy()
		for i, j in enumerate(order):
			self.inv_order[j] = i

		# Re-order the variables: the log-likelihood
		# function (lf) is reordered in a separate method
		self.X = X[:, self.order].astype(np.float32)
		self.unordered_lf = lf

		# Re-order the cliques 
		self.log_potentials = log_potentials
		if cliques is not None:
			self.cliques = []
			for clique in cliques:
				self.cliques.append(self.inv_order[clique])
		else:
			self.cliques = None

		# Create clique dictionaries. This maps variable i
		# to a list of two-length tuples. 
		#   - The first element is the clique_key, which can 
		#     be used to index into log_potentials.
		#	- The second element is the actual clique.
		if self.cliques is not None and self.log_potentials is not None:
			self.clique_dict = {j:[] for j in range(self.p)}
			for clique_key, clique in enumerate(self.cliques):
				for j in clique:
					self.clique_dict[j].append((clique_key, clique))
		else:
			self.clique_dict = None

		# Re-order active frontier
		self.active_frontier = []
		for i in range(len(order)):
			self.active_frontier += [
				[self.inv_order[j] for j in active_frontier[i]]
			]

		# Re-order mu
		self.mu = mu[self.order].reshape(1, -1)

		# If mu == 0, then we can save lots of time
		if np.all(self.mu == 0):
			self._zero_mu_flag = True 
		else:
			self._zero_mu_flag = False

		# Re-order sigma
		self.V = V[self.order][:, self.order]
		self.Q = Q[self.order][:, self.order]

		# Possibly reorder S if it's in kwargs
		if 'S' in kwargs:
			S = kwargs['S']
			kwargs['S'] = S[self.order][:, self.order]

		# Create proposal parameters
		self.create_proposal_params(**kwargs)

	def lf_ratio(self, X, Xjstar, j):
		"""
		Calculates the log of the likelihood ratio
		between two observations: X where X[:,j] 
		is replaced with Xjstar, divided by the likelihood
		of X. This is equivalent to (but sometimes faster) than:

		ld_obs = self.lf(X)
		Xnew = X.copy()
		Xnew[:, j] = Xjstar
		ld_prop = self.lf(Xnew)
		ld_ratio = ld_prop - ld_obs

		When node potentials have been passed, this is much faster
		than calculating the log-likelihood function and subtracting.
		:param X: a n x p matrix of observations
		:param Xjstar: New observations for column j of X
		:param j: an int between 0 and p - 1, telling us which column to replace
		"""

		# Just return the difference in lf if we don't have
		# access to cliques
		if self.clique_dict is None or self.log_potentials is None:
			# Log-likelihood 1
			ld_obs = self.lf(X)
			# New likelihood with Xjstar
			Xnew = X.copy()
			Xnew[:, j] = Xjstar
			ld_prop = self.lf(Xnew)
			ld_ratio = ld_prop - ld_obs

		# If we have access to cliques, we can just compute log-potentials
		else:
			# print(f"I am looping through cliques now")
			cliques = self.clique_dict[j]
			ld_ratio = np.zeros(self.n)

			# Loop through cliques
			for clique_key, clique in cliques:
				# print(f"At clique_key {clique_key},clique {clique}, j={j}")
				# print(f"Orig clique is {self.order[clique]}")
				orig_clique = self.order[clique] # Original ordering

				# Clique representation(s) of X
				Xc = X[:, clique]
				Xcstar = Xc.copy()

				# Which index corresponds to index j in the clique
				new_j = np.where(orig_clique == self.order[j])[0]
				Xcstar[:, new_j] = Xjstar.reshape(-1, 1)

				# Calculate log_potential difference
				ld_ratio += self.log_potentials[clique_key](Xcstar).reshape(-1)
				ld_ratio -= self.log_potentials[clique_key](Xc).reshape(-1)

		return ld_ratio

	def lf(self, X):
		""" Reordered likelihood function """
		return self.unordered_lf(X[:, self.inv_order])

	def center(self, M, active_inds=None):
		"""
		Centers an n x j matrix M. For mu = 0, does not perform
		this computation, which actually is a bottleneck
		for large n and p.
		"""
		if self._zero_mu_flag:
			return M
		elif active_inds is None:
			return M - self.mu[:, 0:M.shape[1]]
		else:
			return M - self.mu[:, active_inds]

	def create_proposal_params(self, **kwargs):
		"""
		Constructs the covariance-guided proposal. 
		:param kwargs: kwargs for gaussian_knockoffs
		method, which finds the optimal S matrix.
		"""

		# Find the optimal S matrix
		_, self.S = knockoffs.gaussian_knockoffs(
			X=self.X,
			Sigma=self.V,
			invSigma=self.Q,
			return_S=True,
			**kwargs
		)
		# Scale (S is in correlation format currently)
		S_scale = np.sqrt(np.diag(self.V))
		self.S = self.S / np.outer(S_scale, S_scale)
		self.G = np.concatenate(
			[np.concatenate([self.V, self.V - self.S]),
			 np.concatenate([self.V - self.S, self.V])],
			axis=1
		)

		# Efficiently calculate p inverses of subsets 
		# of feature-knockoff covariance matrix.
		# Variable names follow the notation in 
		# Appendix D of https://arxiv.org/pdf/1903.00434.pdf.
		# To save memory, we do not store all of these matrices,
		# but rather delete them as we go.
		self.invSigma = self.Q.copy()

		# Suppose X sim N(mu, Sigma) and we have proposals X_{1:j-1}star
		# Then the conditional mean of the proposal Xjstar 
		# is muj + mean_transform @ [X - mu, X_{1:j-1}^* - mu_{1:j-1}]
		# where the brackets [] denote vector concatenation.
		# This code calculates those mean transforms and the 
		# conditional variances.
		self.cond_vars = np.zeros(self.p)
		self.mean_transforms = []

		# Possibly log
		if self.metro_verbose:
			print(f"Metro starting to compute proposal parameters...")
			j_iter = tqdm(range(0, self.p))
		else:
			j_iter = range(0, self.p)

		# Loop through and compute
		for j in j_iter:

			# 1. Compute inverse Sigma
			if j > 0:
				# Extract extra components
				gammaj = self.G[self.p+j-1, 0:self.p+j-1]
				sigma2j = self.V[j, j]

				# Recursion for upper-left block
				invSigma_gamma_j = np.dot(self.invSigma, gammaj)
				denomj = np.dot(gammaj, invSigma_gamma_j) - sigma2j
				numerj = np.outer(invSigma_gamma_j, invSigma_gamma_j.T)
				upper_left_block = self.invSigma - numerj / denomj

				# Lower-left (and upper-right) block
				lower_left = invSigma_gamma_j / denomj

				# Bottom-right block
				bottom_right = -1 / denomj

				# Combine
				upper_half = np.concatenate(
					[upper_left_block, lower_left.reshape(-1,1)],
					axis=1
				)
				lower_half = np.concatenate(
					[lower_left.reshape(1,-1), bottom_right.reshape(1, 1)],
					axis=1
				)

				# Set the new inverse sigma: delete the old one
				# to save memory
				del self.invSigma
				self.invSigma = np.concatenate([upper_half,lower_half], axis=0)

			# 2. Now compute conditional mean and variance
			# Helpful bits
			Gamma12_j = self.G[0:self.p+j, self.p+j]

			# Conditional variance
			self.cond_vars[j] = self.G[self.p+j, self.p+j]
			self.cond_vars[j] -= np.dot(
				Gamma12_j, np.dot(self.invSigma, Gamma12_j),
			)

			# Mean transform
			mean_transform = np.dot(Gamma12_j, self.invSigma).reshape(1, -1)
			self.mean_transforms.append(mean_transform)

	def fetch_proposal_params(self, X, prev_proposals):
		"""
		Returns mean and variance of proposal j given X and
		previous proposals.
		:param X: n x p dimension numpy array of observed data
		:param prev_proposals: n x (j - 1) numpy array of previous
		proposals. If None, assumes j = 0.
		"""

		# Infer j from prev_proposals
		if prev_proposals is not None:
			j = prev_proposals.shape[-1]
		else:
			j = 0

		# First p coordinates of cond_mean
		# X is n x p 
		# self.mu is 1 x p
		# self.mean_transforms[j] is 1 x p + j
		# However, this cond mean only depends on
		# the active variables + [0:j], so to save
		# computation, we only compute that dot
		active_inds = list(range(j+1)) + self.active_frontier[j]
		cond_mean = np.dot(
			self.center(X[:, active_inds], active_inds), 
			self.mean_transforms[j][:, active_inds].T
		)

		# Second p coordinates of cond_mean
		if j != 0:
			# prev_proposals is n x j
			# self.mean_transforms[j] is 1 x p + j
			# self.mu is 1 x p
			cond_mean2 = np.dot(
				self.center(prev_proposals),
				self.mean_transforms[j][:, self.p:].T
			)
			# Add together
			cond_mean += cond_mean2

		# Shift and return
		cond_mean += self.mu[0, j] 
		return cond_mean, self.cond_vars[j]

	def fetch_cached_proposal_params(self, Xtemp, x_flags, j):
		"""
		Same as above, but uses caching to speed up computation.
		This caching can be cheap (if self.cache is False) or
		extremely expensive (if self.cache is True) in terms of
		memory.
		"""

		# Conditional mean only depends on these inds
		active_inds = list(range(j+1)) + self.active_frontier[j]

		# Calculate conditional means from precomputed products
		if self.cache:
			cond_mean = np.where(
				x_flags[:, active_inds],
				self.cached_mean_obs_eq_prop[j],
				self.cached_mean_obs_eq_obs[j],
			).sum(axis=1)
		else:
			active_inds = list(range(j+1)) + self.active_frontier[j]
			cond_mean = np.dot(
				self.center(Xtemp[:, active_inds], active_inds), 
				self.mean_transforms[j][:, active_inds].T
			).reshape(-1)

		# Add the effect of conditioning on the proposals
		cond_mean += self.cached_mean_proposals[j]
		return cond_mean,self.cond_vars[j]

	def q_ll(self, Xjstar, X, prev_proposals):
		"""
		Calculates the log-likelihood of a proposal Xjstar given X 
		and the previous proposals.
		:param Xjstar: n-length numpy array of values to evaluate.
		:param X: n x p dimension numpy array of observed data
		:param prev_proposals: n x (j - 1) numpy array of previous
		proposals. If None, assumes j = 0.
		"""
		cond_mean, cond_var = self.fetch_proposal_params(
			X=X, prev_proposals=prev_proposals
		)
		return gaussian_log_likelihood(Xjstar, cond_mean.reshape(-1), cond_var)

	def sample_proposals(self, X, prev_proposals):
		"""
		Samples 
		:param Xjstar: n-length numpy array of values to evaluate.
		:param X: n x p dimension numpy array of observed data
		:param prev_proposals: n x (j - 1) numpy array of previous
		proposals. If None, assumes j = 0.
		"""
		cond_mean, cond_var = self.fetch_proposal_params(
			X=X, prev_proposals=prev_proposals
		)
		proposal = np.sqrt(cond_var)*np.random.randn(self.n, 1) + cond_mean
		return proposal.reshape(-1)

	def get_key(self, x_flags, j):
		"""
		Fetches key for dp dicts
		"""
		inds = list(self.active_frontier[j])#list(set(self.active_frontier[j]).union(set([j])))
		arr_key = x_flags[0, inds]
		key = arr_key.tostring()
		return key

	def _create_Xtemp(self, x_flags, j):
		"""
		Returns a n x p array Xtemp which effectively does:
		Xtemp = self.X.copy()
		Xtemp[x_flags == 1] = self.X_prop[x_flags == 1].copy()

		TODO: make this so it copies less
		"""

		# Informative error
		if x_flags[:, 0:j].sum() > 0:
			raise ValueError(f"x flags are {x_flags} for j={j}, strange because they should be zero before j")

		Xtemp = np.where(x_flags, self.X_prop, self.X)
		return Xtemp

	def log_q12(self, x_flags, j):
		"""
		Computes q1 and q2 as specified by page 33 of the paper.
		"""

		# Temporary vector of Xs for query
		Xtemp = self._create_Xtemp(x_flags, j)

		# Precompute cond_means for log_q2
		cond_mean2, cond_var = self.fetch_cached_proposal_params(
			Xtemp=Xtemp,
			x_flags=x_flags,
			j=j,
		)

		# q2 is:
		# Pr(Xjstar = xjstar | X = Xtemp, tildeX_{1:j-1}, Xstar_{1:j-1})
		log_q2 = gaussian_log_likelihood(
			X=self.X_prop[:, j],
			mu=cond_mean2.reshape(-1),
			var=cond_var,
		)

		# Adjust cond_mean for q1
		diff = self.X_prop[:, j] - Xtemp[:, j]
		adjustment = self.mean_transforms[j][:, j]*(diff)
		cond_mean1 = cond_mean2.reshape(-1) + adjustment 

		# q1 is:
		# Pr(Xjstar = Xtemp[j] | Xj = xjstar, X_{-j} = X_temp_{-j}, tildeX_{1:j-1}, Xstar_{1:j-1})
		log_q1 = gaussian_log_likelihood(
			X=Xtemp[:, j],
			mu=cond_mean1.reshape(-1),
			var=cond_var,
		)

		return log_q1, log_q2, Xtemp

	def compute_F(self, x_flags, j):
		"""
		Computes the F function from Page 33 pf tje paper: 
		Pr(tildeXj=tildexj, Xjstar=xjstar | Xtemp, tildeX_{1:j-1}, Xjstar_{1:j-1})
		Note that tildexj and xjstar are NOT inputs because they do NOT change
		during the junction tree DP process.
		"""

		# Get key, possibly return cached result
		key = self.get_key(x_flags, j)
		if key in self.F_dicts[j]:
			return self.F_dicts[j][key]
		else:
			self.F_queries += 1


		# q1/q2 terms
		log_q1, log_q2, Xtemp = self.log_q12(x_flags, j)

		# Acceptance mask and probabilities: note that
		# the flag for accepting / rejecting comes from the
		# TRUE knockoffs (e.g. self.acceptances)
		if key in self.acc_dicts[j]:
			acc_probs = self.acc_dicts[j][key]
		else:
			# Pass extra parameters to avoid repeating computation
			acc_probs = self.compute_acc_prob(
				x_flags=x_flags, 
				j=j, 
				log_q1=log_q1, 
				log_q2=log_q2,
				Xtemp=Xtemp
			)
		mask = self.acceptances[:, j] == 1
		result = log_q2 + mask*np.log(acc_probs) + (1-mask)*np.log(1-acc_probs)

		# Cache
		self.F_dicts[j][key] = result

		# Return
		return result

	def compute_acc_prob(
		self,
		x_flags,
		j,
		log_q1=None,
		log_q2=None,
		Xtemp=None,
	):
		"""
		Computes
		Pr(tildeXj = Xjstar | Xtemp, Xtilde_{1:j-1}, Xstar_{1:j})
		"""

		# Get key, possibly return cached result
		key = self.get_key(x_flags, j)
		if key in self.acc_dicts[j]:
			return self.acc_dicts[j][key]

		# 1. q1, q2 ratio
		if log_q1 is None or log_q2 is None:
			log_q1, log_q2, Xtemp = self.log_q12(x_flags, j)
		lq_ratio = log_q1 - log_q2

		# Possibly ceate X temp variable
		if Xtemp is None:
			Xtemp = self._create_Xtemp(x_flags, j)

		# 2. Density ratio
		ld_ratio = self.lf_ratio(
			X=Xtemp,
			Xjstar=self.X_prop[:, j],
			j=j,
		)
		# # a. According to pattern
		# ld_obs = self.lf(Xtemp)
		# # b. When Xj is not observed
		# Xtemp_prop = Xtemp.copy()
		# Xtemp_prop[:, j] = self.X_prop[:, j]
		# ld_prop = self.lf(Xtemp_prop)
		# ld_ratio = ld_prop - ld_obs

		# Delete to save memory
		del Xtemp

		# 3. Calc ln(Fk ratios) for k < j. These should be 0 except
		# when k < j and j in Vk, which is why we loop through 
		# affected variables.
		# Numerator for these ratios use different flags
		Fj_ratio = np.zeros(self.n)
		x_flags_num = x_flags.copy()
		x_flags_num[:, j] = 1

		# Loop through
		for j2 in self.affected_vars[j]:

			# Numerator
			num_key = self.get_key(x_flags_num, j2)
			if num_key in self.F_dicts[j2]:
				Fj2_num = self.F_dicts[j2][num_key]
			else:
				Fj2_num = self.compute_F(x_flags_num, j2)

			# Denominator uses same flags
			denom_key = self.get_key(x_flags, j2)
			if denom_key in self.F_dicts[j2]:
				Fj2_denom = self.F_dicts[j2][denom_key]
			else:
				Fj2_denom = self.compute_F(x_flags, j2) 

			# Add ratio to Fj_ratio
			Fj_ratio = Fj_ratio + Fj2_num - Fj2_denom

		# Put it all together and multiply by gamma
		# Fast_exp function is helpful for profiling
		# (to see how much time is spent here)
		def fast_exp(ld_ratio, lq_ratio, Fj_ratio):
			return np.exp((ld_ratio + lq_ratio + Fj_ratio).astype(np.float32))
		acc_prob = self.gamma * np.minimum(
				1, fast_exp(ld_ratio, lq_ratio, Fj_ratio)
		)

		# Clip to deal with floating point errors
		acc_prob = np.minimum(
			self.gamma, 
			np.maximum(self.clip, acc_prob)
		)

		# Make sure the degenerate case has been computed
		# correctly 
		if x_flags[:,j].sum() > 0:
			if acc_prob[x_flags[:, j] == 1].mean() <= self.gamma:
				msg = f'At step={self.step}, j={j}, we have'
				msg += f"acc_prob = {acc_prob} but x_flags[:, j]={x_flags[:, j]}"
				msg += f"These accetance probs should be ~1"
				raise ValueError(msg)

		# Cache and return
		self.acc_dicts[j][key] = acc_prob
		return acc_prob

	def sample_knockoffs(self, clip=1e-5, cache=None):
		"""
		:param clip: To provide numerical stability,
		we make the minimum acceptance probability 1e-5
		(otherwise some acc probs get very slightly
		negative due to floating point errors)
		:param cache: If True, uses a very memory intensive
		caching system to get a 2-3x speedup when calculating
		conditional means for the proposals. Defaults to true
		if n * (p**2) < 1e9.
		"""

		# Save clip constant for later
		self.clip = clip
		num_params = self.n * (self.p**2)
		if cache is not None:
			self.cache = cache
		else:
			self.cache = num_params < 1e9
		# Possibly log
		if self.metro_verbose:
			if self.cache:
				print(f"Metro will use memory expensive caching for 2-3x speedup, storing {num_params} params")
			else:
				print(f"Metro will not cache cond_means to save a lot of memory")

		# Dynamic programming approach: store acceptance probs
		# as well as Fj values (see page 33)
		self.acc_dicts = [{} for j in range(self.p)]
		self.F_dicts = [{} for j in range(self.p)]

		# Locate previous terms affected by variable j
		self.affected_vars = [[] for k in range(self.p)]
		for j, j2 in itertools.product(range(self.p), range(self.p)):
			if j in self.active_frontier[j2]:
				self.affected_vars[j] += [j2]
		#print(f"affected vars is {self.affected_vars}")

		# Store pattern of TRUE acceptances / rejections
		self.acceptances = np.zeros((self.n, self.p)).astype(np.bool)
		self.final_acc_probs = np.zeros((self.n, self.p))

		# Proposals
		self.X_prop = np.zeros((self.n, self.p)).astype(np.float32)
		self.X_prop[:] = np.nan

		# Start to store knockoffs
		self.Xk = np.zeros((self.n, self.p)).astype(np.float32)
		self.Xk[:] = np.nan

		# Decide whether or not to log
		if self.metro_verbose:
			print(f"Metro beginning to compute proposals...")
			j_iter = tqdm(range(self.p))
		else:
			j_iter = range(self.p)
		# Loop across variables to sample proposals
		for j in j_iter:

			# Sample proposal
			self.X_prop[:, j] = self.sample_proposals(
				X=self.X,
				prev_proposals=self.X_prop[:, 0:j]
			).astype(np.float32)

		# Cache conditional means
		if self.metro_verbose:
			print(f"Metro beginning to cache conditional means...")
			j_iter = tqdm(range(self.p))
		else:
			j_iter = range(self.p)

		# Precompute centerings
		centX = self.center(self.X)
		centX_prop = self.center(self.X_prop)
		self.cached_mean_obs_eq_obs = [None for _ in range(self.p)]
		self.cached_mean_obs_eq_prop = [None for _ in range(self.p)]
		self.cached_mean_proposals = [None for _ in range(self.p)]
		for j in j_iter:

			# We only need to store the coordinates along the active 
			# inds which saves some memory
			active_inds = list(range(j+1)) + self.active_frontier[j]

			# Cache some precomputed conditional means
			# a. Cache the effect of conditioning on Xstar = self.X_prop
			# This is very cheap
			self.cached_mean_proposals[j] = np.dot(
				self.center(self.X_prop[:, 0:j]),
				self.mean_transforms[j][:, self.p:].T
			).reshape(-1)

			# b/c: Possibly cache the effect of conditioning on X = self.X / self.X_prop
			# This is very memory intensive
			if self.cache:
				# a. Cache the effect of conditiong on X = self.X
				cache_obs = (
					centX[:, active_inds]*self.mean_transforms[j][:, 0:self.p][:, active_inds]
				).astype(np.float32)
				self.cached_mean_obs_eq_obs[j] = cache_obs
				# b. Cache the effect of conditioning on X = self.X_prop
				cache_prop = (
					centX_prop[:, active_inds]*self.mean_transforms[j][:, 0:self.p][:, active_inds]
				).astype(np.float32)
				self.cached_mean_obs_eq_prop[j] = cache_prop

		# Loop across variables to compute acc ratios
		prev_proposals = None
		if self.metro_verbose:
			print(f"Metro computing acceptance probabilities...")
			j_iter = tqdm(range(self.p))
		else:
			j_iter = range(self.p)
		for j in j_iter:

			# Cache which knockoff we are sampling
			self.step = j

			# Compute acceptance probability, which is an n-length vector
			acc_prob = self.compute_acc_prob(
				x_flags=np.zeros((self.n, self.p)),
				j=j,
			) 
			self.final_acc_probs[:,j] = acc_prob

			# Sample to get actual acceptances
			self.acceptances[:, j] = np.random.binomial(1, acc_prob).astype(np.bool)

			# Store knockoffs
			mask = self.acceptances[:, j] == 1
			self.Xk[:, j][mask] = self.X_prop[:, j][mask]
			self.Xk[:, j][~mask] = self.X[:, j][~mask] 

		# Return re-sorted knockoffs		
		return self.Xk[:, self.inv_order]


### Knockoff Samplers for T-distributions
def t_markov_loglike(X, rhos, df_t=3):
	"""
	Calculates log-likelihood for markov chain
	specified in https://arxiv.org/pdf/1903.00434.pdf
	"""
	p = X.shape[1]
	if rhos.shape[0] != p - 1:
		raise ValueError(
			f"Shape of rhos {rhos.shape} does not match shape of X {X.shape}"
		)
	inv_scale = np.sqrt(df_t / (df_t - 2))

	# Initial log-like for first variable
	loglike = t_log_likelihood(inv_scale * X[:, 0], df_t=df_t)

	# Differences: these are i.i.d. t
	#print(inv_scale * (X[:, 1:] - rhos * X[:, :-1]))
	conjugates = np.sqrt(1-rhos**2)
	Zjs = inv_scale*(X[:, 1:] - rhos*X[:, :-1]) / conjugates
	Zj_loglike = t_log_likelihood(Zjs, df_t=df_t)

	# Add log-likelihood for differences
	return loglike + Zj_loglike.sum(axis=1)

class ARTKSampler(MetropolizedKnockoffSampler):

	def __init__(
		self,
		X,
		V,
		Q=None,
		df_t=3,
		**kwargs
	):
		"""
		Samples knockoffs for AR1 t designs. (Hence, ARTK).
		:param X: n x p design matrix, presumably following
		t_{df_t}(mu, Sigma) distribution.

		Currently does not support a mean parameter (mu).

		:param V: Covariance matrix. The first diagonal should
		be the pairwise correlations.
		:param Q: Inverse of covariance matrix.
		:param mu: Mean (location) parameter. Defaults to all zeros.
		:param df_t: Degrees of freedom (default: 3).
		:param kwargs: kwargs for Metro sampler.
		"""

		# Rhos and graph
		p = V.shape[0]
		self.df_t = df_t
		self.rhos = np.diag(V, 1)
		if Q is None:
			Q = utilities.chol2inv(V)

		# Account for the fact there will likely be rejections
		if 'rec_prop' not in kwargs:
			kwargs['rec_prop'] = 0.3
			self.rej_rate = 0.3
		else:
			self.rej_rate = kwargs['rec_prop']

		# Cliques and clique log-potentials - start
		# with initial clique. Note that a log-potential
		# for a clique of size k takes an array of size
		# n x k as an input. 
		cliques = [[0]]
		log_potentials = []
		inv_scale = np.sqrt(df_t / (df_t - 2))
		log_potentials.append(lambda X0: t_log_likelihood(inv_scale*X0, df_t=df_t)) 

		# Pairwise log-potentials
		conjugates = np.sqrt(1-self.rhos**2)
		for i, rho, conj in zip(list(range(1, p)), self.rhos, conjugates):
			# Append the clique: X[:, [i+1,i]]
			cliques.append([i-1,i])
			# Takes an n x 2 array as an input
			log_potentials.append(
				lambda Xc: t_log_likelihood(
					inv_scale*(Xc[:,1]-rho*Xc[:,0])/conj, df_t=df_t
				)
			)

		# Loss function (unordered)
		def lf(X):
			return t_markov_loglike(X, self.rhos, self.df_t)

		super().__init__(
			lf=lf,
			X=X,
			mu=np.zeros(p),
			V=V,
			Q=Q,
			undir_graph=np.abs(Q) > 1e-4,
			cliques=cliques,
			log_potentials=log_potentials,
			**kwargs
		)

def t_mvn_loglike(X, invScale, mu=None, df_t=3):
	"""
	Calculates multivariate t log-likelihood
	up to normalizing constant.
	:param X: n x p array of data
	:param invScale: p x p array, inverse multivariate t scale matrix
	:param mu: p-length array, location parameter
	:param df_t: degrees of freedom
	"""
	p = invScale.shape[0]
	if mu is not None:
		X = X - mu
	quad_form = (np.dot(X, invScale) * X).sum(axis=1)
	log_quad = np.log(
		1 + quad_form / df_t
	)
	exponent = -1*(df_t + p) / 2
	return exponent*log_quad 

class BlockTSampler():

	def __init__(
		self,
		X,
		V,
		df_t=3,
		**kwargs
	):
		"""
		Samples knockoffs for block multivariate t designs.
		:param X: n x p design matrix, presumably following
		t_{df_t}(mu, Sigma) distribution.

		:param V: Covariance matrix for t distribution.
		This should be in block-diagonal form.
		:param mu: Mean parameter
		:param df_t: Degrees of freedom (default: 3).
		:param kwargs: kwargs for Metro sampler.
		"""

		# Discover "block structure" of T
		self.p = V.shape[0]
		self.blocks, self.block_inds = graphs.cov2blocks(V)
		self.df_t = df_t
		self.X = X

		# Dummy order / inv_order variables for consistency
		self.order = np.arange(self.p)
		self.inv_order = np.arange(self.p)

		# Account for the fact there will likely be rejections
		if 'rec_prop' not in kwargs:
			kwargs['rec_prop'] = 0.3
			self.rej_rate = 0.3
		else:
			self.rej_rate = kwargs['rec_prop']

		# Loop through blocks and initialize samplers
		self.samplers = []
		self.S = []
		for block, inds in zip(self.blocks, self.block_inds):

			# Invert block and create scale matrix
			inv_block = utilities.chol2inv(block)
			invScale = (df_t) / (df_t - 2) * inv_block

			# Undir graph is all connected
			blocksize = block.shape[0]
			undir_graph = np.ones((blocksize, blocksize))

			# Initialize sampler
			block_sampler = MetropolizedKnockoffSampler(
				lf = lambda X: t_mvn_loglike(X, invScale, df_t=df_t),
				X=X[:, inds],
				mu=np.zeros(blocksize),
				V=block,
				undir_graph=undir_graph,
				**kwargs
			)
			inv_order = block_sampler.inv_order
			self.S.append(block_sampler.S[:, inv_order][inv_order])
			self.samplers.append(block_sampler)

		# Concatenate S
		self.S = sp.linalg.block_diag(*self.S)

	def sample_knockoffs(self, **kwargs):
		"""
		Actually samples knockoffs sequentially for each block.
		kwargs = kwargs for sampler.
		"""
		# Loop through blocks and sample
		self.Xk = []
		self.final_acc_probs = []
		self.acceptances = []

		for j in range(len(self.blocks)):
			# Sample knockoffs
			Xk_block = self.samplers[j].sample_knockoffs(**kwargs)
			self.Xk.append(Xk_block)

			# Save final_acc_probs, acceptances
			block_acc_probs = self.samplers[j].final_acc_probs[:, self.samplers[j].inv_order]
			self.final_acc_probs.append(block_acc_probs)
			block_acc = self.samplers[j].acceptances[:, self.samplers[j].inv_order]
			self.acceptances.append(block_acc)

		# Concatenate + return
		self.Xk = np.concatenate(self.Xk, axis=1)
		self.final_acc_probs = np.concatenate(self.final_acc_probs, axis=1)
		self.acceptances = np.concatenate(self.acceptances, axis=1)
		return self.Xk