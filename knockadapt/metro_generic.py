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
from . import utilities, knockoffs

# Network and UGM tools
import networkx as nx

# Logging
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

class MetropolizedKnockoffSampler():

	def __init__(
			self,
			lf,
			X,
			mu,
			V,
			order,
			active_frontier,
			gamma=0.999,
			metro_verbose=False,
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

		# Save order and inverse order
		self.order = order
		self.inv_order = order.copy()
		for i, j in enumerate(order):
			self.inv_order[j] = i

		# Re-order the variables: the log-likelihood
		# function (lf) is reordered in a separate method
		self.X = X[:, self.order].astype(np.float32)
		self.unordered_lf = lf

		# Re-order active frontier
		self.active_frontier = []
		for i in range(len(order)):
			self.active_frontier += [
				[self.inv_order[j] for j in active_frontier[i]]
			]

		# Count the number of Fj queries we make
		self.F_queries = 0

		# Re-order mu
		self.mu = mu[self.order].reshape(1, -1)

		# If mu == 0, then we can save lots of time
		if np.all(self.mu == 0):
			self._zero_mu_flag = True 
		else:
			self._zero_mu_flag = False

		# Re-order sigma
		self.V = V[self.order][:, self.order]

		# Possibly reorder S if it's in kwargs
		if 'S' in kwargs:
			S = kwargs['S']
			kwargs['S'] = S[self.order][:, self.order]

		# Create proposal parameters
		self.create_proposal_params(**kwargs)

	def lf(self, X):
		""" Reordered likelihood function """
		return self.unordered_lf(X[:, self.inv_order])

	def center(self, M):
		"""
		Centers an n x j matrix M. For mu = 0, does not perform
		this computation, which actually is a bottleneck
		for large n and p.
		"""
		if self._zero_mu_flag:
			return M
		else:
			return M - self.mu[:, 0:M.shape[1]]


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
		self.invSigmas = []
		self.invSigmas.append(utilities.chol2inv(self.V))

		# Possibly log
		if self.metro_verbose:
			print(f"Metro starting to compute proposal parameters...")
			j_iter = tqdm(range(1, self.p))
		else:
			j_iter = range(1, self.p)

		# Loop through and compute
		for j in j_iter:
			
			# Extract extra components
			gammaj = self.G[self.p+j-1, 0:self.p+j-1]
			sigma2j = self.V[j, j]

			# Recursion for upper-left block
			invSigma_gamma_j = np.dot(self.invSigmas[j-1], gammaj)
			denomj = np.dot(gammaj, invSigma_gamma_j) - sigma2j
			numerj = np.outer(invSigma_gamma_j, invSigma_gamma_j.T)
			upper_left_block = self.invSigmas[j-1] - numerj / denomj

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
			invSigmaj = np.concatenate([upper_half,lower_half], axis=0)
			self.invSigmas.append(invSigmaj)

		# Suppose X sim N(mu, Sigma) and we have proposals X_{1:j-1}star
		# Then the conditional mean of the proposal Xjstar 
		# is muj + mean_transform @ [X - mu, X_{1:j-1}^* - mu_{1:j-1}]
		# where the brackets [] denote vector concatenation.
		# This code calculates those mean transforms and the 
		# conditional variances.
		self.cond_vars = np.zeros(self.p)
		self.mean_transforms = []
		for j in range(self.p):

			# Helpful bits
			Gamma12_j = self.G[0:self.p+j, self.p+j]
			Gamma11_j_inv = self.invSigmas[j]

			# Conditional variance
			self.cond_vars[j] = self.G[self.p+j, self.p+j]
			self.cond_vars[j] -= np.dot(
				Gamma12_j, np.dot(Gamma11_j_inv, Gamma12_j),
			)

			# Mean transform
			mean_transform = np.dot(Gamma12_j, Gamma11_j_inv).reshape(1, -1)
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
		cond_mean = np.dot(
			self.center(X),
			self.mean_transforms[j][:, 0:self.p].T
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

		# q1 is:
		# Pr(Xjstar = Xtemp[j] | Xj = xjstar, X_{-j} = X_temp_{-j}, tildeX_{1:j-1}, Xstar_{1:j-1})
		Xtemp1 = Xtemp.copy()
		Xtemp1[:, j] = self.X_prop[:, j]

		log_q1 = self.q_ll(
			Xjstar=Xtemp[:, j],
			X=Xtemp1,
			prev_proposals=self.X_prop[:, 0:j]
		)

		# q2 is:
		# Pr(Xjstar = xjstar | X = Xtemp, tildeX_{1:j-1}, Xstar_{1:j-1})
		log_q2 = self.q_ll(
			Xjstar=self.X_prop[:, j],
			X=Xtemp,
			prev_proposals=self.X_prop[:, 0:j]
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
		# a. According to pattern
		ld_obs = self.lf(Xtemp)
		# b. When Xj is not observed
		Xtemp_prop = Xtemp.copy()
		Xtemp_prop[:, j] = self.X_prop[:, j]
		ld_prop = self.lf(Xtemp_prop)
		ld_ratio = ld_prop - ld_obs

		# Delete to save memory
		del Xtemp_prop
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
		def fast_exp(ld_ratio, lq_ratio, Fj_ratio):
			return np.exp((ld_ratio + lq_ratio + Fj_ratio).astype(np.float32))
		acc_prob = self.gamma * np.minimum(
				1, fast_exp(ld_ratio, lq_ratio, Fj_ratio)
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

	def sample_knockoffs(self, verbose=None):

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


