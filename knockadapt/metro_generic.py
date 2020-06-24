''' 
	

	The metropolized knockoff sampler for an arbitrary probability density
	and graphical structure using covariance-guided proposals.

	See https://arxiv.org/abs/1903.00434 for a description of the algorithm
	and proof of validity and runtime.

	Initial author: Stephen Bates, October 2019.
	Adapted by: Asher Spector, June 2020. 
'''

# The basics
import sys
import numpy as np
import scipy as sp
import itertools
from . import utilities, knockoffs

# Network and UGM tools
import networkx as nx

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

		# Save order and inverse order
		self.order = order
		self.inv_order = order.copy()
		for i, j in enumerate(order):
			self.inv_order[j] = i

		# Re-order the loss function and variables
		self.X = X[:, self.order]
		self.lf = lambda X: lf(X[:, self.inv_order])

		# Re-order active frontier
		self.active_frontier = []
		for i in range(len(order)):
			self.active_frontier += [
				[self.inv_order[j] for j in active_frontier[i]]
			]

		# Count the number of Fj queries we make
		self.Fj_queries = 0

		# Re-order mu, sigma
		self.mu = mu[self.order].reshape(1, -1)
		self.V = V[self.order][:, self.order]

		# Create proposal parameters
		self.create_proposal_params(**kwargs)

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
		for j in range(1, self.p):
			
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
		# Infer j from proposals
		if prev_proposals is not None:
			j = prev_proposals.shape[-1]

			# Calculate conditional means
			# self.mu is 1 x p 
			normalized_obs = np.concatenate(
				[X - self.mu, prev_proposals - self.mu[:, 0:j]], axis=1
			)
		# Otherwise j = 0
		else:
			j = 0
			normalized_obs = X - self.mu

		# Evaluate Gaussian log-likelihood
		# self.mu is a 1 x p array
		# Mean transforms is a 1 x (p + j) array
		# Normalized obs is a n x (p + j) array 
		# cond_mean will be an n x 1 array
		cond_mean = self.mu[:, j].reshape(1, 1)
		cond_mean = cond_mean + np.dot(normalized_obs, self.mean_transforms[j].T).reshape(-1, 1)
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

	def _create_Xtemp(self, acc, j):
		"""
		Creates the temporary variable Xtemp for queries
		to densities / proposal functions.
		"""

		# Create temporary X variable for queries
		X_temp = self.X.copy()
		X_temp[acc == 1] = self.X_prop[acc == 1]
		X_temp[:, 0:j+1] = self.X[:, 0:j+1]
		return X_temp

	def log_q12(self, acc, j):
		"""
		Computes the terms q1 and q1 from page 33 of the paper 
		"""

		# Temporary variable
		X_temp = self._create_Xtemp(acc, j)

		# First, conditioned on Xj = Xj
		X_temp1 = np.concatenate(
			[
				X_temp[:, 0:j],
				self.X_prop[:, j].reshape(-1, 1),
				X_temp[:, j+1:]
			],
			axis=1
		)
		log_q1 = self.q_ll(
			Xjstar=X_temp[:, j],
			X=X_temp1,
			prev_proposals=self.X_prop[:, 0:j]
		)

		# Second, conditioned on Xj = xjstar
		log_q2 = self.q_ll(
			Xjstar=self.X_prop[:, j],
			X=X_temp,
			prev_proposals=self.X_prop[:, 0:j]
		)

		# Save memory
		del X_temp
		del X_temp1

		return log_q1, log_q2


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

	def get_key(self, acc, j):
		"""
		Fetches key for acceptence dicts
		returns: stringified-key (hashable), array key
		"""
		inds = list(set(self.active_frontier[j]).union(set([j])))
		arr_key = acc[0, inds] # Could make this :, inds
		return arr_key.tostring(), arr_key 

	def compute_Fj(
		self,
		acc,
		j
		):
		"""
		Computes ln Pr(tildeXj, Xjstar | X_{Vj} = z_{vj}, X_{V_j^c}, tildeX_{1:j-1}, Xstar_{1:j-1}) 
		By Lemma 4 of the Metro paper, this depends on variable k
		iff k in bar Vj. Note active frontier is the set bar Vj / {1,...,j}
		:param acc: This specifies the values of zvj. In particular,
		Vj, we set X[i, k] = X_prop[i, k] iff acc[i, k] == 1
		"""

		# Key for dp dicts
		key, readable_key = self.get_key(acc, j)
		if key in self.Fj_dicts[j]:
			print(f"You are causing extra recursion, stop this")
			return self.Fj_dicts[j][key]

		# Log our queries
		self.Fj_queries += 1

		# Log-q1/q2 -- see page 33 of the paper
		log_q1, log_q2 = self.log_q12(acc, j)

		# Compute acceptance proposals
		if key in self.acc_dicts[j]:
			acc_prob = self.acc_dicts[j][key]
		else:
			acc_prob = self.compute_acc_prob(acc, j)

		# Acceptance / rejection mask
		result = acc[:, j] * np.log(acc_prob)
		result = result + (1 - acc[:, j]) * np.log(1 - acc_prob)
		result = result + log_q2

		# Store result
		print(f"For j={j}, caching key={readable_key}")
		self.Fj_dicts[j][key] = result

		# Return
		return result

	def compute_acc_prob(self, acc, j, log_q1=None, log_q2=None):
		''' Computes the acceptance probability at step j.

			This calculation is based on the observed sequence of proposals and 
			accept/rejects of the steps before j, and the configuration of variables after j
			specified by the acceptances after j.

			Args:
				lf (function that takes a 1-dim numpy array) : the log probability density
				x (1-dim numpy array, length p) : the observed sample
				x_prop (1-dim numpy array, length p) : the sequence of proposals.
					Only needs to be filled up to the last nonzero entry of 'acc'.
				acc (1-dim 0-1 numpy array, length p) : the sequence of acceptances (1) 
					and rejections (0).
				j (int) : the active index
				active_frontier (list of lists) : as above
				affected_vars (list of lists) : entry j gives the set of variables occuring
					before j that are affected by j's value. This is a pre-processed version of 
					"active frontier" that contains the same information in a convenient form.
				dp_dicts (list of dicts) : stores the results of calls to this function.
				gamma (float) : multiplier for the acceptance probability, between 0 and 1.

		'''

		# Return acceptance probs if computed previously
		key, readable_key = self.get_key(acc, j)
		if key in self.acc_dicts[j]:
			print(f"You are causing extra recursion, stop this")
			return self.acc_dicts[j][key]

		# Else, compute the entry
		acc0 = acc.copy() # Rejection pattern if we reject at j
		acc1 = acc.copy() # Rejection pattern if we accept at j
		acc0[:, j] = 0
		acc1[:, j] = 1

		# TODO GET RID OF THE COPIES
		# AND CREATE CONCATENATES: MAY SAVE TONS OF TIME

		# Step 1: Compute q1/q2 from page 33
		if log_q1 is None or log_q2 is None:
			log_q1, log_q2 = self.log_q12(acc, j)
		lq_ratio = log_q1 - log_q2

		# Step 2: Compute density ratio
		# Create temporary X variable for queries
		X_temp = self._create_Xtemp(acc, j)

		# Compute the actual likelihoods
		X_temp[:, j] = self.X[:, j]
		ld_obs = self.lf(X_temp)
		X_temp[:, j] = self.X_prop[:, j]
		ld_prop = self.lf(X_temp)
		ld_ratio = ld_prop - ld_obs

		# Save memory
		del X_temp

		# Loop across history to adjust for the observed knockoff sampling pattern
		Fj_ratio = np.zeros(self.n)
		for j2 in self.affected_vars[j]:

			# Keys
			key0, readable_key0 = self.get_key(acc0, j2)
			key1, readable_key1 = self.get_key(acc1, j2)

			# Logging
			print(f"Beginning calcs for step={self.step}, j={j}, j2={j2}, acc0={acc0[0]}, acc1={acc1[0]}")
			print(f"For this step, keys for {j2} are {readable_key0} and {readable_key1}")

			# Fj value for acc0
			if key0 in self.Fj_dicts[j2]:
				print(f"DP for j2={j2}, key={readable_key0}")
				Fj2_0 = self.Fj_dicts[j2][key0]
			else:
				Fj2_0 = self.compute_Fj(acc0, j2)
			# Fj value for acc1
			if key1 in self.Fj_dicts[j2]:
				print(f"DP for j2={j2}, key={readable_key1}")
				Fj2_1 = self.Fj_dicts[j2][key1]
			else:
				Fj2_1 = self.compute_Fj(acc1, j2)

			# Account for whether variable j2 was accepted/rejected
			Fj_ratio += Fj2_1 - Fj2_0

		# Probability of acceptance at step j, given past rejection pattern
		# print(f"step={self.step}, j={j}, LD+LQ rat", np.around(ld_ratio+lq_ratio, 3))
		# print(f"step={self.step}, j={j}, LQ rat", lq_ratio)
		# print(f"step={self.step}, j={j}, Fj rat", np.around(Fj_ratio, 3))
		acc_prob = self.gamma * np.minimum(
			1, np.exp(ld_ratio + lq_ratio + Fj_ratio)
		) 

		# Store result
		self.acc_dicts[j][key] = acc_prob 

		return acc_prob

	def sample_knockoffs(self):

		# Dynamic programming approach: store acceptance probs
		# as well as Fj values (see page 33)
		self.acc_dicts = [{} for j in range(self.p)]
		self.Fj_dicts = [{} for j in range(self.p)]

		# Make sure our DP for n variables actually works
		self.key_to_acc = [{} for j in range(self.p)]
		# TODO TEST THIS

		# Locate previous terms affected by variable j
		self.affected_vars = [[] for k in range(self.p)]
		for j, j2 in itertools.product(range(self.p), range(self.p)):
			if j in self.active_frontier[j2]:
				self.affected_vars[j] += [j2]
		print(f"affected vars is {self.affected_vars}")

		# Store pattern of TRUE acceptances / rejections
		self.acceptances = np.zeros((self.n, self.p))
		self.final_acc_probs = np.zeros((self.n, self.p))

		# Proposals
		self.X_prop = np.zeros((self.n, self.p))
		self.X_prop[:] = np.nan

		# Loop across variables
		prev_proposals = None
		for j in range(self.p):

			# Cache which knockoff we are sampling
			self.step = j
			print("\n")
			print(f"For step={self.step}, active_frontier={self.active_frontier[j]}, affected_vars={self.affected_vars[j]}")
			
			# Sample proposal
			self.X_prop[:, j] = self.sample_proposals(
				X=self.X,
				prev_proposals=self.X_prop[:, 0:j]
			)

			# Compute acceptance probability, which is an n-length vector
			acc_prob = self.compute_acc_prob(
				acc=self.acceptances,
				j=j,
			) 
			self.final_acc_probs[:,j] = acc_prob

			# Sample to get actual acceptances
			self.acceptances[:, j] = np.random.binomial(1, acc_prob)

		# Create knockoffs and return
		self.Xk = self.X.copy()
		self.Xk[self.acceptances == 1.0] = self.X_prop[self.acceptances == 1.0]
		
		return self.Xk


