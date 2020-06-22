''' 
	The metropolized knockoff sampler for an arbitrary probability density
	and graphical structure using covariance-guided proposals.

	See https://arxiv.org/abs/1903.00434 for a description of the algorithm
	and proof of validity and runtime.

	Initial author: Stephen Bates, October 2019.
	Adapted by: Asher Spector, June 2020. 
'''

import numpy as np
import itertools
from . import utilities, knockoffs


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
			gamma=0.999
			):
		"""
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
		ratio.
		"""

		# Random params
		self.n = X.shape[0]
		self.p = X.shape[1]

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

		# Re-order mu, sigma
		self.mu = mu[self.order].reshape(1, -1)
		self.V = V[self.order][:, self.order]

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

	def q_ll(self, Xjstar, X, prev_proposals):
		"""
		Calculates the log-likelihood of a proposal Xjstar given X 
		and the previous proposals.
		:param Xjstar: n-length numpy array of values to evaluate.
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
		return gaussian_log_likelihood(Xjstar, cond_mean, self.cond_vars[j])

def gaussian_proposal(j, xj):
	''' Sample proposal by adding independent Gaussian noise.

	'''

	return xj + np.random.normal()

def single_metro(lf, x, order, active_frontier, sym_proposal = gaussian_proposal, gamma = .99):
	''' Samples a knockoff using the Metro algorithm, using an 
			arbitrary ordering of the variables.

		Args:
			lf (function that takes a 1-d numpy array) : the log probability 
				density, only needs to be specified up to an additive constant
			x (1-dim numpy array, length p) : the observed sample
			order (1-dim numpy array, length p) : ordering to sample the variables.
				Should be a vector with unique entries 0,...,p-1.
			active_fontier (list of lists) : a list of length p where
				entry j is the set of entries > j that are in V_j. This specifies
				the conditional independence structure of the distribution given by
				lf. See page 34 of the paper.
			sym_proposal (function that takes two scalars) : symmetric proposal function

		Returns:
			xk: a vector of length d, the sampled knockoff

	'''
	#reindex to sample variables in ascending order
	inv_order = order.copy()
	for i, j in enumerate(order):
		inv_order[j] = i

	def lf2(x):
		return lf(x[inv_order])

	active_frontier2 = []
	for i in range(len(order)):
		active_frontier2 += [[inv_order[j] for j in active_frontier[i]]]

	def sym_proposal2(j, xj):
		return sym_proposal(order[j], xj)

	# call the metro function that samples variables in ascending order
	return ordered_metro(lf2, x[order], active_frontier2, sym_proposal2, gamma)[inv_order]



def ordered_metro(lf, x, active_frontier, sym_proposal = gaussian_proposal, gamma = .99):
	''' Samples a knockoff using the Metro algorithm, moving from variable 1
		to variable n.

		Args:
			lf (function that takes a 1-d numpy array) : the log probability 
				density, only needs to be specified up to an additive constant
			x (1-dim numpy array, length p) : the observed sample
			active_fontier (list of lists) : a list of length p where
				entry j is the set of entries > j that are in V_j. This specifies
				the conditional independence structure of the distribution given by
				lf. See page 34 of the paper.
			sym_proposal (function that takes two scalars) : symmetric proposal function

		Returns:
			xk: a vector of length d, the sampled knockoff

	'''

	# locate the previous terms that affected by variable j
	affected_vars = [[] for k in range(len(x))]
	for j, j2 in itertools.product(range(len(x)), range(len(x))):
		if j in active_frontier[j2]:
			affected_vars[j] += [j2]

	# store dynamic programming intermediate results
	dp_dicts = [{} for j in range(len(x))] 

	x_prop = np.zeros(len(x)) #proposals
	x_prop[:] = np.nan
	acc = np.zeros(len(x)) #pattern of acceptances

	#loop across variables
	for j in range(len(x)):
		# sample proposal)
		x_prop[j] = sym_proposal(j, x[j])

		# compute accept/reject probability and sample
		acc_prob = compute_acc_prob(lf, x, x_prop, acc, j, active_frontier, affected_vars, dp_dicts, gamma)
		acc[j] = np.random.binomial(1, acc_prob)

	xk = x.copy()
	xk[acc == 1.0] = x_prop[acc == 1.0]
	
	return xk


def compute_acc_prob(lf, x, x_prop, acc, j, active_frontier, affected_vars, dp_dicts, gamma = .99):
	''' Computes the acceptance probability at step j. Intended for use only as
		a subroutine of the "ordered_metro" function.

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
			active_fontier (list of lists) : as above
			affected_vars (list of lists) : entry j gives the set of variables occuring
				before j that are affected by j's value. This is a pre-processed version of 
				"active frontier" that contains the same information in a convenient form.
			dp_dicts (list of dicts) : stores the results of calls to this function.
			gamma (float) : multiplier for the acceptance probability, between 0 and 1.

	'''

	# return entry if previously computed
	key = acc[active_frontier[j]].tostring()
	if key in dp_dicts[j]:
		return(dp_dicts[j][key])
	
	# otherwise, compute the entry
	acc0 = acc.copy() #rejection pattern if we reject at j
	acc1 = acc.copy() #rejection pattern if we accept at j
	acc0[j] = 0
	acc1[j] = 1

	# compute terms from the query to the density
	x_temp1 = x.copy()
	x_temp1[acc == 1] = x_prop[acc == 1] # new point to query
	x_temp1[0:j] = x[0:j]
	x_temp1[j] = x[j]
	x_temp2 = x_temp1.copy()
	x_temp2[j] = x_prop[j] #new point to query if proposal accepted
	ld_obs = lf(x_temp1) #log-density with observed point at j
	ld_prop = lf(x_temp2) #log-density with proposed point at j

	#loop across history to adjust for the observed knockoff sampling pattern
	for j2 in affected_vars[j]:
		p0 = compute_acc_prob(lf, x, x_prop, acc0, j2, active_frontier, affected_vars, dp_dicts, gamma)
		p1 = compute_acc_prob(lf, x, x_prop, acc1, j2, active_frontier, affected_vars, dp_dicts, gamma)
		if(acc[j2] == 1):
			ld_obs += np.log(p0)
			ld_prop += np.log(p1)
		else:
			ld_obs += np.log(1 - p0)
			ld_prop += np.log(1 - p1)

	#probability of acceptance at step j, given past rejection pattern
	acc_prob = gamma * min(1, np.exp(ld_prop - ld_obs)) 
	dp_dicts[j][acc[active_frontier[j]].tostring()] = acc_prob #store result

	return acc_prob

