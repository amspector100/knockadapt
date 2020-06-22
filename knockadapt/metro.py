"""
This code is adapted from the metro package, introduced
by the paper https://arxiv.org/abs/1903.00434. Most of this code
was initially written by Wenshuo Wang and Stephen Bates.
"""
import time
import math
import numpy as np
from scipy.stats import t
from scipy.stats import multivariate_normal
from pydsdp.dsdp5 import dsdp
from . import knockoffs 

def log(x):
	""" Helper function which errors correctly """
	if x==0:
		return(float("-inf"))
	return(math.log(x))
  
def ar1t_knockoffs(
	n,
	Sigma,
	df_t=5,
	verbose=True,
	**kwargs
):
	"""
	:param X: the n x p data matrix. This must be samples
	from an AR1 T distribution.
	:param Sigma: The correlation matrix
	:param **kwargs: Kwargs for the gaussian_knockoffs generator. 
	This will not be used to generate knockoffs, but only to
	create the S matrix.
	"""

	# Raise error if Sigma is not a correlation matrix
	knockoffs.TestIfCorrMatrix(Sigma)

	# Key dgp parameters
	p = Sigma.shape[0]
	rhos = np.diag(Sigma, 1) # First diagonal off the main diagonal

	# Sample X using AR definition
	X = np.zeros((n, p))
	df_scale = np.sqrt((df_t-2)/df_t)
	X[:, 0] = t.rvs(df=df_t, size=(n,))*df_scale
	for j in range(1,p):
		X[:,j] = df_scale*np.sqrt(1-rhos[j-1]**2)*t.rvs(df=df_t, size=(n,)) + rhos[j-1]*X[:,j-1]

	# Find S matrix.
	if verbose:
		print(f"Finding S matrix...")
		_, S = knockoffs.gaussian_knockoffs(
			X=X,
			Sigma=Sigma,
			verbose=verbose,
			return_S=True,
			**kwargs
			)
		S_diag = np.diag(S)

	# Helper functions ---------
	def p_marginal_trans_log(
			j,
			xj,
			xjm1
		):
		"""
		TODO: CHANGE THIS TO BE ZERO INDEXED. 
		 Unnormalized marginal for AR1 t covariance model
		:param j: which coordinate we are at
		:param xj: The value of the jth variable. This may also
		be a n-dimensional vector.
		:param xjm1: The value of the j-1th variable. This may
		also be a n-dimensional vector. 
		Implicit parameters:
		:param rhos: The correlation parameters. Should be 
		of dimension p - 1.
		:param df_t: The degrees of freedom
		:param p: The dimensionality """
		if j>p:
			return("error!")
		if j==1:
			return(np.log(t.pdf(xj*np.sqrt(df_t/(df_t/2)), df=df_t))+0.5*np.log(df_t)-0.5*np.log(df_t-2))
		j = j - 1
		return(np.log(t.pdf((xj-rhos[j-1]*xjm1)*np.sqrt(df_t/(df_t-2))/np.sqrt(1-rhos[j-1]**2),df=df_t))+0.5*(np.log(df_t)-np.log(df_t-2)-np.log(1-rhos[j-1]**2)))

	def p_marginal_log(X):
		""" Combines previous functions to calculate 
		overall likelihood of X from t distribution
		:param x: n x p dimensional array of values.
		Implicit parameters:
		:param df_t: Degrees of freedom for t distribution
		:param rhos: p-1 dimensional array of correlations
		"""
		# Reshape 1D arrays 
		if len(X.shape) == 1:
			X = X.reshape(1, -1)
		n, p = X.shape

		# Iterate through AR1 process and calculate
		res = p_marginal_trans_log(1, X[:,0], 0)
		if p==1:
			return(res)
		# TODO: make this zero indexed for my sanity
		for j in range(2,p+1):
			res = res + p_marginal_trans_log(
				j=j,
				xj=X[:, j-1],
				xjm1=X[:, j-2],
			)
		return(res)

	# starting to calculate the parameters for covariance-guided proposals
	Cov_matrix = Sigma.copy()
	Cov_matrix_off = Sigma - S

	# Not exactly sure what this does, related to above - refactoring is on the todo list
	inverse_all = np.zeros([2*p-1,2*p-1])
	inverse_all[0,0] = (1/(1-rhos[0]**2))/Cov_matrix[0,0]
	inverse_all[0,1] = (-rhos[0]/(1-rhos[0]**2))/np.sqrt(Cov_matrix[0,0]*Cov_matrix[1,1])
	inverse_all[p-1,p-1] = (1/(1-rhos[p-2]**2))/Cov_matrix[p-1,p-1]
	inverse_all[p-1,p-2] = (-rhos[p-2]/(1-rhos[p-2]**2))/np.sqrt(Cov_matrix[p-1,p-1]*Cov_matrix[p-2,p-2])
	if p>=3:
		for i in range(1,p-1):
			inverse_all[i,i-1] = (-rhos[i-1]/(1-rhos[i-1]**2))/np.sqrt(Cov_matrix[i,i]*Cov_matrix[i-1,i-1])
			inverse_all[i,i] = ((1-rhos[i-1]**2*rhos[i]**2)/((1-rhos[i-1]**2)*(1-rhos[i]**2)))/Cov_matrix[i,i]
			inverse_all[i,i+1] = (-rhos[i]/(1-rhos[i]**2))/np.sqrt(Cov_matrix[i,i]*Cov_matrix[i+1,i+1])
	temp_mat = Cov_matrix_off @ inverse_all[0:p,0:p]
	prop_mat = temp_mat
	upper_matrix = np.concatenate((Cov_matrix, Cov_matrix_off), axis = 1)
	lower_matrix = np.concatenate((Cov_matrix_off, Cov_matrix), axis = 1)
	whole_matrix = np.concatenate((upper_matrix, lower_matrix), axis = 0)
	cond_means_coeff = []
	cond_vars = [0]*p
	temp_means_coeff = np.reshape(whole_matrix[p,0:p],[1,p]) @ inverse_all[0:p,0:p]
	cond_means_coeff.append(temp_means_coeff)
	cond_vars[0] = (Cov_matrix[0,0] - cond_means_coeff[0] @ np.reshape(whole_matrix[p,0:p],[p,1]))[0,0]
	for il in range(1,p):
		temp_var = Cov_matrix[il-1]
		temp_id = np.zeros([p+il-1,p+il-1])
		temp_row = np.zeros([p+il-1,p+il-1])
		temp_id[il-1,il-1] = 1
		temp_row[il-1,:] = S_diag[il-1] * inverse_all[il-1,0:(p+il-1)]
		temp_col = np.matrix.copy(np.transpose(temp_row))
		temp_fourth = S_diag[il-1]**2 * np.reshape(inverse_all[il-1,0:(p+il-1)],[p+il-1,1]) @ np.reshape(inverse_all[il-1,0:(p+il-1)],[1,p+il-1])
		temp_numerator = temp_id - temp_row - temp_col + temp_fourth
		temp_denominator = -S_diag[il-1] * (2-S_diag[il-1]*inverse_all[il-1,il-1])
		temp_remaining = -S_diag[il-1]*inverse_all[il-1,0:(p+il-1)]
		temp_remaining[il-1] = 1 + temp_remaining[il-1]
		inverse_all[0:(p+il-1),0:(p+il-1)] = inverse_all[0:(p+il-1),0:(p+il-1)] - (1/temp_denominator)*temp_numerator
		inverse_all[p+il-1,p+il-1] = -1/temp_denominator
		inverse_all[p+il-1,0:(p+il-1)] = 1/temp_denominator * temp_remaining
		inverse_all[0:(p+il-1),p+il-1] = np.matrix.copy(inverse_all[p+il-1,0:(p+il-1)])
		temp_means_coeff = np.reshape(whole_matrix[p+il,0:(p+il)],[1,p+il]) @ inverse_all[0:(p+il),0:(p+il)]
		cond_means_coeff.append(temp_means_coeff)
		cond_vars[il] = (Cov_matrix[il,il] - cond_means_coeff[il] @ np.reshape(whole_matrix[p+il,0:(p+il)],[p+il,1]))[0,0]

	# The main markov sampling method
	def SCEP_MH_MC(X, gamma, mu_vector, cond_coeff, cond_means_coeff, cond_vars):
		"""
		:param X: n x p dimension array of observations
		:param gamma: Multiply try parameter gamma
		:param mu_vector: p dimension array, marginal mean of the data

		:param cond_coeff: 
		:param cond_means_coeff:
		:param cond_vars:
		pass
		"""
		p = X.shape[1]
		rej = 0
		Xk = np.empty((n,p))

		# Conditional mean / variance for gaussian proposals
		residuals = X - mu_vector.reshape(1, p)
		cond_mean = mu_vector.reshape(1, p) + np.dot(residuals, cond_coeff)
		cond_cov = Cov_matrix - cond_coeff @ Cov_matrix_off
		proposals = cond_mean + multivariate_normal.rvs(np.zeros(p), cond_cov, size = (n,))

		# Helper function: proposal log-pdf
		def q_prop_pdf_log(num_j, vec_j, prop_j):
			"""
			:param num_j: The index j
			:param vec_j: A n x p + j length array
			:param prop_j: An n-length array of proposals"""

			num_j = num_j + 1 # Gr todo: use zero indexing
			if num_j!=(vec_j.shape[1]-p+1):
				raise ValueError(f"Unexpected shape for vec_j {vec_j.shape} given num_j={num_j}")
			# Calculate conditional ("temp") mean
			cat_mu_vector = np.concatenate(
				[mu_vector, mu_vector[0:(num_j-1)]]
			).reshape(1, -1)
			temp_mean = cond_means_coeff[num_j-1] @ (vec_j - cat_mu_vector).T
			temp_mean += mu_vector[num_j-1]
			temp_mean = temp_mean.reshape(-1) # Reshape to n-length vector
			# Create likelihood
			return(-(prop_j-temp_mean)**2/(2*cond_vars[num_j-1])-0.5*np.log(2*np.pi*cond_vars[num_j-1]))
		
		# Paralel chains for proposals?
		parallel_chains = np.stack([X for _ in range(p)], axis=2)
		for j in range(p):
			parallel_chains[:,j,j] = proposals[:,j]

		# Marginal density for initial features
		# and initialize the parallel marg densities
		marg_density_log = p_marginal_log(X)
		parallel_marg_density_log = np.stack(
			[marg_density_log for _ in range(p)],
			axis=1
		)
		# One day, these terms will get informative variable names. Maybe.
		for j in range(p):
			if j==0:
				term1 = p_marginal_trans_log(1, proposals[:,0], np.zeros(p))
				term2 = p_marginal_trans_log(2, X[:, 1], proposals[:, 0])
				term3 = p_marginal_trans_log(1, X[:, 0], np.zeros(p))
				term4 = p_marginal_trans_log(2, X[:, 1], X[:, 0])
				parallel_marg_density_log[:, j] += term1 + term2 - term3 - term4 
			if j==p-1:
				term1 = p_marginal_trans_log(p, proposals[:, p-1], X[:, p-2])
				term2 = p_marginal_trans_log(p, X[:, p-1], X[:, p-2])
				parallel_marg_density_log[:, j] += term1 - term2
			if j>0 and j<p-1:
				term1 =  p_marginal_trans_log(j+1, proposals[:, j], X[:, j-1]) 
				term2 = p_marginal_trans_log(j+2, X[:, j+1], proposals[:, j])
				term3 = p_marginal_trans_log(j+1, X[:, j], X[:, j-1])
				term4 = p_marginal_trans_log(j+2, X[:, j+1], X[:, j])
				parallel_marg_density_log[:, j] += term1 + term2 - term3 - term4
		
		# Cond densities
		cond_density_log = np.zeros(n)
		parallel_cond_density_log = np.zeros((n,p))
		for j in range(p):
			# Options: true vector vs. alternate vector (with proposals) 
			true_vec = np.concatenate([X, proposals[:, 0:j]], axis=1)
			alter_vec = np.concatenate([X, proposals[:, 0:j]], axis=1)
			alter_vec[:, j] = proposals[:, j]

			# Compute acceptance ratio (vector of n)
			acc_term1 = q_prop_pdf_log(j, alter_vec, X[:, j])
			acc_term2 = parallel_marg_density_log[:, j]
			acc_term3 = parallel_cond_density_log[:, j]
			acc_term4 = (marg_density_log + cond_density_log + q_prop_pdf_log(j, true_vec, proposals[:, j]))
			acc_ratio_log = acc_term1 + acc_term2 + acc_term3 - acc_term4

			# Masking for acceptance
			unifs = np.random.uniform(1, size=(n,))
			accept_mask = (np.log(unifs) <= acc_ratio_log + np.log(gamma))

			# First, append to Xk
			Xk[:, j] = accept_mask * proposals[:, j] + (1 - accept_mask) * X[:, j]

			# Update log conditional density -- have to use masks
			cond_density_log += q_prop_pdf_log(j, true_vec, proposals[:, j])
			gamma_accept_mask = accept_mask * (np.minimum(0, acc_ratio_log) + np.log(gamma)) # Acceptances
			if gamma == 1:
				# Prevent log errors when 1 - gamma = 0
				gamma_accept_mask += np.array([-1*np.inf if not x else 0 for x in accept_mask])
			else:
				gamma_accept_mask += (1 - accept_mask) * np.log((1-gamma)*np.minimum(1, np.exp(acc_ratio_log)))
			cond_density_log += gamma_accept_mask

			# Update true_vecs and cond densities
			if j+2 <= p:
				true_vec_j = np.concatenate(
					[parallel_chains[:, j+1, :], proposals[:, 0:j]], axis=1
				)
				alter_vec_j = np.concatenate(
					[parallel_chains[:, j+1, :], proposals[:, 0:j]], axis=1
				)
				alter_vec_j[:, j] = proposals[:, j]
				j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, X[:, j]) +\
								  parallel_cond_density_log[:, j] +\
								  parallel_marg_density_log[:, j] +\
								  p_marginal_trans_log(j+2, proposals[:, j+1], proposals[: ,j]) -\
								  p_marginal_trans_log(j+2, X[:, j+1], proposals[:, j])
				if j+3<=p:
					j_acc_ratio_log += p_marginal_trans_log(j+3, X[:, j+2], proposals[:, j+1]) -\
									   p_marginal_trans_log(j+3, X[:, j+2], X[:, j+1])
				print("GAMMA MASK", gamma_accept_mask)
				true_vec_q_props = q_prop_pdf_log(j, true_vec_j, proposals[:, j]) # Memoize
				j_acc_ratio_log += -1*parallel_cond_density_log[:, j+1] +\
									parallel_marg_density_log[:, j+1] +\
									true_vec_q_props
				parallel_cond_density_log[:, j+1] += true_vec_q_props + gamma_accept_mask

			if j+3<=p:
				for ii in range(j+2,p):
					parallel_cond_density_log[:, ii] = cond_density_log

			# Count rejections
			rej += (accept_mask==0).sum()

		return Xk, rej

	# TODO: The marginal mean of X should probably be a function input
	mu = np.zeros(p)
	Xk, rejections = SCEP_MH_MC(
		X=X,
		gamma=1,
		mu_vector=mu,
		cond_coeff=prop_mat,
		cond_means_coeff=cond_means_coeff, # p-length list. ith entry is 1 x p+i length array
		cond_vars=cond_vars
	)

	# bigmatrix is an nx2p matrix, each row being an indpendent sample of (X, \tilde X).
	if verbose:
		print("The rejection rate is "+str(rejections/(p*n))+".")

	return X, Xk, rejections
