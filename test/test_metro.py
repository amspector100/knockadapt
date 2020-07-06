import os
import pytest
import numpy as np
import networkx as nx
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt import metro, tree_processing
from knockadapt.nonconvex_sdp import fk_precision_trace
import time

class TestMetroProposal(unittest.TestCase):

	def test_gaussian_likelihood(self):

		X = np.array([0.5, 1, 2, 3])
		mu = 0.5
		var = 0.2

		# Scipy result
		norm_rv = stats.norm(loc=mu, scale=np.sqrt(var))
		sp_result = norm_rv.logpdf(X)

		# Custom result
		custom_result = metro.gaussian_log_likelihood(X, mu, var)
		self.assertTrue(
			np.abs(sp_result-custom_result).sum() < 0.001,
			msg=f'scipy result {sp_result} does not match custom result {custom_result}'
		)


	def test_proposal_covs(self):

		# Fake data 
		np.random.seed(110)
		n = 5
		p = 200
		X,_,_,Q,V = graphs.sample_data(method='AR1', rho=0.1, n=n, p=p)
		
		# Metro sampler, proposal params
		metro_sampler = metro.MetropolizedKnockoffSampler(
			lf=lambda x: np.log(x).sum(),
			X=X,
			mu=np.zeros(p),
			V=V,
			undir_graph = np.abs(Q) > 1e-3,
			S=np.eye(p),
		)

		# Test that invSigma is as it should be
		G = metro_sampler.G
		for j in [p-1]:
			Gjinv = np.linalg.inv(G[0:p+j, 0:p+j])
			np.testing.assert_almost_equal(
				Gjinv, metro_sampler.invSigma, decimal=3,
				err_msg=f'Metro sampler fails to correctly calculate {j}th invSigma'
			)

		# Test that proposal likelihood is correct
		mu = np.zeros(2*p)
		mvn = stats.multivariate_normal(
			mean=mu, cov=metro_sampler.G
		)

		# Scipy likelihood
		features = mvn.rvs()
		scipy_likelihood = mvn.logpdf(features)

		# Calculate a new likelihood using the proposal params
		X = features[0:p].reshape(1, -1)
		Xstar = features[p:].reshape(1, -1)

		# Base likelihood for first p variables
		loglike = stats.multivariate_normal(
			mean=np.zeros(p), cov=V
		).logpdf(X)

		# Likelihood of jth variable given first j - 1
		prev_proposals = None
		for j in range(p):

			# Test = scipy likelihood at this point
			scipy_likelihood = stats.multivariate_normal(
				mean=np.zeros(p+j), 
				cov=metro_sampler.G[0:p+j, 0:p+j]
			).logpdf(features[0:p+j])
			self.assertTrue(
				np.abs(loglike-scipy_likelihood) < 0.001,
				f"Proposal likelihood for j={j-1} fails: output {loglike}, expected {scipy_likelihood} (scipy)"
			)

			# Add loglike
			loglike += metro_sampler.q_ll(
				Xjstar=Xstar[:, j],
				X=X, 
				prev_proposals=prev_proposals
			)
			prev_proposals = Xstar[:, 0:j+1]

	def test_compatibility_error(self):
		""" Ensures metro class errors when you pass a non-compatible
		proposal matrix """

		# Fake data 
		np.random.seed(110)
		n = 5
		p = 200
		X,_,_,Q,V = graphs.sample_data(method='AR1', rho=0.3, n=n, p=p)

		# Metro sampler, proposal params
		def incorrect_undir_graph():
			metro_sampler = metro.MetropolizedKnockoffSampler(
				lf=lambda x: np.log(x).sum(),
				X=X,
				mu=np.zeros(p),
				V=V,
				undir_graph=np.eye(p),
				S=np.eye(p),
			)
		# Make sure the value error increases 
		self.assertRaisesRegex(
			ValueError, "Precision matrix Q is not compatible",
			incorrect_undir_graph
		)

class TestMetroSample(unittest.TestCase):


	def test_ar1_sample(self):

		# Fake data
		np.random.seed(110)
		n = 30000
		p = 8
		X,_,_,Q,V = graphs.sample_data(method='AR1', n=n, p=p)
		_, S = knockadapt.knockoffs.gaussian_knockoffs(
			X=X, Sigma=V, method='mcv', return_S=True
		)

		# Graph structure + junction tree
		Q_graph = (np.abs(Q) > 1e-5)
		Q_graph = Q_graph - np.eye(p)

		# Metro sampler + likelihood
		mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)
		def mvn_likelihood(X):
			return mvn.logpdf(X)
		gamma = 0.9999
		metro_sampler = metro.MetropolizedKnockoffSampler(
			lf=mvn_likelihood,
			X=X,
			mu=np.zeros(p),
			V=V,
			undir_graph=Q_graph,
			S=S,
			gamma=gamma,
		)

		# Output knockoffs
		Xk = metro_sampler.sample_knockoffs()

		# Acceptance rate should be exactly one
		acc_rate = metro_sampler.final_acc_probs.mean()
		self.assertTrue(
			acc_rate - gamma > -1e-3, 
			msg = f'For AR1 gaussian design, metro has acc_rate={acc_rate} < gamma={gamma}'
		)

		# Check covariance matrix
		features = np.concatenate([X, Xk], axis=1)
		emp_corr_matrix = np.corrcoef(features.T)
		G = np.concatenate(
			[np.concatenate([V, V-S]),
			 np.concatenate([V-S, V]),], 
			axis=1
		)

		np.testing.assert_almost_equal(
			emp_corr_matrix, G, decimal=2,
			err_msg=f"For AR1 gaussian design, metro does not match theoretical matrix"
		)


	def test_dense_sample(self):

		# Fake data
		np.random.seed(110)
		n = 10000
		p = 4

		X,_,_,Q,V = graphs.sample_data(
			method='daibarber2016',
			rho=0.6, n=n, p=p,
			gamma=1, group_size=p
		)
		_, S = knockadapt.knockoffs.gaussian_knockoffs(
			X=X, Sigma=V, method='mcv', return_S=True
		)

		# Network graph
		Q_graph = (np.abs(Q) > 1e-5)
		Q_graph = Q_graph - np.eye(p)
		undir_graph = nx.Graph(Q_graph)
		width, T = tree_processing.treewidth_decomp(undir_graph)
		order, active_frontier = tree_processing.get_ordering(T)

		# Metro sampler and likelihood
		mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)
		def mvn_likelihood(X):
			return mvn.logpdf(X)
		gamma = 0.99999
		metro_sampler = metro.MetropolizedKnockoffSampler(
			lf=mvn_likelihood,
			X=X,
			mu=np.zeros(p),
			V=V,
			order=order,
			active_frontier=active_frontier,
			gamma=gamma,
			S=S,
			metro_verbose=True
		)

		# Output knockoffs
		Xk = metro_sampler.sample_knockoffs()

		# Acceptance rate should be exactly one
		acc_rate = metro_sampler.final_acc_probs.mean()
		self.assertTrue(
			acc_rate - gamma > -1e-3, 
			msg = f'For equi gaussian design, metro has acc_rate={acc_rate} < gamma={gamma}'
		)

		# Check covariance matrix
		features = np.concatenate([X, Xk], axis=1)
		emp_corr_matrix = np.corrcoef(features.T)
		G = np.concatenate(
			[np.concatenate([V, V-S]),
			 np.concatenate([V-S, V]),], 
			axis=1
		)

		np.testing.assert_almost_equal(
			emp_corr_matrix, G, decimal=2,
			err_msg=f"For equi gaussian design, metro does not match theoretical matrix"
		)

class TestARTK(unittest.TestCase):

	def test_t_log_likelihood(self):

		# Fake data
		np.random.seed(110)
		n = 15
		p = 10
		df_t = 5
		X1 = np.random.randn(n, p)
		X2 = np.random.randn(n, p)

		# Scipy ratios
		sp_like1 = stats.t.logpdf(X1, df=df_t)
		sp_like2 = stats.t.logpdf(X2, df=df_t)
		sp_diff = sp_like1 - sp_like2

		# Custom ratios
		custom_like1 = metro.t_log_likelihood(X1, df_t=df_t)
		custom_like2 = metro.t_log_likelihood(X2, df_t=df_t)
		custom_diff = custom_like1 - custom_like2

		np.testing.assert_almost_equal(
				custom_diff, sp_diff, decimal=2,
				err_msg=f"custom t_log_likelihood and scipy t.logpdf disagree"
		)

	def test_tmarkov_likelihood(self):

		# Data
		np.random.seed(110)
		n = 15
		p = 10
		df_t = 5
		X1 = np.random.randn(n, p)
		X2 = np.random.randn(n, p)
		V = np.eye(p)
		Q = np.eye(p)

		# Scipy likelihood ratio for X, scale matrix
		inv_scale = np.sqrt(df_t / (df_t - 2))
		sp_like1 = stats.t.logpdf(inv_scale*X1, df=df_t).sum(axis=1)
		sp_like2 = stats.t.logpdf(inv_scale*X2, df=df_t).sum(axis=1)
		sp_ratio = sp_like1 - sp_like2

		# General likelihood
		rhos = np.zeros(p-1)
		ar1_like1 = metro.t_markov_loglike(X1, rhos, df_t=df_t)
		ar1_like2 = metro.t_markov_loglike(X2, rhos, df_t=df_t)
		ar1_ratio = ar1_like1 - ar1_like2

		self.assertTrue(
			np.abs(ar1_ratio - sp_ratio).sum() < 0.01,
			f"AR1 ratio {ar1_ratio} and scipy ratio {sp_ratio} disagree for independent t vars"
		)

		# Test again with df_t --> infinity, so it should be approx gaussian
		X1,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, p=p, method='AR1', a=3, b=1
		)
		X2 = np.random.randn(n, p)

		# Ratio using normals
		df_t = 100000
		mu = np.zeros(p)
		norm_like1 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X1)
		norm_like2 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X2)
		norm_ratio = norm_like1 - norm_like2

		# Ratios using T
		rhos = np.diag(V, 1)
		ar1_like1 = metro.t_markov_loglike(X1, rhos, df_t=df_t)
		ar1_like2 = metro.t_markov_loglike(X2, rhos, df_t=df_t)
		ar1_ratio = ar1_like1 - ar1_like2

		self.assertTrue(
			np.abs(ar1_ratio - norm_ratio).mean() < 0.01,
			f"AR1 ratio {ar1_ratio} and gaussian ratio {norm_ratio} disagree for corr. t vars, df={df_t}"
		)

		# Check consistency of tsampler class
		tsampler = metro.ARTKSampler(
			X=X1,
			V=V,
			df_t=df_t,
		)
		new_ar1_like1 = tsampler.lf(tsampler.X)
		self.assertTrue(
			np.abs(ar1_like1 - new_ar1_like1).sum() < 0.01,
			f"AR1 loglike inconsistent between class ({new_ar1_like1}) and function ({ar1_ratio})"
		)

	def test_tmarkov_samples(self):

		# Test to make sure low df --> heavy tails
		# and therefore acceptances < 1
		np.random.seed(110)
		n = 1000000
		p = 5
		df_t = 5
		X,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, p=p, method='AR1', rho=0.3, x_dist='ar1t'
		)
		S = np.eye(p)

		# Sample t 
		tsampler = metro.ARTKSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S,
			metro_verbose=True
		)

		# Correct junction tree
		self.assertTrue(
			tsampler.width==1, 
			f"tsampler should have width 1, not {tsampler.width}" 
		)

		# Sample
		Xk = tsampler.sample_knockoffs()

		# Check empirical means
		# Check empirical covariance matrix
		muk_hat = np.mean(Xk, axis=0)
		np.testing.assert_almost_equal(
			muk_hat, np.zeros(p), decimal=2,
			err_msg=f"For ARTK sampler, empirical mean of Xk does not match mean of X" 
		)

		# Check empirical covariance matrix
		Vk_hat = np.corrcoef(Xk.T)
		np.testing.assert_almost_equal(
			V, Vk_hat, decimal=2,
			err_msg=f"For ARTK sampler, empirical covariance of Xk does not match cov of X" 
		)

		# Check that marginal fourth moments match
		X4th = np.mean(np.power(X, 4), axis=0)
		Xk4th = np.mean(np.power(Xk, 4), axis=0)
		np.testing.assert_almost_equal(
			X4th / 10, Xk4th / 10, decimal=1,
			err_msg=f"For ARTK sampler, fourth moment of Xk does not match theoretical fourth moment" 
		)


class TestBlockT(unittest.TestCase):

	def test_tmvn_log_likelihood(self):

		# Fake data
		np.random.seed(110)
		n = 10
		p = 10
		df_t = 100000

		# Test that the likelihood --> gaussian as df_t --> infinity
		X1,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, p=p, method='daibarber2016', gamma=0.3, rho=0.8, x_dist='blockt'
		)
		X2 = np.random.randn(n, p)


		# Ratio using normals
		mu = np.zeros(p)
		norm_like1 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X1)
		norm_like2 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X2)
		norm_ratio = norm_like1 - norm_like2

		# Ratios using T
		tmvn_like1 = metro.t_mvn_loglike(X1, Q, df_t=df_t)
		tmvn_like2 = metro.t_mvn_loglike(X2, Q, df_t=df_t)
		tmvn_ratio = tmvn_like1 - tmvn_like2
		self.assertTrue(
			np.abs(tmvn_ratio - norm_ratio).mean() < 0.01,
			f"T MVN ratio {tmvn_ratio} and gaussian ratio {norm_ratio} disagree for corr. t vars, df={df_t}"
		)


	def test_blockt_samples(self):

		# Test to make sure low df --> heavy tails
		# and therefore acceptances < 1
		np.random.seed(110)
		n = 2000000
		p = 6
		df_t = 5
		X,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, 
			p=p,
			method='daibarber2016',
			rho=0.4,
			gamma=0,
			group_size=3,
			x_dist='blockt',
			df_t=df_t,
		)
		S = np.eye(p)

		# Sample t 
		tsampler = metro.BlockTSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S,
			metro_verbose=True
		)

		# Sample
		Xk = tsampler.sample_knockoffs()

		# Check empirical means
		# Check empirical covariance matrix
		muk_hat = np.mean(Xk, axis=0)
		np.testing.assert_almost_equal(
			muk_hat, np.zeros(p), decimal=2,
			err_msg=f"For block T sampler, empirical mean of Xk does not match mean of X" 
		)

		# Check empirical covariance matrix
		Vk_hat = np.cov(Xk.T)
		np.testing.assert_almost_equal(
			V, Vk_hat, decimal=2,
			err_msg=f"For block T sampler, empirical covariance of Xk does not match cov of X" 
		)

		# Check that marginal fourth moments match
		X4th = np.mean(np.power(X, 4), axis=0)
		Xk4th = np.mean(np.power(Xk, 4), axis=0)
		np.testing.assert_almost_equal(
			X4th / 10, Xk4th / 10, decimal=1,
			err_msg=f"For block T sampler, fourth moment of Xk does not match theoretical fourth moment" 
		)

class TestIsing(unittest.TestCase):

	def test_divconquer_likelihoods(self):

		# Test to make sure the way we split up
		# cliques does not change the likelihood
		np.random.seed(110)
		n = 10
		p = 625
		mu = np.zeros(p)
		X,_,_,undir_graph,_ = knockadapt.graphs.sample_data(
			n=n, 
			p=p,
			method='ising',
			x_dist='gibbs',
		)
		np.fill_diagonal(undir_graph, 1)

		# Read V
		file_directory = os.path.dirname(os.path.abspath(__file__))
		V = np.loadtxt(f'{file_directory}/test_covs/vout{p}.txt')

		# Initialize sampler
		metro_sampler = metro.IsingKnockoffSampler(
			X=X,
			undir_graph=undir_graph,
			mu=mu,
			V=V,
			max_width=2,
		)

		# Non-divided likelihood
		nondiv_like = 0
		for clique, lp in zip(metro_sampler.cliques, metro_sampler.log_potentials):
			nondiv_like += lp(X[:, np.array(clique)])
		
		# Divided likelihood for the many keys
		many_div_like = np.zeros(n)
		for dc_key in metro_sampler.dc_keys:
			# Initialize likelihood for these data points
			div_like = 0
			# Helpful constants
			seps = metro_sampler.separators[dc_key]
			n_inds = metro_sampler.X_ninds[dc_key]
			# Add separator-to-separator cliques manually
			for clique, lp in zip(metro_sampler.cliques, metro_sampler.log_potentials):
				if clique[0] not in seps or clique[1] not in seps:
					continue
				sepX = X[n_inds]
				div_like += lp(sepX[:, np.array(clique)])

			# Now loop through other blocks
			div_dict_list = metro_sampler.divconq_info[dc_key]
			for block_dict in div_dict_list:
				blockX = X[n_inds][:, block_dict['inds']]
				for clique, lp in zip(block_dict['cliques'], block_dict['lps']):
					div_like +=  lp(blockX[:, clique])
			many_div_like[n_inds] = np.array(div_like)

		# Test to make sure these likelihoods agree
		np.testing.assert_almost_equal(
			nondiv_like, many_div_like, decimal=5,
			err_msg=f"Non-divided clique potentials {nondiv_like} do not agree with divided cliques {div_like}"
		)

	def test_large_ising_samples(self):

		# Test that sampling does not throw an error
		np.random.seed(110)
		n = 100
		p = 625
		mu = np.zeros(p)
		X,_,_,undir_graph,_ = knockadapt.graphs.sample_data(
			n=n, 
			p=p,
			method='ising',
			x_dist='gibbs',
		)
		np.fill_diagonal(undir_graph, 1)

		# We load custom cov/q matrices for this
		file_directory = os.path.dirname(os.path.abspath(__file__))
		V = np.loadtxt(f'{file_directory}/test_covs/vout{p}.txt')
		Q = np.loadtxt(f'{file_directory}/test_covs/qout{p}.txt')
		max_nonedge = np.max(np.abs(Q[undir_graph == 0]))
		self.assertTrue(
			max_nonedge < 1e-5,
			f"Estimated precision for ising{p} has max_nonedge {max_nonedge} >= 1e-5"
		)

		# Initialize sampler
		metro_sampler = metro.IsingKnockoffSampler(
			X=X,
			undir_graph=undir_graph,
			mu=mu,
			V=V,
			Q=Q,
			max_width=4,
			method='equicorrelated',
		)

		# Sample and hope for no errors
		Xk = metro_sampler.sample_knockoffs()

	def test_small_ising_samples(self):

		# Test samples to make sure the 
		# knockoff properties hold
		np.random.seed(110)
		n = 100000
		p = 9
		mu = np.zeros(p)
		X,_,_,undir_graph,_ = knockadapt.graphs.sample_data(
			n=n, 
			p=p,
			method='ising',
			x_dist='gibbs',
		)
		np.fill_diagonal(undir_graph, 1)

		# We load custom cov/q matrices for this
		file_directory = os.path.dirname(os.path.abspath(__file__))
		V = np.loadtxt(f'{file_directory}/test_covs/vout{p}.txt')
		Q = np.loadtxt(f'{file_directory}/test_covs/qout{p}.txt')
		max_nonedge = np.max(np.abs(Q[undir_graph == 0]))
		self.assertTrue(
			max_nonedge < 1e-5,
			f"Estimated precision for ising{p} has max_nonedge {max_nonedge} >= 1e-5"
		)

		# Initialize sampler
		metro_sampler = metro.IsingKnockoffSampler(
			X=X,
			undir_graph=undir_graph,
			mu=mu,
			V=V,
			Q=Q,
			max_width=2,
		)

		# Sample
		Xk = metro_sampler.sample_knockoffs()

		# Check empirical means
		# Check empirical covariance matrix
		mu_hat = X.mean(axis=0)
		muk_hat = np.mean(Xk, axis=0)
		np.testing.assert_almost_equal(
			muk_hat, mu_hat, decimal=2,
			err_msg=f"For Ising sampler, empirical mean of Xk does not match mean of X" 
		)

		# Check empirical covariance matrix
		V_hat = np.cov(X.T)
		Vk_hat = np.cov(Xk.T)
		np.testing.assert_almost_equal(
			V_hat / 2, Vk_hat / 2, decimal=1,
			err_msg=f"For Ising sampler, empirical covariance of Xk does not match cov of X" 
		)

		# Check that marginal fourth moments match
		X4th = np.mean(np.power(X, 4), axis=0)
		Xk4th = np.mean(np.power(Xk, 4), axis=0)
		np.testing.assert_almost_equal(
			X4th / 10, Xk4th / 10, decimal=1,
			err_msg=f"For Ising sampler, fourth moment of Xk does not match theoretical fourth moment" 
		)

class TestEICV(unittest.TestCase):

	def test_attribute_consistency(self):
		"""
		Tests that EICV method appropriately resets 
		acc_prob, Xprop, and F dict after computation
		"""

		# Data generating process
		np.random.seed(110)
		n = 7
		p = 6
		df_t = 3
		X,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, 
			p=p,
			method='daibarber2016',
			rho=0.4,
			gamma=1,
			group_size=5,
			x_dist='blockt',
			df_t=df_t,
		)
		S = np.eye(p)

		# Sample t 
		tsampler = metro.BlockTSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S,
			metro_verbose=True
		)
		tsampler.sample_knockoffs()
		
		# Get some attributes
		Xk = tsampler.samplers[0].Xk.copy()
		Xprop = tsampler.samplers[0].X_prop.copy()
		accs = tsampler.samplers[0].acceptances.copy()
		acc_probs = tsampler.samplers[0].final_acc_probs.copy()
		boolkey = np.zeros((n, p))
		boolkey[:, 2] = 1
		key = tsampler.samplers[0].get_key(boolkey, 3)
		sample_acc_prob = tsampler.samplers[0].F_dicts[3][key]
		sample_F_val = tsampler.samplers[0].acc_dicts[3][key]

		# EICV for j = 0
		tsampler.samplers[0].estimate_single_ECV(j=0, B=10)
		tsampler.samplers[0].estimate_single_ECV(j=3, B=3)

		# Check that we get the same answer
		Xk2 = tsampler.samplers[0].Xk
		Xprop2 = tsampler.samplers[0].X_prop
		accs2 = tsampler.samplers[0].acceptances
		acc_probs2 = tsampler.samplers[0].final_acc_probs.copy()
		sample_acc_prob2 = tsampler.samplers[0].F_dicts[3][key]
		sample_F_val2 = tsampler.samplers[0].acc_dicts[3][key]

		np.testing.assert_almost_equal(
			Xk, Xk2, decimal=6, 
			err_msg=f"Xk is not consistent before / after estimating EICV"
		)
		np.testing.assert_almost_equal(
			Xprop, Xprop2, decimal=6, 
			err_msg=f"Xprop is not consistent before / after estimating EICV"
		)
		np.testing.assert_almost_equal(
			accs, accs2, decimal=6, 
			err_msg=f"accs is not consistent before / after estimating EICV"
		)
		np.testing.assert_almost_equal(
			acc_probs, acc_probs2, decimal=6, 
			err_msg=f"acc_probs is not consistent before / after estimating EICV"
		)
		np.testing.assert_almost_equal(
			sample_acc_prob, sample_acc_prob2, decimal=6, 
			err_msg=f"acc_dict is not consistent before / after estimating EICV"
		)
		np.testing.assert_almost_equal(
			sample_F_val, sample_F_val2, decimal=6, 
			err_msg=f"F_dict is not consistent before / after estimating EICV"
		)


	def test_gaussian_ecv(self):
		"""
		Checks that gaussian eicv matches theoretical eicv
		"""

		# Fake data
		np.random.seed(110)
		n = 30000
		p = 5
		X,_,_,Q,V = graphs.sample_data(method='AR1', rho=0.3, n=n, p=p)
		_, S = knockadapt.knockoffs.gaussian_knockoffs(
			X=X, Sigma=V, method='mcv', return_S=True
		)

		# Graph structure + junction tree
		Q_graph = (np.abs(Q) > 1e-5)
		Q_graph = Q_graph - np.eye(p)

		# Metro sampler + likelihood
		mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)
		def mvn_likelihood(X):
			return mvn.logpdf(X)
		gamma = 0.9999999
		metro_sampler = metro.MetropolizedKnockoffSampler(
			lf=mvn_likelihood,
			X=X,
			mu=np.zeros(p),
			V=V,
			undir_graph=Q_graph,
			S=S,
			gamma=gamma,
		)

		# Output knockoffs
		Xk = metro_sampler.sample_knockoffs()
		for j in range(p):
			ECV, _, _ = metro_sampler.estimate_single_ECV(j=j, B=5)
			self.assertTrue(
				np.abs(1 / ECV - metro_sampler.invG[j, j]) < 1e-2,
				f"For gaussian case, j={j}, 1 / ECV (1 / {ECV}) disagrees with ground truth {metro_sampler.invG[j,j]} "
			)

		# Check EICV approx equal FK precision trace
		EICV = metro_sampler.estimate_EICV(B=5)
		expected = fk_precision_trace(Sigma=V, S=S, invSigma=Q)
		self.assertTrue(
			np.abs(EICV - expected) < 1e-2, 
			f"For gaussian case, estimated EICV ({EICV}) != ground truth {expected}"
		)
		
	def test_artk_ecv(self):

		# Test to make sure acceptances < 1
		np.random.seed(110)
		n = 10000
		p = 7
		df_t = 5
		X,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, p=p, method='AR1', rho=0.3, x_dist='ar1t'
		)
		S = np.eye(p)

		# Sample t 
		tsampler = metro.ARTKSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S,
			metro_verbose=True
		)
		tsampler.sample_knockoffs()
		Xk = tsampler.Xk

		# Resample using ECV
		j = 1
		_, _, new_Xkj = tsampler.estimate_single_ECV(j=j, B=10)
		
		# Check empirical means
		muk_hat = Xk[:, j].mean()
		muk_resampled = np.mean(new_Xkj[:, 0], axis=0)
		np.testing.assert_almost_equal(
			muk_hat, muk_resampled, decimal=2,
			err_msg=f"For ARTK sampler, means of resampled Xk do not match mean of X" 
		)

		# Check that marginal fourth moments match
		Xk4th = np.power(Xk[:, j], 4).mean()
		Xk4th_resampled = np.power(new_Xkj[:, 1], 4).mean()
		np.testing.assert_almost_equal(
			Xk4th / 10, Xk4th_resampled / 10, decimal=1,
			err_msg=f"For ARTK sampler, fourth moment of resampled Xk do not those of Xk" 
		)

	def test_blockt_ecvs(self):

		# Check with one S matrix
		np.random.seed(110)
		n = 10000
		p = 10

		# Sample t 
		df_t = 5
		rho = 0.5
		X,_,_,Q,V = knockadapt.graphs.sample_data(
			n=n, 
			p=p,
			method='daibarber2016',
			rho=rho,
			gamma=0,
			group_size=5,
			x_dist='blockt',
			df_t=df_t,
		)

		# Sample for one S matrix
		S = (1-0.01)*np.eye(p)
		tsampler = metro.BlockTSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S,
			metro_verbose=True
		)
		tsampler.sample_knockoffs()
		EICV_SDP = tsampler.estimate_EICV()

		# Sample for MCV matrix
		S_MCV = (1-rho)*np.eye(p)
		tsampler = metro.BlockTSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S_MCV,
			metro_verbose=True
		)
		tsampler.sample_knockoffs()
		EICV_MCV = tsampler.estimate_EICV()
		self.assertTrue(
			EICV_MCV < EICV_SDP,
			f"Unexpectedly SDP EICV ({EICV_SDP}) performs better for block T (vs {EICV_MCV})"
		)

if __name__ == '__main__':
	unittest.main()