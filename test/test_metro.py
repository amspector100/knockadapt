import pytest
import numpy as np
import networkx as nx
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt import metro, tree_processing
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
			gamma=0.9999
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




if __name__ == '__main__':
	unittest.main()