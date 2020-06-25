import pytest
import numpy as np
import networkx as nx
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt import metro, metro_generic, tree_processing
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
		custom_result = metro_generic.gaussian_log_likelihood(X, mu, var)
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
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
			lf=lambda x: np.log(x).sum(),
			X=X,
			mu=np.zeros(p),
			V=V,
			order=np.arange(0, p, 1),
			active_frontier=[[] for _ in range(p)],
			S=np.eye(p),
		)

		# Test that invSigma is as it should be
		G = metro_sampler.G
		for j in [1, int(p/2), p-1]:
			Gjinv = np.linalg.inv(G[0:p+j, 0:p+j])
			np.testing.assert_almost_equal(
				Gjinv, metro_sampler.invSigmas[j], decimal=3,
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
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
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
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
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



if __name__ == '__main__':
	unittest.main()