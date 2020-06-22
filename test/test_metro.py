import pytest
import numpy as np
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt import metro, metro_generic
import time

class TestGenericMetro(unittest.TestCase):

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

		# Fake data + loss function
		np.random.seed(110)
		n = 5
		p = 100
		X,_,_,Q,V = graphs.sample_data(method='AR1', n=n, p=p)
		
		# Metro sampler, proposal params
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
			lf=lambda x: np.log(x).sum(),
			X=X,
			mu=np.zeros(p),
			V=V,
			order=np.arange(0, p, 1),
			active_frontier=[[] for _ in range(p)],
		)
		metro_sampler.create_proposal_params(sdp_verbose=False)

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
			)[0,0]
			prev_proposals = Xstar[:, 0:j+1]
# class TestFdrControl(unittest.TestCase):

# 	def test_t_generation(self):

# 		p = 3
# 		n = 100
# 		time0 = time.time()
# 		V = graphs.AR1(p=p, rho=0.5)
# 		X, Xk, rejection = metro.ar1t_knockoffs(
# 			n=n, Sigma=V, method='sdp', df_t=3,
# 		)


# 	def test_t_generation(self):

# 		p = 3
# 		n = 1000
# 		time0 = time.time()
# 		V = graphs.AR1(p=p, rho=0.5)
# 		X, Xk, rejection = metro.ar1t_knockoffs(
# 			n=n, Sigma=V, method='sdp', df_t=3,
# 		)
# 		print(time.time() - time0)
# 		print(Xk.shape)
# 		print(X.shape)
# 		raise ValueError()





if __name__ == '__main__':
	unittest.main()