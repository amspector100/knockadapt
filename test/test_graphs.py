import numpy as np
import scipy as sp
import unittest
from .context import knockadapt


from knockadapt import graphs

class TestSampleData(unittest.TestCase):
	""" Tests sample_data function """

	def test_logistic(self):

		np.random.seed(110)

		p = 50
		X, y, beta, Q, corr_matrix = graphs.sample_data(
			p = p, y_dist = 'binomial'
		)

		# Test outputs are binary
		y_vals = np.unique(y)
		np.testing.assert_array_almost_equal(
			y_vals, np.array([0, 1]), 
			err_msg = 'Binomial flag not producing binary responses' 
		)

		# Test conditional mean for a single X val - start by 
		# sampling ys
		N = 5000
		X_repeated = np.repeat(X[0], N).reshape(p, N).T
		ys = graphs.sample_glm_response(
			X_repeated, beta, y_dist = 'binomial'
		)

		# Then check that the theoretical/empirical mean are the same
		cond_mean = 1/(1+np.exp(-1*np.dot(X_repeated[0], beta)))
		emp_cond_mean = ys.mean(axis = 0)
		np.testing.assert_almost_equal(
			cond_mean, emp_cond_mean, decimal = 2
		)

	def test_beta_gen(self):


		# Test sparsity
		p = 100
		_, _, beta, _, _ = graphs.sample_data(
			p = p, sparsity = 0.3, coeff_size = 0.3,
		)
		self.assertTrue((beta != 0).sum() == 30,
						msg = 'sparsity parameter yields incorrect sparsity')
		abs_coefs = np.unique(np.abs(beta[beta != 0]))
		np.testing.assert_array_almost_equal(
			abs_coefs, np.array([0.3]), 
			err_msg = 'beta generation yields incorrect coefficients'
		)

		# Test number of selections for groups
		sparsity = 0.2
		groups = np.concatenate(
			[np.arange(0, 50, 1), np.arange(0, 50, 1)]
		)
		_, _, beta, _, _ = graphs.sample_data(
			p = p, sparsity=sparsity, groups = groups,
		)

		# First, test that the correct number of features is chosen
		num_groups = np.unique(groups).shape[0]
		expected_nonnull_features = sparsity*p
		self.assertTrue((beta != 0).sum() == expected_nonnull_features,
						msg = 'sparsity for groups chooses incorrect number of features')
		
		# Check that the correct number of GROUPS has been chosen
		expected_nonnull_groups = sparsity*num_groups
		selected_groups = np.unique(groups[beta != 0]).shape[0]
		self.assertTrue(selected_groups == expected_nonnull_groups,
						msg = 'group sparsity parameter does not choose coeffs within a group' )

	def test_coeff_dist(self):

		# Test normal
		np.random.seed(110)
		p = 1000
		_, _, beta, _, _ = graphs.sample_data(
			p = p, sparsity = 1, coeff_size=1,
			coeff_dist = 'normal', sign_prob = 0
		)
		expected = 1
		mean_est = beta.mean()
		self.assertTrue(
			np.abs(mean_est - expected) < 0.1,
			msg = f"coeff_dist (normal) mean is wrong: expected mean 1 but got mean {mean_est}"
		)


		# Test uniform
		np.random.seed(110)
		p = 1000
		_, _, beta, _, _ = graphs.sample_data(
			p = p, sparsity = 1, coeff_size=1,
			coeff_dist = 'uniform', sign_prob = 0
		)
		expected = 0.5
		mean_est = beta.mean()
		self.assertTrue(
			np.abs(mean_est - expected) < 0.1,
			msg = f"coeff_dist (uniform) mean is wrong: expected mean 1 but got mean {mean_est}"
		)
		mbeta = np.max(beta)
		self.assertTrue(
			mbeta <= 1,
			msg = f'coeff_dist (uniform) produces max beta abs of {mbeta} > 1 for coeff_size = 1'
		)

		# Test Value-Error
		def sample_bad_dist():
			graphs.sample_data(p = 100, coeff_dist = 'baddist')
		self.assertRaisesRegex(
			ValueError, " must be 'none', 'normal' or 'uniform'",
			sample_bad_dist
		)
	
	def test_beta_sign_prob(self):


		# Test signs of beta
		p = 100
		for sign_prob in [0, 1]:
			_, _, beta, _, _ = graphs.sample_data(
				p = p, sparsity = 1, coeff_size=1,
				sign_prob = sign_prob
			)
			sum_beta = beta.sum()
			expected = p*(1 - 2*sign_prob)
			self.assertTrue(
				sum_beta == expected,
				msg = f"sign_prob ({sign_prob}) fails to correctly control sign of beta"
			)


	def test_daibarber2016_sample(self):

		# Check that defaults are correct - start w cov matrix
		_, _, beta, _, V, _ = graphs.daibarber2016_graph()

		# Construct expected cov matrix -  this is a different
		# construction than the actual function
		def construct_expected_V(p, groupsize, rho, gamma):

			# Construct groups with rho ingroup correlation
			block = np.zeros((groupsize, groupsize)) + rho
			block += (1-rho)*np.eye(groupsize)
			blocks = [block for _ in range(int(p/groupsize))]
			expected = sp.linalg.block_diag(*blocks)

			# Add gamma between-group correlations
			expected[expected==0] = gamma*rho
			return expected

		expected = construct_expected_V(p=1000, groupsize=5, rho=0.5, gamma=0)

		# Test equality with actual one
		np.testing.assert_array_almost_equal(
			V, expected, err_msg = 'Default daibarber2016 cov matrix is incorrect'
		)

		# Check number of nonzero groups
		groupsize = 5
		nonzero_inds = np.arange(0, 1000, 1)[beta != 0]
		num_nonzero_groups = np.unique(nonzero_inds // 5).shape[0]
		self.assertTrue(
			num_nonzero_groups==20, 
			msg = f'Default daibarber2016 beta has {num_nonzero_groups} nonzero groups, expected 20'
		)

		# Check number of nonzero features
		num_nonzero_features = (beta != 0).sum()
		self.assertTrue(
			num_nonzero_features==100, 
			msg = f'Default daibarber2016 beta has {num_nonzero_features} nonzero features, expected 100'
		)

	def test_AR1_sample(self):

		# Check that rho parameter works
		rho = 0.3
		p = 500
		_,_,_,_,Sigma = graphs.sample_data(
			p=p, method='AR1', rho=rho
		)
		np.testing.assert_almost_equal(
			np.diag(Sigma, k=1), 
			np.array([rho for _ in range(p-1)]),
			decimal=4,
			err_msg="Rho parameter for AR1 graph sampling fails"
		)

		# Error testing
		def ARsample():
			graphs.sample_data(method='AR1', rho=1.5)
		self.assertRaisesRegex(
			ValueError, "must be a correlation between -1 and 1",
			ARsample
		)


		# Check that a, b parameters work
		np.random.seed(110)
		a = 100
		b = 100
		_,_,_,_,Sigma = graphs.sample_data(
			p=500, method='AR1', a=a, b=b
		)
		mean_rho = np.diag(Sigma, k=1).mean()
		expected = a/(a+b)
		np.testing.assert_almost_equal(
			mean_rho, a/(a+b),
			decimal=2,
			err_msg=f'random AR1 gen has unexpected avg rho {mean_rho} vs {expected} '
		)

if __name__ == '__main__':
	unittest.main()