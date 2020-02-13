import numpy as np
import unittest
from .context import knockadapt

from knockadapt import utilities, graphs, knockoff_stats
from knockadapt.knockoff_stats import calc_data_dependent_threshhold

class TestGroupLasso(unittest.TestCase):
	""" Tests fitting of group lasso """

	def test_LCD(self):

		# Fake data
		Z = np.array([-1, -2, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0])
		groups = np.array([1, 1, 1, 2, 2, 2])
		W = knockoff_stats.calc_LCD(Z, groups)
		np.testing.assert_array_almost_equal(
			W, np.array([4, 0]), decimal = 3,
			err_msg = 'calc_LCD function incorrectly calculates group LCD'
		)

		# Again
		Z2 = np.array([0, 1, 2, 3, -1, -2, -3, -4])
		groups2 = np.array([1, 2, 3, 4])
		W2 = knockoff_stats.calc_LCD(Z2, groups2)
		np.testing.assert_array_almost_equal(
			W2, np.array([-1, -1, -1, -1]),
			err_msg = 'calc_LCD function incorrectly calculates group LCD'
		)

	def test_LCD(self):

		# Fake data
		Z = np.array([-1, -2, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0])
		groups = np.array([1, 1, 1, 2, 2, 2])
		W = knockoff_stats.calc_LCD(Z, groups)
		np.testing.assert_array_almost_equal(
			W, np.array([4, 0]), decimal = 3,
			err_msg = 'calc_LCD function incorrectly calculates group LCD'
		)

		# Again
		Z2 = np.array([0, 1, 2, 3, -1, -2, -3, -4])
		groups2 = np.array([1, 2, 3, 4])
		W2 = knockoff_stats.calc_LCD(Z2, groups2)
		np.testing.assert_array_almost_equal(
			W2, np.array([-1, -1, -1, -1]),
			err_msg = 'calc_LCD function incorrectly calculates group LCD'
		)

	def test_corr_diff(self):

		# Fake data (p = 5)
		n = 10000
		p = 5
		X = np.random.randn(n, p)
		knockoffs = np.random.randn(n, p)
		groups = np.array([1, 1, 2, 2, 2])

		# Calc y
		beta = np.array([1, 1, 0, 0, 0])
		y = np.dot(X, beta.reshape(-1, 1))

		# Correlations
		W = knockoff_stats.marg_corr_diff(X, knockoffs, y, groups = None)

		self.assertTrue(
			np.abs(W[0] - 1/np.sqrt(2)) < 0.05, 
			msg = 'marg_corr_diff statistic calculates correlations incorrectly'
		)

	def test_linear_coef_diff(self):

		# Fake data (p = 5)
		n = 1000
		p = 5
		X = np.random.randn(n, p)
		knockoffs = np.random.randn(n, p)
		groups = np.array([1, 1, 2, 2, 2])

		# Calc y
		beta = np.array([1, 1, 0, 0, 0])
		y = np.dot(X, beta.reshape(-1, 1))

		# Correlations
		W = knockoff_stats.linear_coef_diff(X, knockoffs, y, groups = groups)

		np.testing.assert_array_almost_equal(
			W, np.array([2, 0]), decimal = 3
		)



	def test_gaussian_fit(self):

		n = 500
		p = 200
		np.random.seed(110)
		X, y, beta, Q, corr_matrix, groups = graphs.daibarber2016_graph(
			n = n, p = p, y_dist = 'gaussian'
		)
		fake_knockoffs = np.random.randn(X.shape[0], X.shape[1])

		# Get best model for pyglmnet
		glasso1, rev_inds1 = knockoff_stats.fit_group_lasso(
			X, fake_knockoffs, y, groups = groups,
			use_pyglm = True, y_dist = 'gaussian',
			max_iter = 20, tol = 5e-2, learning_rate = 3
		)
		beta_pyglm = glasso1.beta_[rev_inds1][0:p]
		corr1 = np.corrcoef(beta_pyglm, beta)[0, 1]
		self.assertTrue(corr1 > 0.5,
						msg = f'Pyglm fits gauissan very poorly (corr = {corr1} btwn real/fitted coeffs)'
		)

		# Test again, fitting regular lasso
		glasso2, rev_inds2 = knockoff_stats.fit_lasso(
			X, fake_knockoffs, y, y_dist = 'gaussian', max_iter = 50
		)
		beta2 = glasso2.coef_[rev_inds2][0:p]
		corr2 = np.corrcoef(beta2, beta)[0, 1]

		self.assertTrue(corr2 > 0.5,
						msg = f'SKlearn lasso fits gaussian very poorly (corr = {corr2} btwn real/fitted coeffs)'
		)


	def test_logistic_fit(self):
		""" Tests logistic fit of group lasso on an easy case
		(dai barber dataset). If this test fails, knockoffs will
		have pretty atrocious power on binary outcomes. """

		n = 300
		p = 100
		np.random.seed(110)
		X, y, beta, Q, corr_matrix, groups = graphs.daibarber2016_graph(
			n = n, p = p, y_dist = 'binomial'
		)

		# These are not real - just helpful syntactically
		fake_knockoffs = X

		# Get best model for pyglmnet
		glasso1, rev_inds1 = knockoff_stats.fit_group_lasso(
			X, fake_knockoffs, y, groups = groups,
			use_pyglm = True, y_dist = 'binomial',
			max_iter = 20, tol = 5e-2, learning_rate = 3
		)
		beta_pyglm = glasso1.beta_[rev_inds1][0:p]
		corr1 = np.corrcoef(beta_pyglm, beta)[0, 1]
		self.assertTrue(corr1 > 0.5,
						msg = f'Pyglm fits logistic very poorly (corr = {corr1} btwn real/fitted coeffs)')


		# Get best model for group-lasso
		glasso2, rev_inds2 = knockoff_stats.fit_group_lasso(
			X, fake_knockoffs, y, groups = groups,
			use_pyglm = False, y_dist = 'binomial',
			max_iter = 20, tol = 5e-2, learning_rate = 3
		)
		beta_gl = glasso2.coef_[rev_inds2][0:p].reshape(p)
		corr2 = np.corrcoef(beta_gl, beta)[0, 1]
		self.assertTrue(corr2 > 0.5,
						msg = f'group-lasso fits logistic very poorly (corr = {corr2} btwn real/fitted coeffs)')


		# Test again, fitting regular lasso
		glasso3, rev_inds3 = knockoff_stats.fit_lasso(
			X = X, knockoffs = fake_knockoffs, y = y, y_dist = 'binomial',
			max_iter = 50
		)
		beta3 = glasso3.coef_[0, rev_inds3][0:p]
		corr3 = np.corrcoef(beta3, beta)[0, 1]
		self.assertTrue(corr3 > 0.5,
						msg = f'SKlearn lasso fits logistic very poorly (corr = {corr3} btwn real/fitted coeffs)'
		)


class TestDataThreshhold(unittest.TestCase):
	""" Tests data-dependent threshhold """

	def test_unbatched(self):

		W1 = np.array([1, -2, 3, 6, 3, -2, 1, 2, 5, 3, 0.5, 1, 1, 1, 1, 1, 1, 1])
		T1 = calc_data_dependent_threshhold(W1, fdr = 0.2)
		self.assertTrue(T1==0, msg=f'Incorrect data dependent threshhold: T1 should be 0, not {T1}')

		W2 = np.array([-1, -2, -3])
		T2 = calc_data_dependent_threshhold(W2, fdr = 0.3)
		self.assertTrue(T2==np.inf, msg=f'Incorrect data dependent threshhold: T2 should be inf, not {T2}')

		W3 = np.array([-5, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
		T3 = calc_data_dependent_threshhold(W3, fdr = 0.2)
		self.assertTrue(T3 == 5, msg=f'Incorrect data dependent threshhold: T3 should be 5, not {T3}')

	def test_batched(self):

		W1 = np.array([1]*10)
		W2 = np.array([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
		W3 = np.array([-1]*10)
		combined = np.stack([W1, W2, W3]).transpose()
		Ts = calc_data_dependent_threshhold(combined, fdr = 0.2)
		expected = np.array([0, 2, np.inf])
		np.testing.assert_array_almost_equal(
			Ts, expected, 
			err_msg = f"Incorrect data dependent threshhold (batched): Ts should be {expected}, not {Ts}"
		)

if __name__ == '__main__':
	unittest.main()