import numpy as np
import unittest
from .context import knockadapt

from knockadapt import graphs

class TestSampleData(unittest.TestCase):
	""" Tests sample_data function """

	def test_logistic(self):

		X, y, beta, Q, corr_matrix = graphs.sample_data(
			y_dist = 'binomial'
		)

		# Test outputs are binary
		y_vals = np.unique(y)
		np.testing.assert_array_almost_equal(
			y_vals, np.array([0, 1]), 
			err_msg = 'Binomial flag not producing binary responses' 
		)

		# Test conditional mean correlation

	def test_beta_gen(self):


		# Test sparsity
		_, _, beta, _, _ = graphs.sample_data(
			p = 100, sparsity = 0.3, coeff_size = 0.3,
		)
		self.assertTrue((beta != 0).sum() == 30,
						msg = 'sparsity parameter yields incorrect sparsity')
		abs_coefs = np.unique(np.abs(beta[beta != 0]))
		np.testing.assert_array_almost_equal(
			abs_coefs, np.array([0.3]), 
			err_msg = 'beta generation yields incorrect coefficients'
		)

		# Test number of selections for groups
		k = 10
		groups = np.concatenate(
			[np.arange(0, 50, 1), np.arange(0, 50, 1)]
		)
		_, _, beta, _, _ = graphs.sample_data(
			p = 100, sparsity = 0.5, groups = groups,
			k = k,
		)
		self.assertTrue((beta != 0).sum() == 2*k,
						msg = 'sparsity for groups chooses incorrect number of features')
		selected_groups = np.unique(groups[beta != 0]).shape[0]
		self.assertTrue(selected_groups == k,
						msg = 'group sparsity parameter does not choose coeffs within a group' )




if __name__ == '__main__':
	unittest.main()