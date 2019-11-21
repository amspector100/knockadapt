import numpy as np
import unittest
from .context import knockadapt

from knockadapt import utilities, graphs, knockoff_stats

class TestGroupLasso(unittest.TestCase):
	""" Tests fitting of group lasso """

	def test_logistic_fit(self):
		""" Tests logistic fit of group lasso on an easy case
		(dai barber dataset). If this test fails, knockoffs will
		have pretty atrocious power on binary outcomes. """

		n = 200
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
			use_pyglm = True, y_dist = 'binomial'
		)
		beta_pyglm = glasso1.beta_[rev_inds1][0:p]
		corr1 = np.corrcoef(beta_pyglm, beta)[0, 1]
		self.assertTrue(corr1 > 0.5,
						msg = f'Pyglm fits logistic very poorly (corr = {corr1} btwn real/fitted coeffs)')


		# Get best model for group-lasso
		glasso2, rev_inds2 = knockoff_stats.fit_group_lasso(
			X, fake_knockoffs, y, groups = groups,
			use_pyglm = False, y_dist = 'binomial'
		)
		beta_gl = glasso2.coef_[rev_inds2][0:p].reshape(p)
		corr2 = np.corrcoef(beta_gl, beta)[0, 1]
		self.assertTrue(corr2 > 0.5,
						msg = f'group-lasso fits logistic very poorly (corr = {corr2} btwn real/fitted coeffs)')





if __name__ == '__main__':
	unittest.main()