import numpy as np
import unittest
from .context import knockadapt

from knockadapt import utilities, graphs, knockoffs

class TestSDP(unittest.TestCase):
	""" Tests an easy case of SDP and ASDP """

	def test_easy_sdp(self):


		# Test non-group SDP first
		n = 200
		p = 50
		X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
			n = n, p = p, gamma = 0.3
		)

		# S matrix
		trivial_groups = np.arange(0, p, 1) + 1
		S_triv = knockoffs.solve_group_SDP(corr_matrix, trivial_groups)
		np.testing.assert_array_almost_equal(
			S_triv, np.eye(p), decimal = 2,
			err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
		)

		# Repeat for group_gaussian_knockoffs method
		_, S_triv2 = knockoffs.group_gaussian_knockoffs(
			X = X, Sigma = corr_matrix, groups = trivial_groups, 
			return_S = True, sdp_verbose = False, verbose = False
		)
		np.testing.assert_array_almost_equal(
			S_triv2, np.eye(p), decimal = 2, 
			err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
		)

		# Test slightly harder case
		_,_,_,_, expected_out, _ = graphs.daibarber2016_graph(
			n = n, p = p, gamma = 0
		)
		_, S_harder = knockoffs.group_gaussian_knockoffs(
			X = X, Sigma = corr_matrix, groups = groups, 
			return_S = True, sdp_verbose = False, verbose = False
		)
		np.testing.assert_almost_equal(
			S_harder, expected_out, decimal = 2,
			err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
		)

		# Repeat for ASDP
		_, S_harder_ASDP = knockoffs.group_gaussian_knockoffs(
			X = X, Sigma = corr_matrix, groups = groups, method = 'ASDP',
			return_S = True, sdp_verbose = False, verbose = False
		)
		np.testing.assert_almost_equal(
			S_harder_ASDP, expected_out, decimal = 2,
			err_msg = 'solve_group_ASDP does not produce optimal S matrix (daibarber graphs)'
		)


class testKnockoffs(unittest.TestCase):
	""" Tests whether knockoffs have correct distribution empirically"""

	def test_ungrouped_knockoffs(self):


		# Test knockoff construction
		n = 100000
		copies = 3
		p = 10
		for rho in [0.1, 0.5, 0.9]:

			for gamma in [0, 0.5, 1]:

				X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
					n = n, p = p, gamma = gamma, rho = rho
				)
				# S matrix
				trivial_groups = np.arange(0, p, 1) + 1
				all_knockoffs, S = knockoffs.group_gaussian_knockoffs(
					X = X, Sigma = corr_matrix, groups = trivial_groups, 
					copies = copies, return_S = True,
					sdp_verbose = False, verbose = False
				)

				# Calculate empirical covariance matrix
				knockoff_copy = all_knockoffs[:, :, 0]
				features = np.concatenate([X, knockoff_copy], axis = 1)
				G = np.corrcoef(features, rowvar = False)
				
				# Now we need to show G has the correct structure - three tests
				np.testing.assert_array_almost_equal(
					G[:p, :p], corr_matrix, decimal = 2,
					err_msg = f'''Empirical corr matrix btwn X and knockoffs
					has incorrect values (specifically, cov(X, X) section)
					Daibarber graph, rho = {rho}, gamma = {gamma}
					'''
				)

				# Cov(X, knockoffs)
				np.testing.assert_array_almost_equal(
					corr_matrix - G[p:, :p], S, decimal = 2,
					err_msg = f'''Empirical corr matrix btwn X and knockoffs
					has incorrect values (specifically, cov(X, knockoffs) section)
					Daibarber graph, rho = {rho}, gamma = {gamma}
					'''
				)
				# Cov(knockoffs, knockoffs)
				np.testing.assert_array_almost_equal(
					G[p:, p:], corr_matrix, decimal = 2,
					err_msg = f'''Empirical corr matrix btwn X and knockoffs
					has incorrect values (specifically, cov(knockoffs, knockoffs) section)
					Daibarber graph, rho = {rho}, gamma = {gamma}
					'''
				)


if __name__ == '__main__':
	unittest.main()