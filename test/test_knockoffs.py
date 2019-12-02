import numpy as np
import unittest
from .context import knockadapt

from knockadapt import utilities, graphs, knockoffs

class TestSDP(unittest.TestCase):
	""" Tests fitting of group lasso """

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




if __name__ == '__main__':
	unittest.main()