import numpy as np
from statsmodels.stats.moment_helpers import cov2corr
import unittest
from .context import knockadapt

from knockadapt import utilities, graphs, knockoffs


class CheckSMatrix(unittest.TestCase):

	# Helper function
	def check_S_properties(self, V, S, groups):

		# Test PSD-ness of S
		min_S_eig = np.linalg.eigh(S)[0].min()
		self.assertTrue(
			min_S_eig > 0, f'S matrix is not positive semidefinite: mineig is {min_S_eig}' 
		)

		# Test PSD-ness of 2V - S
		min_diff_eig = np.linalg.eigh(2*V - S)[0].min()
		self.assertTrue(
			min_diff_eig > 0, f"2Sigma-S matrix is not positive semidefinite: mineig is {min_diff_eig}"
		)

		# Calculate conditional knockoff matrix
		invV = utilities.chol2inv(V)
		invV_S = np.dot(invV, S)
		Vk = 2 * S - np.dot(S, invV_S)

		# Test PSD-ness of the conditional knockoff matrix
		min_Vk_eig = np.linalg.eigh(Vk)[0].min()
		self.assertTrue(
			min_Vk_eig > 0, f"conditional knockoff matrix is not positive semidefinite: mineig is {min_Vk_eig}"
		)

		# Test that S is just a block matrix
		p = V.shape[0]
		S_test = np.zeros((p, p))
		for j in np.unique(groups):

			# Select subset of S
			inds = np.where(groups == j)[0]
			full_inds = np.ix_(inds, inds)
			group_S = S[full_inds]

			# Fill only in this subset of S
			S_test[full_inds] = group_S


		# return
		np.testing.assert_almost_equal(
			S_test, S, decimal = 5, err_msg = "S matrix is not a block matrix of the correct shape"
		)


class TestEquicorrelated(CheckSMatrix):
	""" Tests equicorrelated knockoffs and related functions """

	def test_eigenvalue_calculation(self):

		# Test to make sure non-group and group versions agree
		# (in the case of no grouping)
		p = 100
		groups = np.arange(0, p, 1) + 1
		for rho in [0, 0.3, 0.5, 0.7]:
			V = np.zeros((p, p)) + rho
			for i in range(p):
				V[i, i] = 1
			expected_gamma = min(1, 2*(1-rho))
			gamma = knockoffs.calc_min_group_eigenvalue(
				Sigma=V, groups=groups, 
			)
			np.testing.assert_almost_equal(
				gamma, expected_gamma, decimal = 3, 
				err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'
			)

		# Test non equicorrelated version
		V = np.random.randn(p, p)
		V = np.dot(V.T, V) + 0.1*np.eye(p)
		V = cov2corr(V)
		expected_gamma = min(1, 2*np.linalg.eigh(V)[0].min())
		gamma = knockoffs.calc_min_group_eigenvalue(
			Sigma=V, groups=groups
		)
		np.testing.assert_almost_equal(
			gamma, expected_gamma, decimal = 3, 
			err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'

		)

	def test_equicorrelated_construction(self):

		# Test S matrix construction
		p = 100
		groups = np.arange(0, p, 1) + 1
		V = np.random.randn(p, p)
		V = np.dot(V.T, V) + 0.1*np.eye(p)
		V = cov2corr(V)

		# Expected construction
		expected_gamma = min(1, 2*np.linalg.eigh(V)[0].min())
		expected_S = expected_gamma*np.eye(p)

		# Equicorrelated
		S = knockoffs.equicorrelated_block_matrix(Sigma=V, groups=groups)

		# Test to make sure the answer is expected
		np.testing.assert_almost_equal(
			S, expected_S, decimal = 3, 
			err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'

		)

		# # Do it again with a block matrix - start by constructing
		# # something we can easily analyze
		# group_sizes = [3, 2, 3, 4, 1]
		# groups = []
		# for i, size in enumerate(group_sizes):
		# 	to_add = [i]*size
		# 	groups += to_add
		# groups = np.array(groups) + 1

	def test_psd(self):

		# Test S matrix construction
		p = 100
		V = np.random.randn(p, p)
		V = np.dot(V.T, V) + 0.1*np.eye(p)
		V = cov2corr(V)

		# Create for various groups
		groups = np.random.randint(1, p, size=(p))
		groups = utilities.preprocess_groups(groups)
		S = knockoffs.equicorrelated_block_matrix(Sigma=V, groups=groups)

		# Check S properties
		self.check_S_properties(V, S, groups)




class TestSDP(CheckSMatrix):
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
		self.check_S_properties(corr_matrix, S_triv, trivial_groups)

		# Repeat for group_gaussian_knockoffs method
		_, S_triv2 = knockoffs.group_gaussian_knockoffs(
			X = X, Sigma = corr_matrix, groups = trivial_groups, 
			return_S = True, sdp_verbose = False, verbose = False,
			method = 'sdp'
		)
		np.testing.assert_array_almost_equal(
			S_triv2, np.eye(p), decimal = 2, 
			err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
		)
		self.check_S_properties(corr_matrix, S_triv2, trivial_groups)

		# Test slightly harder case
		_,_,_,_, expected_out, _ = graphs.daibarber2016_graph(
			n = n, p = p, gamma = 0
		)
		_, S_harder = knockoffs.group_gaussian_knockoffs(
			X = X, Sigma = corr_matrix, groups = groups, 
			return_S = True, sdp_verbose = False, verbose = False,
			method = 'sdp'
		)
		np.testing.assert_almost_equal(
			S_harder, expected_out, decimal = 2,
			err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
		)
		self.check_S_properties(corr_matrix, S_harder, groups)

		# Repeat for ASDP
		_, S_harder_ASDP = knockoffs.group_gaussian_knockoffs(
			X = X, Sigma = corr_matrix, groups = groups, method = 'ASDP',
			return_S = True, sdp_verbose = False, verbose = False
		)
		np.testing.assert_almost_equal(
			S_harder_ASDP, expected_out, decimal = 2,
			err_msg = 'solve_group_ASDP does not produce optimal S matrix (daibarber graphs)'
		)
		self.check_S_properties(corr_matrix, S_harder_ASDP, groups)


	def test_sdp_tolerance(self):

		# Get graph
		np.random.seed(110)
		Q = graphs.ErdosRenyi(p=50, tol=1e-1)
		V = cov2corr(utilities.chol2inv(Q))
		groups = np.concatenate([np.zeros(10) + j for j in range(5)]) + 1
		groups = groups.astype('int32')

		# Solve SDP
		for tol in [1e-3, 0.01, 0.02]:
			S = knockoffs.solve_group_SDP(
				Sigma=V, 
				groups=groups, 
				sdp_verbose=False, 
				objective="pnorm",  
				num_iter=10,
				tol=tol
			)
			G = np.hstack([np.vstack([V, V-S]), np.vstack([V-S, V])])
			mineig = np.linalg.eig(G)[0].min()
			self.assertTrue(
				tol - mineig < 1e3,
				f'sdp solver fails to control minimum eigenvalues: tol is {tol}, val is {mineig}'
			)
			self.check_S_properties(V, S, groups)


	def test_corrmatrix_errors(self):
		""" Tests that SDP raises informative errors when sigma is not scaled properly"""

		# Get graph
		np.random.seed(110)
		Q = graphs.ErdosRenyi(p=50, tol=1e-1)
		V = utilities.chol2inv(Q)
		groups = np.concatenate([np.zeros(10) + j for j in range(5)]) + 1
		groups = groups.astype('int32')


		# Helper function
		def SDP_solver():
			return knockoffs.solve_group_SDP(V, groups)

		# Make sure the value error increases 
		self.assertRaisesRegex(
			ValueError, "Sigma is not a correlation matrix",
			SDP_solver
		)


class TestKnockoffs(unittest.TestCase):
	""" Tests whether knockoffs have correct distribution empirically"""

	def test_method_parser(self):

		# Easiest test
		method1 = 'hello'
		out1 = knockoffs.parse_method(method1, None, None)
		self.assertTrue(
			out1 == method1, 
			"parse method fails to return non-None methods"
		)

		# Default is TFKP
		p = 1000
		groups = np.arange(1, p+1, 1)
		out2 = knockoffs.parse_method(None, groups, p)
		self.assertTrue(
			out2 == 'tfkp', 
			"parse method fails to return tfkp by default"
		)

		# Otherwise SDP
		groups[-1] = 1
		out2 = knockoffs.parse_method(None, groups, p)
		self.assertTrue(
			out2 == 'sdp', 
			"parse method fails to return SDP for grouped knockoffs"
		)

		# Otherwise ASDP
		p = 1001
		groups = np.ones(p)
		out2 = knockoffs.parse_method(None, groups, p)
		self.assertTrue(
			out2 == 'asdp', 
			"parse method fails to return asdp for large p"
		)

	def test_error_raising(self):

		# Generate data
		n = 10
		p = 100
		X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
			n = n, p = p, gamma = 1, rho = 0.8
		)
		S_bad = np.eye(p)

		def fdr_vio_knockoffs():
			knockoffs.group_gaussian_knockoffs(
				X = X, 
				Sigma = corr_matrix,
				S=S_bad,
				sdp_verbose = False, 
				verbose = False
			)

		self.assertRaisesRegex(
			np.linalg.LinAlgError,
			"meaning FDR control violations are extremely likely",
			fdr_vio_knockoffs, 
		)

	def test_ungrouped_knockoffs(self):


		# Test knockoff construction for TFKP and SDP
		# on equicorrelated matrices
		n = 100000
		copies = 3
		p = 10
		for rho in [0.1, 0.5, 0.9]:

			for gamma in [0, 0.5, 1]:

				for method in ['tfkp', 'sdp']:

					X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
						n = n, p = p, gamma = gamma, rho = rho
					)
					# S matrix
					trivial_groups = np.arange(0, p, 1) + 1
					all_knockoffs, S = knockoffs.group_gaussian_knockoffs(
						X = X, 
						Sigma = corr_matrix,
						groups = trivial_groups, 
						copies = copies,
						method = method, 
						return_S = True,
						sdp_verbose = False, 
						verbose = False
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