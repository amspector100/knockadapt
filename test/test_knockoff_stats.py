import numpy as np
import unittest
from .context import knockadapt

from knockadapt import knockoff_stats as kstats
from knockadapt import utilities, graphs
from knockadapt.knockoff_stats import data_dependent_threshhold

class TestGroupLasso(unittest.TestCase):
	""" Tests fitting of group lasso """

	def test_LCD(self):

		# Fake data
		Z = np.array([-1, -2, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0])
		groups = np.array([1, 1, 1, 2, 2, 2])
		W = kstats.combine_Z_stats(Z, groups)
		np.testing.assert_array_almost_equal(
			W, np.array([4, 0]), decimal = 3,
			err_msg = 'calc_LCD function incorrectly calculates group LCD'
		)

		# Again
		Z2 = np.array([0, 1, 2, 3, -1, -2, -3, -4])
		groups2 = np.array([1, 2, 3, 4])
		W2 = kstats.combine_Z_stats(Z2, groups2, group_agg = 'avg')
		np.testing.assert_array_almost_equal(
			W2, np.array([-1, -1, -1, -1]),
			err_msg = 'calc_LCD function incorrectly calculates group LCD'
		)

	def test_margcorr_statistic(self):

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
		margcorr = kstats.MargCorrStatistic()
		W = margcorr.fit(X, knockoffs, y, groups = None)

		self.assertTrue(
			np.abs(W[0] - 1/np.sqrt(2)) < 0.05, 
			msg = 'marg_corr_diff statistic calculates correlations incorrectly'
		)

	def test_lars_solver_fit(self):
		""" Tests power of lars lasso solver """

		# Get DGP, knockoffs, S matrix
		np.random.seed(1)
		p = 100
		n = 150
		rho = 0.7
		X, y, beta, Q, V = graphs.sample_data(
			method='daibarber2016',
			rho=rho,
			gamma=1,
			sparsity=0.5,
			coeff_size=5,
			n=n,
			p=p,
			sign_prob=0,
			coeff_dist="uniform"
		)
		S = (1-rho)*np.eye(p)
		groups = np.arange(1, p+1, 1)
		knockoffs = knockadapt.knockoffs.group_gaussian_knockoffs(
			X=X,
			Sigma=V,
			S=S,
			groups=groups,
		)[:,:,0]
		# Test lars solver
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X,
			knockoffs,
			y,
			use_lars=True,
		)
		W = lasso_stat.W
		Z = lasso_stat.Z
		T = data_dependent_threshhold(W, fdr = 0.2)
		selections = (W >= T).astype('float32')
		power = ((beta != 0)*selections).sum()/np.sum(beta != 0)
		fdp = ((beta == 0)*selections).sum()/max(np.sum(selections), 1)
		self.assertTrue(
			power==1.0,
			msg = f"Power {power} for LARS solver in equicor case should be 1"
		)



	def test_lars_path_fit(self):
		""" Tests power of lars path statistic """
		# Get DGP, knockoffs, S matrix
		np.random.seed(110)
		p = 100
		n = 300
		rho = 0.7
		X, y, beta, Q, V = graphs.sample_data(
			method='daibarber2016',
			rho=rho,
			gamma=1,
			sparsity=0.5,
			coeff_size=5,
			n=n,
			p=p,
			sign_prob=0.5,
		)
		S = (1-rho)*np.eye(p)
		groups = np.arange(1, p+1, 1)
		knockoffs = knockadapt.knockoffs.group_gaussian_knockoffs(
			X=X,
			Sigma=V,
			S=S,
			groups=groups,
		)[:,:,0]
		# Repeat for LARS path statistic
		# Test lars solver
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X,
			knockoffs,
			y,
			zstat='lars_path',
			pair_agg='sm',
		)
		W = lasso_stat.W
		Z = lasso_stat.Z
		T = data_dependent_threshhold(W, fdr = 0.2)
		selections = (W >= T).astype('float32')
		power = ((beta != 0)*selections).sum()/np.sum(beta != 0)
		self.assertTrue(
			power>0.9,
			msg = f"Power {power} for LARS path statistic in equicor case should be > 0.9"
		)

	def test_ols_fit(self):

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
		lmodel = kstats.OLSStatistic()
		W = lmodel.fit(X, knockoffs, y, groups = groups)

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
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X,
			fake_knockoffs,
			y,
			groups=groups,
			use_pyglm=True,
			y_dist=None,
			max_iter=20,
			tol=5e-2,
			learning_rate=3,
			group_lasso=True
		)
		glasso1 = lasso_stat.model
		rev_inds1 = lasso_stat.rev_inds
		beta_pyglm = glasso1.beta_[rev_inds1][0:p]
		corr1 = np.corrcoef(beta_pyglm, beta)[0, 1]
		self.assertTrue(corr1 > 0.5,
						msg = f'Pyglm fits gauissan very poorly (corr = {corr1} btwn real/fitted coeffs)'
		)
		score_type = lasso_stat.score_type
		self.assertTrue(
			score_type=='mse',
			msg = f'Pyglm group_lasso has incorrect score type ({score_type}), expected mse'
		)

		# Test again, fitting regular lasso
		lasso_stat2 = kstats.LassoStatistic()
		lasso_stat2.fit(
			X,
			fake_knockoffs,
			y,
			groups=groups,
			y_dist='gaussian',
			max_iter=50,
			group_lasso=False,
		)
		beta2 = lasso_stat2.model.coef_[lasso_stat2.rev_inds][0:p]
		corr2 = np.corrcoef(beta2, beta)[0, 1]
		self.assertTrue(corr2 > 0.5,
						msg = f'SKlearn lasso fits gaussian very poorly (corr = {corr2} btwn real/fitted coeffs)'
		)
		score_type = lasso_stat2.score_type
		self.assertTrue(
			score_type=='mse_cv',
			msg = f'Sklearn lasso has incorrect score type ({score_type}), expected mse_cv'
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
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X, fake_knockoffs, y, 
			groups=groups,
			use_pyglm=True,
			y_dist=None,
			max_iter=20,
			tol=5e-2,
			learning_rate=3,
			group_lasso=True,
		)
		glasso1 = lasso_stat.model
		rev_inds1 = lasso_stat.rev_inds
		beta_pyglm = glasso1.beta_[rev_inds1][0:p]
		corr1 = np.corrcoef(beta_pyglm, beta)[0, 1]
		self.assertTrue(corr1 > 0.5,
						msg = f'Pyglm fits logistic very poorly (corr = {corr1} btwn real/fitted coeffs)')


		# Get best model for group-lasso
		lasso_stat2 = kstats.LassoStatistic()
		lasso_stat2.fit(
			X,
			fake_knockoffs,
			y,
			groups=groups,
			use_pyglm=False,
			max_iter=20,
			tol=5e-2,
			learning_rate=3,
			group_lasso=True
		)
		glasso2 = lasso_stat2.model
		rev_inds2 = lasso_stat2.rev_inds
		beta_gl = glasso2.coef_[rev_inds2][0:p].reshape(p)
		corr2 = np.corrcoef(beta_gl, beta)[0, 1]
		self.assertTrue(corr2 > 0.5,
						msg = f'group-lasso fits logistic very poorly (corr = {corr2} btwn real/fitted coeffs)')


		# Test again, fitting logistic (regular) lasso
		lasso_stat3 = kstats.LassoStatistic()
		lasso_stat3.fit(
			X=X,
			knockoffs=fake_knockoffs,
			y=y,
			y_dist=None,
			max_iter=50,
		)
		glasso3 = lasso_stat3.model
		rev_inds3 = lasso_stat3.rev_inds
		beta3 = glasso3.coef_[0, rev_inds3][0:p]
		corr3 = np.corrcoef(beta3, beta)[0, 1]
		self.assertTrue(corr3 > 0.5,
						msg = f'SKlearn lasso fits logistic very poorly (corr = {corr3} btwn real/fitted coeffs)'
		)
		score_type = lasso_stat3.score_type
		self.assertTrue(
			score_type=='accuracy_cv',
			msg = f'Sklearn logistic lasso has incorrect score type ({score_type}), expected accuracy_cv'
		)


	def test_antisymmetric_fns(self):

		n = 100
		p = 20
		np.random.seed(110)
		X, y, beta, _, corr_matrix = graphs.sample_data(
			n = n, p = p, y_dist = 'gaussian', 
			coeff_size = 100, sign_prob = 1
		)		
		groups = np.arange(1, p+1, 1)

		# These are not real, just helpful syntatically
		fake_knockoffs = np.zeros((n, p))

		# Run to make sure there are no errors for
		# different pair_aggs
		np.random.seed(110)
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X = X, 
			knockoffs = fake_knockoffs,
			y = y, 
			y_dist = None,
			pair_agg='cd'
		)
		W_cd = lasso_stat.W
		Z_cd = lasso_stat.Z
		W_cd[np.abs(W_cd) < 10] = 0
		Z_cd[np.abs(Z_cd) < 10] = 0
		np.testing.assert_array_almost_equal(
			W_cd, -1*Z_cd[0:p], 
			err_msg = 'pair agg CD returns weird W stats'
		)


		# Run to make sure there are no errors for
		# different pair_aggs
		np.random.seed(110)
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X = X, 
			knockoffs = fake_knockoffs,
			y = y, 
			y_dist = None,
			pair_agg='sm'
		)
		Z_sm = lasso_stat.Z
		W_sm = lasso_stat.W
		np.testing.assert_array_almost_equal(
			W_sm, np.abs(Z_sm[0:p]), decimal=3,
			err_msg = 'pair agg SM returns weird W stats'
		)

		# Run to make sure there are no errors for
		# different pair_aggs
		np.random.seed(110)
		lasso_stat = kstats.LassoStatistic()
		lasso_stat.fit(
			X = X, 
			knockoffs = fake_knockoffs,
			y = y, 
			y_dist = None,
			pair_agg='scd'
		)
		W_scd = lasso_stat.W
		Z_scd = lasso_stat.Z
		W_scd[np.abs(W_scd) < 10] = 0
		Z_scd[np.abs(Z_scd) < 10] = 0
		np.testing.assert_array_almost_equal(
			W_scd, Z_scd[0:p], 
			err_msg = 'pair agg SCD returns weird W stats'
		)

class TestDataThreshhold(unittest.TestCase):
	""" Tests data-dependent threshhold """

	def test_unbatched(self):

		W1 = np.array([1, -2, 3, 6, 3, -2, 1, 2, 5, 3, 0.5, 1, 1, 1, 1, 1, 1, 1])
		T1 = data_dependent_threshhold(W1, fdr = 0.2)
		expected = np.abs(W1).min()
		self.assertTrue(T1==expected, msg=f'Incorrect data dependent threshhold: T1 should be 0, not {T1}')

		W2 = np.array([-1, -2, -3])
		T2 = data_dependent_threshhold(W2, fdr = 0.3)
		self.assertTrue(T2==np.inf, msg=f'Incorrect data dependent threshhold: T2 should be inf, not {T2}')

		W3 = np.array([-5, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
		T3 = data_dependent_threshhold(W3, fdr = 0.2)
		self.assertTrue(T3 == 5, msg=f'Incorrect data dependent threshhold: T3 should be 5, not {T3}')

	def test_batched(self):

		W1 = np.array([1]*10)
		W2 = np.array([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
		W3 = np.array([-1]*10)
		combined = np.stack([W1, W2, W3]).transpose()
		Ts = data_dependent_threshhold(combined, fdr = 0.2)
		expected = np.array([1, 2, np.inf])
		np.testing.assert_array_almost_equal(
			Ts, expected, 
			err_msg = f"Incorrect data dependent threshhold (batched): Ts should be {expected}, not {Ts}"
		)

	def test_zero_handling(self):
		""" Makes sure Ts != 0 """

		W1 = np.array([1]*10 + [0]*10)
		W2 = np.array([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8] + [0]*10)
		W3 = np.array([-1]*10 + [0]*10)
		combined = np.stack([W1, W2, W3]).transpose()
		Ts = data_dependent_threshhold(combined, fdr = 0.2)
		expected = np.array([1, 2, np.inf])
		np.testing.assert_array_almost_equal(
			Ts, expected, 
			err_msg = f"Incorrect data dependent threshhold (batched): Ts should be {expected}, not {Ts}"
		)


if __name__ == '__main__':
	unittest.main()