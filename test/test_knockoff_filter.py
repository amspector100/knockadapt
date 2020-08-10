import pytest
import numpy as np
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt.knockoff_filter import KnockoffFilter

class TestFdrControl(unittest.TestCase):

	def check_fdr_control(
			self, 
			reps=50,
			q=0.2,
			alpha=0.05,
			filter_kwargs={},
			S=None,
			fixedX=False,
			infer_sigma=False,
			**kwargs
		):

		np.random.seed(110)

		# Create and name DGP
		X0, _, beta, _, Sigma = graphs.sample_data(
			**kwargs
		)
		# Now pop corr_matrix
		if 'corr_matrix' in kwargs:
			kwargs.pop('corr_matrix')

		basename = ''
		for key in kwargs:
			basename += f'{key}={kwargs[key]} '

		# Two settings: one grouped, one not
		p = Sigma.shape[0]
		groups1 = np.arange(1, p+1, 1)
		name1 = basename + ' (ungrouped)'
		groups2 = np.random.randint(1, p+1, size=(p,))
		groups2 = utilities.preprocess_groups(groups2)
		name2 = basename + ' (grouped)'

		for name, groups in zip([name1, name2], [groups1, groups2]):
				
			# Solve SDP
			if S is None and not fixedX and not infer_sigma:
				_, S = knockadapt.knockoffs.gaussian_knockoffs(
					X=X0,
					Sigma=Sigma,
					groups=groups,
					method='sdp',
					return_S=True,
				)
			if not fixedX:
				invSigma = utilities.chol2inv(Sigma)
			group_nonnulls = utilities.fetch_group_nonnulls(beta, groups)

			# Container for fdps
			fdps = []

			# Sample data reps times
			for j in range(reps):
				np.random.seed(j)
				X, y, _, _, _ = graphs.sample_data(
					corr_matrix=Sigma,
					beta=beta,
					**kwargs
				)

				# Infer y_dist
				if 'y_dist' in kwargs:
					y_dist = kwargs['y_dist']
				else:
					y_dist = 'gaussian'

				# Run (MX) knockoff filter 
				if fixedX or infer_sigma:
					mu_arg = None
					Sigma_arg = None
					invSigma_arg = None
				else:
					mu_arg = np.zeros(p)
					Sigma_arg = Sigma
					invSigma_arg = invSigma
				knockoff_filter = KnockoffFilter(fixedX=fixedX)

				# Knockoff kwargs
				knockoff_kwargs={
					'S':S, 
					'invSigma':invSigma_arg,
					'verbose':False,
				}
				if 'df_t' in kwargs:
					knockoff_kwargs['df_t'] = kwargs['df_t']

				selections = knockoff_filter.forward(
					X=X, 
					y=y, 
					mu=mu_arg,
					Sigma=Sigma_arg, 
					groups=groups,
					knockoff_kwargs=knockoff_kwargs,
					fdr=q,
					**filter_kwargs,
				)
				del knockoff_filter

				# Calculate fdp
				fdp = np.sum(selections*(1-group_nonnulls))/max(1, np.sum(selections))
				fdps.append(fdp)

			fdps = np.array(fdps)
			fdr = fdps.mean()
			fdr_se = fdps.std()/np.sqrt(reps)

			norm_quant = stats.norm.ppf(1-alpha)

			self.assertTrue(
				fdr - norm_quant*fdr_se <= q,
				msg = f'MX filter FDR is {fdr} with SE {fdr_se} with q = {q} for DGP {name}'
			)

class TestKnockoffFilter(TestFdrControl):
	""" Tests knockoff filter (mostly MX, some FX tests) """

	@pytest.mark.quick
	def test_quality_metrics(self):

		# Fake gaussian data
		np.random.seed(110)
		n = 500000
		p = 5
		rho = 0.5
		S = 0.9*np.eye(p)
		X, y, _, _, V = graphs.sample_data(
			method='daibarber2016', rho=rho, gamma=1, n=n, p=p,
		)

		# Quality metrics
		mxfilter = KnockoffFilter()
		mxfilter.Sigma = V
		mxfilter.forward(
			X=X,
			y=y,
			mu=np.zeros(p),
			Sigma=V,
			groups=None,
			knockoff_kwargs={'S':S},
		)
		corrs, ecvs = mxfilter.compute_quality_metrics()

		# Since these are gaussian knockoffs, they should
		# align with S
		true_mac = np.abs(1-np.diag(S)).mean()
		est_mac = np.abs(corrs).mean()
		self.assertTrue(
			np.abs(true_mac - est_mac) < 0.001,
			msg = f"Knockoff filter incorrectly computes gaussian MAC ({est_mac} vs {true_mac})" 
		)

		# Compute LMCV again with different S
		Ginv = np.linalg.inv(np.concatenate([
			np.concatenate([V, V-S]),
			np.concatenate([V-S, V])
		], axis=1))
		true_ECVs = 1 / np.diag(Ginv[0:p][:, 0:p])
		np.testing.assert_almost_equal(
			ecvs, true_ECVs, decimal=6,
			err_msg = f"Knockoff filter incorrectly computes gaussian ECVS"
		)

		# Fake non-gaussian data


	def test_gnull_control(self):
		""" Test FDR control under global null """

		# Scenario 1: AR1 a = 1, b = 1, global null
		self.check_fdr_control(
			n=100, p=50, method='AR1', sparsity=0, y_dist='gaussian', reps=15
		)

		# Scenario 2: Erdos Renyi
		self.check_fdr_control(
			n=300, p=50, method='ver', sparsity=0, y_dist='gaussian', reps=15,
			filter_kwargs = {'feature_stat':'ols'}
		)

		# Erdos Renyi, but with Ridge Statistic
		self.check_fdr_control(
			n=100, p=50, method='ver', sparsity=0, y_dist='gaussian', reps=15,
			filter_kwargs = {'feature_stat':'ridge'}
		)

		# Scenario 3: Dai Barber
		self.check_fdr_control(
			method='daibarber2016', rho=0.6, sparsity=0, y_dist='binomial', reps=15
		)

	def test_sparse_control(self):
		""" Test FDR control under global null """

		# Scenario 1: AR1 a = 1, b = 1, 
		self.check_fdr_control(
			n=300, p=100, method='AR1', sparsity=0.2, y_dist ='binomial', reps=15,		)

		# Scenario 2: Erdos Renyi
		self.check_fdr_control(
			n=100, p=100, method='ver', sparsity=0.2, y_dist='gaussian', reps=15,
			filter_kwargs={'feature_stat_kwargs':{'debias':True}}
		)

		# Scenario 3: Dai Barber
		self.check_fdr_control(
			method='daibarber2016', rho=0.8, sparsity=0.2, y_dist='binomial', reps=15
		)

	def test_dense_control(self):
		""" Test FDR control under global null """

		# Scenario 1: AR1 a = 1, b = 1, global null
		self.check_fdr_control(
			n=300, p=50, method='AR1', sparsity=0.5, y_dist='gaussian', reps=15,
		)

		# Scenario 2: Erdos Renyi
		self.check_fdr_control(
			n=100, p=50, method='ver', sparsity=0.5, y_dist='binomial', reps=15
		)

		# Scenario 3: Dai Barber
		self.check_fdr_control(
			method='daibarber2016', rho=0.4, sparsity=0.5, y_dist='gaussian', reps=15,
			filter_kwargs={'feature_stat':'margcorr'}
		)

	def test_nonlinear_control(self):
		""" Test FDR control for nonlinear responses """

		# Scenario 1: AR1 a = 1, b = 1, global null
		self.check_fdr_control(
			n=300,
			p=50,
			method='AR1',
			sparsity=0.5,
			y_dist='gaussian', 
			cond_mean='pairint',
			reps=15,
			filter_kwargs = {'feature_stat':'randomforest'}
		)

		# Scenario 2: Erdos Renyi
		self.check_fdr_control(
			n=100,
			p=50,
			method='ver',
			sparsity=0.5,
			y_dist='binomial',
			cond_mean='pairint',
			reps=15,
		)

	def test_recycling_control(self):

		# Scenario 1: AR1, recycle half
		self.check_fdr_control(
			reps=15, n=300, p=50, method='AR1', sparsity=0.5, y_dist='gaussian', 
			filter_kwargs={'recycle_up_to':0.5},
		)

		# Scenario 2: AR1, recycle exactly 23
		self.check_fdr_control(
			reps=15, n=300, p=50, method='AR1', sparsity=0.5, y_dist='gaussian', 
			filter_kwargs={'recycle_up_to':28},
		)

	@pytest.mark.slow
	def test_inferred_mx_control(self):
		self.check_fdr_control(
			reps=15,
			n=200,
			p=100,
			method='AR1',
			sparsity=0,
			y_dist='gaussian', 
			infer_sigma=True
		)

		self.check_fdr_control(
			reps=15,
			n=200, 
			p=150,
			method='ver',
			sparsity=0,
			y_dist='gaussian', 
			infer_sigma=True, 
			filter_kwargs = {'shrinkage':'graphicallasso'}
		)


	@pytest.mark.slow
	def test_fxknockoff_control(self):

		# Scenario 1: AR1, recycle, lasso, p = 50
		self.check_fdr_control(
			fixedX=True, reps=15, n=500, p=50, method='AR1', sparsity=0.5, y_dist='gaussian', 
		)

	@pytest.mark.slow
	def test_deeppink_control(self):
		self.check_fdr_control(
			reps=15, n=5000, p=150, method='AR1', sparsity=0.5, y_dist='gaussian', 
			filter_kwargs={'feature_stat':'deeppink'},
		)

	def test_t_control(self):
		""" FDR control with t-distributed designs """

		# Scenario 1: AR1 a = 1, b = 1, low sparsity
		self.check_fdr_control(
			n=500, p=50, method='AR1', sparsity=0.5, x_dist='ar1t',
			reps=15,
			df_t=5, 
			filter_kwargs={
				'knockoff_type':'artk',
			},
		)
		# Scenario 2: block-T R1 a = 1, b = 1, high sparsity
		self.check_fdr_control(
			n=500, p=50, method='daibarber2016', gamma=0,
			sparsity=0.5, x_dist='blockt', reps=15, df_t=5, 
			filter_kwargs={
				'knockoff_type':'blockt', 
			},
		)

	@pytest.mark.slow
	def test_ising_control(self):

		# Need to pull in specially-estimated Sigma
		from .context import file_directory
		p = 49
		V = np.loadtxt(f'{file_directory}/test_covs/vout{p}.txt')
		self.check_fdr_control(
			n=500, p=49, method='ising', corr_matrix=V,
			sparsity=0.5, x_dist='gibbs', reps=15,
			filter_kwargs={
				'knockoff_type':'ising', 
			},
		)

	@pytest.mark.slow
	def test_ising_dlasso(self):
		""" Makes sure Ising works in combination with debiased lasso """

		# Need to pull in specially-estimated Sigma
		from .context import file_directory
		p = 49
		V = np.loadtxt(f'{file_directory}/test_covs/vout{p}.txt')
		self.check_fdr_control(
			n=500, p=49, method='ising', corr_matrix=V,
			sparsity=0.5, x_dist='gibbs', reps=2, q=1,
			filter_kwargs={
				'knockoff_type':'ising', 'feature_stat':'dlasso',
			},
		)

	@pytest.mark.slow
	def test_lars_control(self):

		# Scenario 1: daibarber2016
		p = 500
		rho = 0.3
		S = (1-rho)*np.eye(p)
		self.check_fdr_control(
			reps=10, 
			n=1000,
			p=p,
			S=S,
			method='daibarber2016',
			gamma=1,
			rho=rho,
			sparsity=0.5, 
			y_dist='gaussian', 
			coeff_dist='uniform', 
			coeff_size=5, 
			filter_kwargs={
				'feature_stat_kwargs':{'use_lars':True}
			},
		)

	@pytest.mark.quick
	def test_selection_procedure(self):

		mxfilter = KnockoffFilter()
		W1 = np.concatenate([np.ones(10), -0.4*np.ones(100)])
		selections = mxfilter.make_selections(W1, fdr=0.1)
		num_selections = np.sum(selections)
		expected = np.sum(W1 > 0)
		self.assertTrue(
			 num_selections == expected,
			f"selection procedure makes {num_selections} discoveries, expected {expected}"
		)

		# Repeat to test zero handling
		W2 = np.concatenate([np.abs(np.random.randn(500)), np.zeros(1)])
		selections2 = mxfilter.make_selections(W2, fdr=0.2)
		num_selections2 = np.sum(selections2)
		expected2 = np.sum(W2 > 0)
		self.assertTrue(
			 num_selections2 == expected2,
			f"selection procedure makes {num_selections2} discoveries, expected {expected2}"
		)

	@pytest.mark.quick
	def test_sdp_degen(self):

		mxfilter = KnockoffFilter()
		p=50
		n=100
		rho=0.8
		S = min(1, 2-2*rho)*np.eye(p)
		X, y, _, _, Sigma = graphs.sample_data(
			rho=rho,
			gamma=1,
			method='daibarber2016',
			p=p,
			n=n,
		)

		# Baseline, no degeneracy
		mxfilter.forward(
			X=X, 
			y=y, 
			Sigma=Sigma, 
			knockoff_kwargs={'S':S, 'verbose':False},
		)
		colsum = X + mxfilter.knockoffs
		colsum_nunique = np.unique(colsum).shape[0]
		self.assertTrue(
			colsum_nunique == n*p,
			f'Expected {n*p} unique values for _sdp_degen False, got {colsum_nunique}'
		)

		# Try again with degeneracy 
		mxfilter = KnockoffFilter()
		mxfilter.forward(
			X=X, 
			y=y, 
			Sigma=Sigma, 
			knockoff_kwargs={'S':S, 'verbose':False, '_sdp_degen':True},
		)
		colsum = np.around(X + mxfilter.knockoffs, 12) # rounding to prevent floating-pt errors
		colsum_nunique = np.unique(colsum).shape[0]
		self.assertTrue(
			colsum_nunique == n,
			f'Expected {n} unique values for _sdp_degen True, got {colsum_nunique}'
		)


if __name__ == '__main__':
	unittest.main()