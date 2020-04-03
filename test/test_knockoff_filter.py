import numpy as np
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt.knockoffs import solve_group_SDP
from knockadapt.knockoff_filter import mx_knockoff_filter

class TestFdrControl(unittest.TestCase):

	def check_fdr_control(
			self, 
			reps=50,
			q=0.2,
			alpha=0.05,
			filter_kwargs={},
			**kwargs
		):

		np.random.seed(110)

		# Create and name DGP
		_, _, beta, _, Sigma = graphs.sample_data(
			**kwargs
		)
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
			S = solve_group_SDP(Sigma, groups=groups)
			invSigma = utilities.chol2inv(Sigma)
			group_nonnulls = utilities.fetch_group_nonnulls(beta, groups)

			# Container for fdps
			fdps = []

			# Sample data reps times
			for j in range(reps):
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

				# Run MX knockoff filter
				selections = mx_knockoff_filter(
					X=X, 
					y=y, 
					Sigma=Sigma, 
					groups=groups,
					knockoff_kwargs={'S':S, 'invSigma':invSigma, 'verbose':False},
					fdr=q,
					**filter_kwargs,
				)

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

class TestMXKnockoffFilter(TestFdrControl):
	""" Tests MX knockoff filter """

	def test_gnull_control(self):
		""" Test FDR control under global null """

		# Scenario 1: AR1 a = 1, b = 1, global null
		self.check_fdr_control(
			n=100, p=50, method='AR1', sparsity=0, y_dist='gaussian', reps=15
		)

		# Scenario 2: Erdos Renyi
		self.check_fdr_control(
			n=300, p=50, method='ErdosRenyi', sparsity=0, y_dist='gaussian', reps=15,
			filter_kwargs = {'feature_stat_fn':'ols'}
		)

		# Scenario 3: Dai Barber
		self.check_fdr_control(
			method='daibarber2016', rho=0.6, sparsity=0, y_dist='binomial', reps=15
		)

	def test_sparse_control(self):
		""" Test FDR control under global null """

		# Scenario 1: AR1 a = 1, b = 1, global null
		self.check_fdr_control(
			n=300, p=100, method='AR1', sparsity=0.2, y_dist ='binomial', reps=15,		)

		# Scenario 2: Erdos Renyi
		self.check_fdr_control(
			n=100, p=100, method='ErdosRenyi', sparsity=0.2, y_dist='gaussian', reps=15
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
			n=100, p=50, method='ErdosRenyi', sparsity=0.5, y_dist='binomial', reps=15
		)

		# Scenario 3: Dai Barber
		self.check_fdr_control(
			method='daibarber2016', rho=0.4, sparsity=0.5, y_dist='gaussian', reps=15,
			filter_kwargs={'feature_stat_fn':'margcorr'}
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



if __name__ == '__main__':
	unittest.main()