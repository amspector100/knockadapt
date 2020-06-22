import pytest
import numpy as np
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt import metro, metro_generic
import time

class TestGenericMetro(unittest.TestCase):

	def test_proposal_covs(self):

		# Fake data + loss function
		np.random.seed(110)
		n = 5
		p = 100
		X,_,_,Q,V = graphs.sample_data(method='AR1', n=n, p=p)
		
		# Metro sampler, proposal params
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
			lf=lambda x: np.log(x).sum(),
			X=X,
			V=V,
			order=np.arange(0, p, 1),
			active_frontier=[[] for _ in range(p)],
		)
		metro_sampler.create_proposal_params(sdp_verbose=True)

		# Test that invSigma is as it should be
		G = metro_sampler.G
		for j in [1, 50, 99]:
			Gjinv = np.linalg.inv(G[0:p+j, 0:p+j])
			np.testing.assert_almost_equal(
				Gjinv, metro_sampler.invSigmas[j], decimal=3,
				err_msg=f'Metro sampler fails to correctly calculate {j}th invSigma'
			)





# class TestFdrControl(unittest.TestCase):

# 	def test_t_generation(self):

# 		p = 3
# 		n = 100
# 		time0 = time.time()
# 		V = graphs.AR1(p=p, rho=0.5)
# 		X, Xk, rejection = metro.ar1t_knockoffs(
# 			n=n, Sigma=V, method='sdp', df_t=3,
# 		)


# 	def test_t_generation(self):

# 		p = 3
# 		n = 1000
# 		time0 = time.time()
# 		V = graphs.AR1(p=p, rho=0.5)
# 		X, Xk, rejection = metro.ar1t_knockoffs(
# 			n=n, Sigma=V, method='sdp', df_t=3,
# 		)
# 		print(time.time() - time0)
# 		print(Xk.shape)
# 		print(X.shape)
# 		raise ValueError()





if __name__ == '__main__':
	unittest.main()