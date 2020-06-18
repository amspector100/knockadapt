import pytest
import numpy as np
from scipy import stats
import unittest
from .context import knockadapt

from knockadapt import utilities
from knockadapt import graphs
from knockadapt import metro
import time


class TestFdrControl(unittest.TestCase):

	def test_t_generation(self):

		p = 3
		n = 100
		time0 = time.time()
		V = graphs.AR1(p=p, rho=0.5)
		X, Xk, rejection = metro.ar1t_knockoffs(
			n=n, Sigma=V, method='sdp', df_t=3,
		)


	def test_t_generation(self):

		p = 100
		n = 1000
		time0 = time.time()
		V = graphs.AR1(p=p, rho=0.5)
		X, Xk, rejection = metro.ar1t_knockoffs(
			n=n, Sigma=V, method='sdp', df_t=3,
		)
		print(time.time() - time0)
		print(Xk.shape)
		print(X.shape)
		raise ValueError()





if __name__ == '__main__':
	unittest.main()