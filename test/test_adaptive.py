import numpy as np
import unittest
from .context import knockadapt

from knockadapt import utilities, graphs, adaptive

class TestGroupKnockoffEval(unittest.TestCase):
	""" Tests GroupKnockoffEval class """

	def sample_knockoffs(self):
		""" Tests recycling  """

		n = 200
		p = 30
		q = 0.05
		np.random.seed(110)
		X, y, beta, Q, corr_matrix, groups = graphs.daibarber2016_graph(
			n = n, p = p, y_dist = 'binomial'
		)




if __name__ == '__main__':
	unittest.main()