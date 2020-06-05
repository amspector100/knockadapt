import numpy as np
from . import utilities
from . import knockoff_stats as kstats
from .knockoffs import gaussian_knockoffs


class KnockoffFilter:
	"""
	:param fixedX: If True, creates fixed-X knockoffs.
	Defaults to False (model-X knockoffs).
	"""
	def __init__(self, fixedX=False):
		self.debias = False
		self.fixedX = fixedX

	def sample_knockoffs(
		self, X, Sigma, groups, knockoff_kwargs, recycle_up_to,
	):

		# SDP degen flag (for internal use)
		if "_sdp_degen" in knockoff_kwargs:
			_sdp_degen = knockoff_kwargs.pop("_sdp_degen")
		else:
			_sdp_degen = False

		# If fixedX, signal this to knockoff kwargs
		if self.fixedX:
			knockoff_kwargs['fixedX'] = True
			Sigma = None

		# Initial sample
		knockoffs, S = gaussian_knockoffs(
			X=X, groups=groups, Sigma=Sigma, return_S=True, **knockoff_kwargs,
		)
		knockoffs = knockoffs[:, :, 0]

		# Possibly use recycling
		if recycle_up_to is not None:

			# Split
			rec_knockoffs = X[:recycle_up_to]
			new_knockoffs = knockoffs[recycle_up_to:]

			# Combine
			knockoffs = np.concatenate((rec_knockoffs, new_knockoffs), axis=0)

		# For high precision simulations of degenerate knockoffs,
		# ensure degeneracy
		if _sdp_degen:
			sumcols = X[:, 0] + knockoffs[:, 0]
			knockoffs = sumcols.reshape(-1, 1) - X

		self.knockoffs = knockoffs
		self.S = S

		# Possibly invert joint feature-knockoff cov matrix for debiasing lasso
		if self.debias:
			self.G = np.concatenate(
				[
					np.concatenate([self.Sigma, self.Sigma - self.S]),
					np.concatenate([self.Sigma - self.S, self.Sigma]),
				],
				axis=1,
			)
			self.Ginv = utilities.chol2inv(self.G)
		else:
			self.Ginv = None
		return knockoffs

	def make_selections(self, W, fdr):
		"""" Calculate data dependent threshhold and selections """
		T = kstats.data_dependent_threshhold(W=W, fdr=fdr)
		selected_flags = (W >= T).astype("float32")
		return selected_flags

	def forward(
		self,
		X,
		y,
		Sigma=None,
		groups=None,
		knockoffs=None,
		feature_stat="lasso",
		fdr=0.10,
		feature_stat_kwargs={},
		knockoff_kwargs={"sdp_verbose": False},
		recycle_up_to=None,
	):
		"""
		:param X: n x p design matrix
		:param y: p-length response array
		:param Sigma: p x p covariance matrix of X. Alternatively,
		this can be None if fitting fixedX knockoffs.
		:param groups: Grouping of features, p-length
		array of integers from 1 to m (with m <= p).
		:param knockoffs: n x p array of knockoffs.
		If None, will construct second-order group MX knockoffs.
		Defaults to group gaussian knockoff constructor.
		:param feature_stat: Function used to
		calculate W-statistics in knockoffs. 
		Defaults to group lasso coefficient difference.
		:param fdr: Desired fdr.
		:param feature_stat: A classname with a fit method.
		The fit method must takes X, knockoffs, y, and groups,
		and returns a set of p anti-symmetric knockoff 
		statistics. Can also be one of "lasso", "ols", or "margcorr." 
		:param feature_stat_kwargs: Kwargs to pass to 
		the feature statistic.
		:param knockoff_kwargs: Kwargs to pass to the 
		knockoffs constructor.
		:param recycle_up_to: Three options:
			- if None, does nothing.
			- if an integer > 1, uses the first "recycle_up_to"
			rows of X as the the first "recycle_up_to" rows of knockoffs.
			- if a float between 0 and 1 (inclusive), interpreted
			as the proportion of knockoffs to recycle. 
		For more on recycling, see https://arxiv.org/abs/1602.03574
		"""

		# Preliminaries
		self.Sigma = Sigma
		n = X.shape[0]
		p = X.shape[1]
		if groups is None:
			groups = np.arange(1, p + 1, 1)

		# Parse recycle_up_to
		if recycle_up_to is None:
			pass
		elif recycle_up_to <= 1:
			recycle_up_to = int(recycle_up_to * n)
		else:
			recycle_up_to = int(recycle_up_to)
 
		# Parse feature statistic function
		if feature_stat == "lasso":
			feature_stat = kstats.LassoStatistic()
			if 'debias' in feature_stat_kwargs:
				if feature_stat_kwargs['debias']:
					self.debias = True
		elif feature_stat == 'dlasso':
			feature_stat = kstats.LassoStatistic()
			self.debias = True
		elif feature_stat == 'ridge':
			feature_stat = kstats.RidgeStatistic()
		elif feature_stat == "ols":
			feature_stat = kstats.OLSStatistic()
		elif feature_stat == "margcorr":
			feature_stat = kstats.MargCorrStatistic()
		elif feature_stat == 'randomforest':
			feature_stat = kstats.RandomForestStatistic()
		elif feature_stat == 'deeppink':
			feature_stat = kstats.DeepPinkStatistic()

		# Sample knockoffs
		if knockoffs is None:
			knockoffs = self.sample_knockoffs(
				X=X,
				Sigma=Sigma,
				groups=groups,
				knockoff_kwargs=knockoff_kwargs,
				recycle_up_to=recycle_up_to,
			)
			if self.debias:
				# This is only computed if self.debias is True
				feature_stat_kwargs["Ginv"] = self.Ginv
				feature_stat_kwargs['debias'] = True

		# Feature statistics
		feature_stat.fit(
			X=X, knockoffs=knockoffs, y=y, groups=groups, **feature_stat_kwargs
		)
		# Inherit some attributes
		self.fstat = feature_stat
		self.Z = self.fstat.Z
		self.W = self.fstat.W
		self.score = self.fstat.score
		self.score_type = self.fstat.score_type

		self.selected_flags = self.make_selections(self.W, fdr)

		# Return
		return self.selected_flags