"""
This code is adapted from the metro package, introduced
by the paper https://arxiv.org/abs/1903.00434. Most of this code
was initially written by Wenshuo Wang and Stephen Bates.
"""

import numpy as np
import itertools
from scipy.stats import t

# # Helper functions for T-MTM
# def p_marginal_trans_log(
# 		j,
# 		xj,
# 		xjm1,
# 		df_t,
# 		rhos,
# 		p
# 	):
# 	"""
# 	:param j: Coordinate
# 	:param xj: Value of variable
# 	:param df_t: Degrees of freedom
# 	:param rhos: Values of rho
# 	:param p: Dimensionality
# 	"""
#     if j>p:
#         raise ValueErorr(
#         	f"Error, j ({j}) > p ({p}) "
#         )
#     if j==1:
#     	# Marginal T prob 
#     	main_log_prob = np.log(t.pdf(xj * np.sqrt(df_t/(df_t/2)), df=df_t))
#         consts = 0.5*np.log(df_t)-0.5*np.log(df_t-2)
#         return main_log_prob + consts
#     else:
#     	j = j - 1 # For indexing ??
#     	joint_input = (xj - rhos[j-1]*xjm1)*np.sqrt(df_t/(df_t-2))/np.sqrt(1-rhos[j-1]**2)
#     	main_log_prob = np.log(t.pdf(joint_input), df=df_t)

#     return(0.5*(np.log(df_t)-np.log(df_t-2)-np.log(1-rhos[j-1]**2)))
