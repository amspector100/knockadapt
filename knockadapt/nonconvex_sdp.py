import warnings
import numpy as np
import scipy as sp
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utilities
from .utilities import calc_group_sizes, preprocess_groups

# Multiprocessing tools
import itertools
from functools import partial
from multiprocessing import Pool


def blockdiag_to_blocks(M, groups):
    """
    Given a matrix M, pulls out the diagonal blocks as specified by groups.
    :param M: p x p numpy array
    :param groups: p length numpy array 
    """
    blocks = []
    for j in np.sort(np.unique(groups)):
        inds = np.where(groups == j)[0]
        full_inds = np.ix_(inds, inds)
        blocks.append(M[full_inds].copy())
    return blocks


def fk_precision_trace(Sigma, S, invSigma=None):
    """ Computes inverse of trace of feature-knockoff
    precision matrix using numpy (no backprop) """

    # Inverse of cov matrix
    if invSigma is None:
        invSigma = utilities.chol2inv(Sigma)

    # Construct schurr complemenent of G
    diff = Sigma - S
    G_schurr = Sigma - np.dot(np.dot(diff, invSigma), diff)

    # Inverse of eigenvalues
    trace_invG = (1 / np.linalg.eigh(G_schurr)[0]).sum()
    return trace_invG


def block_diag_sparse(*arrs):
    """ Given a list of 2D torch tensors, creates a sparse block-diagonal matrix
    See https://github.com/pytorch/pytorch/issues/31942
    """
    bad_args = []
    for k, arr in enumerate(arrs):
        if isinstance(arr, nn.Parameter):
            arr = arr.data
        if not (isinstance(arr, torch.Tensor) and arr.ndim == 2):
            bad_args.append(k)

    if len(bad_args) != 0:
        raise ValueError(f"Args in {bad_args} positions must be 2D tensor")

    shapes = torch.tensor([a.shape for a in arrs])
    out = torch.zeros(
        torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device
    )
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r : r + rr, c : c + cc] = arrs[i]
        r += rr
        c += cc
    return out


class FKPrecisionTraceLoss(nn.Module):
    """
    A pytorch class to compute S-matrices for 
    (gaussian) MX knockoffs which minimizes the 
    trace of the feature-knockoff precision matrix
    (the inverse of the feature-knockoff 
    covariance/Grahm matrix, G).

    :param Sigma: p x p numpy matrix. Must already
    be sorted by groups.
    :param groups: p length numpy array of groups. 
    These must already be sorted and correspond to 
    the Sigma.
    :param init_S: The initialization values for the S
    block-diagonal matrix. 
    - A p x p matrix. The block-diagonal of this matrix,
    as specified by groups, will be the initial values 
    for the S matrix.
    - A list of square numpy matrices, with the ith element
    corresponding to the block of the ith group in S.
    Default: Half of the identity.
    :param rec_prop: The proportion of data you are planning
    to recycle. (The optimal S matrix depends on the recycling
    proportion.)
    """

    def __init__(self, Sigma, groups, init_S=None, invSigma=None, rec_prop=0):

        super().__init__()

        # Groups MUST be sorted
        sorted_groups = np.sort(groups)
        if not np.all(groups == sorted_groups):
            raise ValueError("Sigma and groups must be sorted prior to input")

        # Save sigma and groups
        self.groups = torch.from_numpy(groups).long()
        self.group_sizes = torch.from_numpy(calc_group_sizes(groups)).long()
        self.Sigma = torch.from_numpy(Sigma).float()
        # self.register_buffer('Sigma', torch.from_numpy(Sigma))

        # Save inverse cov matrix
        if invSigma is None:
            invSigma = utilities.chol2inv(Sigma)
        self.invSigma = torch.from_numpy(invSigma).float()

        # Save recycling proportion
        self.rec_prop = rec_prop

        # Create new blocks
        if init_S is None:
            blocks = [0.5 * torch.eye(gj) for gj in self.group_sizes]
        elif isinstance(init_S, np.ndarray):
            blocks = blockdiag_to_blocks(init_S, groups)
            # Torch-ify and take sqrt
            blocks = [torch.from_numpy(block) for block in blocks]
            blocks = [torch.cholesky(block) for block in blocks]
        else:
            # Check for correct number of blocks
            num_blocks = len(init_S)
            num_groups = np.unique(groups).shape[0]
            if num_blocks != num_groups:
                raise ValueError(
                    f"Length of init_S {num_blocks} doesn't agree with num groups {num_groups}"
                )
            # Torch-ify and take sqrt
            blocks = [torch.from_numpy(block) for block in init_S]
            blocks = [torch.cholesky(block) for block in blocks]

        self.blocks = [nn.Parameter(block.float()) for block in blocks]

        # Register the blocks as parameters
        for i, block in enumerate(self.blocks):
            self.register_parameter(f"block{i}", block)

        self.update_sqrt_S()
        self.scale_sqrt_S(tol=1e-5, num_iter=10)

    def update_sqrt_S(self):
        """ Updates sqrt_S using the block parameters """
        self.sqrt_S = block_diag_sparse(*self.blocks)

    def pull_S(self):
        """ Returns the S matrix """
        self.update_sqrt_S()
        S = torch.mm(self.sqrt_S.t(), self.sqrt_S)
        return S

    def forward(self):
        """ Calculates trace of inverse grahm feature-knockoff matrix"""

        # Create schurr complement
        S = self.pull_S()
        S = (1 - self.rec_prop) * S  # Account for recycling calcing loss
        diff = self.Sigma - S
        G_schurr = self.Sigma - torch.mm(torch.mm(diff, self.invSigma), diff)

        # Take eigenvalues
        eigvals = torch.symeig(G_schurr, eigenvectors=True)
        eigvals = eigvals[0]
        inv_eigvals = 1 / eigvals
        return inv_eigvals.sum()

    def scale_sqrt_S(self, tol, num_iter):
        """ Scales sqrt_S such that 2 Sigma - S is PSD."""

        # No gradients
        with torch.no_grad():

            # Construct S
            S = self.pull_S()
            # Find optimal scaling
            _, gamma = utilities.scale_until_PSD(
                self.Sigma.numpy(), S.numpy(), tol=tol, num_iter=num_iter
            )
            # Scale blocks
            for block in self.blocks:
                block.data = np.sqrt(gamma) * block.data
            self.update_sqrt_S()

    def project(self, **kwargs):
        """ Project by scaling sqrt_S """
        self.scale_sqrt_S(**kwargs)


class NonconvexSDPSolver:
    """ 
    Projected gradient descent to solve SDP
    for non-convex loss functions (specifically 
    the trace of the feature-knockoff precision 
    matrix).
    :param Sigma: p x p numpy array, the correlation matrix
    :param groups: p-length numpy array specifying groups
    :param losscalc: A pytorch class wrapping nn.module
    which contains the following methods:
    - .forward() which calculates the loss based on the
    internally stored S matrix.
    - .project() which ensures that both the internally-stored
    S matrix as well as (2*Sigma - S) are PSD.
    - .pull_S(), which returns the internally-stored S matrix.
    If None, creates a FKPrecisionTraceLoss class. 
    :param kwargs: Passed to FKPrecisionTraceLoss 
    """

    def __init__(self, Sigma, groups, losscalc=None, **kwargs):

        # Add Sigma
        self.Sigma = Sigma
        self.groups = groups

        # Sort by groups for ease of computation
        inds, inv_inds = utilities.permute_matrix_by_groups(groups)
        self.inds = inds
        self.inv_inds = inv_inds
        self.sorted_Sigma = self.Sigma[inds][:, inds]
        self.sorted_groups = self.groups[inds]

        # Loss calculator
        if losscalc is not None:
            self.losscalc = losscalc
        else:
            self.losscalc = FKPrecisionTraceLoss(
                Sigma=self.sorted_Sigma, groups=self.sorted_groups, **kwargs
            )

        # Initialize cache of optimal S
        with torch.no_grad():
            init_loss = self.losscalc()
            if init_loss < 0:
                init_loss = np.inf
        self.cache_S(init_loss)

        # Initialize attributes which save losses over time
        self.all_losses = []
        self.projected_losses = []

    def cache_S(self, new_loss):
        # Cache optimal solution
        with torch.no_grad():
            self.opt_loss = new_loss
            self.opt_S = self.losscalc.pull_S().clone().detach().numpy()

    def optimize(
        self,
        lr=1e-2,
        sdp_verbose=False,
        max_epochs=100,
        tol=1e-5,
        line_search_iter=10,
        cache_loss=True,
        **kwargs,
    ):
        """
        :param lr: Initial learning rate (default 1e-2)
        :param sdp_verbose: if true, reports progress
        :param max_epochs: Maximum number of epochs in SGD
        :param tol: Mimimum eigenvalue allowed for PSD matrices
        :param line_search_iter: Number of line searches to do
        when scaling sqrt_S.
        :param cache: If true, cache the loss at each iteration
        for later analysis.
        """
        # Optimizer
        params = list(self.losscalc.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-2)

        for j in range(max_epochs):

            # Step 1: Calculate loss (trace of feature-knockoff precision)
            loss = self.losscalc()
            if cache_loss:
                self.all_losses.append(loss.item())

            # Step 2: Step along the graient
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Step 3: Reproject to be PSD
            if j % 10 == 0 or j == max_epochs - 1:
                self.losscalc.project(tol=tol, num_iter=line_search_iter)

                # If this is optimal after reprojecting, save
                with torch.no_grad():
                    new_loss = self.losscalc()
                if new_loss < self.opt_loss and new_loss >= 0:
                    self.cache_S(new_loss)

                # Possibly cache loss
                if cache_loss:
                    self.projected_losses.append(new_loss.item())

        # Shift, scale, and return
        sorted_S = self.opt_S
        S = sorted_S[self.inv_inds][:, self.inv_inds]
        S = utilities.shift_until_PSD(S, tol=tol)
        S, _ = utilities.scale_until_PSD(
            self.Sigma, S, tol=tol, num_iter=line_search_iter
        )
        return S


# def SDP_gradient_solver(
#     Sigma,
#     groups,
#     lr=1e-2,
#     sdp_verbose=False,

# ):
#     """ Projected gradient descent to solve the SDP.

#     TODO: possibly use diagnostic here: https://arxiv.org/pdf/1710.06382.pdf
#     to reduce the learning rate appropriately. (page 7)

#     :param Sigma: p x p numpy array, the correlation matrix
#     :param groups: p-length numpy array specifying groups
#     :param lr: Initial learning rate (default 1e-2)
#     :param sdp_verbose: if true, reports progress
#     :param max_epochs: Maximum number of epochs in SGD
#     :param tol: Mimimum eigenvalue allowed for PSD matrices
#     :param line_search_iter: Number of line searches to do
#     when scaling sqrt_S."""

#     # Initialize the model
#     fk_precision_calc = FKPrecisionTraceLoss(
#         Sigma=Sigma, groups=groups, **kwargs
#     )
#     # Optimizer
#     params = list(fk_precision_calc.parameters())
#     optimizer = torch.optim.Adam(params,lr=1e-2)

#     for j in range(max_epochs):

#         # Step 1: Calculate inverse of trace of grahm matrix
#         # and step along the gradient
#         loss = fk_precision_calc()
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         optimizer.step()

#         # Step 2: Reproject to be PSD
#         if j % 10 == 0:
#             fk_precision_calc.scale_sqrt_S(tol=tol, num_iter=line_search_iter)

#     S = fk_precision_calc.pull_S().detach().numpy()
#     return S
