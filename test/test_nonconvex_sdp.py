import numpy as np
import scipy as sp
import unittest
from .context import knockadapt

from knockadapt import nonconvex_sdp, utilities


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

class TestUtilFunctions(unittest.TestCase):
    """ Tests a couple of the block-diagonal utility functions"""

    def test_blockdiag_to_blocks(self):

        # Create block sizes and blocks
        block_nos = knockadapt.utilities.preprocess_groups(
            np.random.randint(1, 50, 100)
        )        
        block_nos = np.sort(block_nos)
        block_sizes = knockadapt.utilities.calc_group_sizes(block_nos)
        blocks = [np.random.randn(b, b) for b in block_sizes]

        # Create block diagonal matrix in scipy
        block_diag = sp.linalg.block_diag(*blocks)
        blocks2 = nonconvex_sdp.blockdiag_to_blocks(block_diag, block_nos)
        for expected, out in zip(blocks, blocks2):
            np.testing.assert_almost_equal(
                out, expected, err_msg='blockdiag_to_blocks incorrectly separates blocks'
            )


class TestNonconvexSDP(CheckSMatrix):
    """ Tests the NonconvexSDPSOlver, FKPRecisionTraceLoss classes"""

    def test_scale_sqrt_S(self):
        """ Tests the function which scales sqrt S"""

        # Construct covariance matrix
        p = 50
        rho = 0.8
        Sigma = np.zeros((p, p)) + rho
        Sigma += (1-rho)*np.eye(p)
        groups = np.arange(1, p+1, 1)
        init_blocks = [np.eye(1) for _ in range(p)]

        # Create model - this automatically scales the
        # initial blocks properly
        fk_precision_calc = nonconvex_sdp.FKPrecisionTraceLoss(
            Sigma, groups, init_S=init_blocks
        )
        # Check for proper scaling
        S = fk_precision_calc.pull_S().detach().numpy()
        expected = min(1, 2-2*rho)*np.eye(p)
        np.testing.assert_almost_equal(
            S, expected, decimal=1,
            err_msg=f'Initial scaling fails, expected {expected} but got {S} for equicorrelated rho={rho}'
        )

    def test_group_sorting_error(self):
        """ Tests that InvGrahmTrace class raises an error if the cov 
        matrix/groups are improperly sorted"""

        # Groups and sigma 
        p = 50
        Sigma = np.eye(p)
        groups = knockadapt.utilities.preprocess_groups(
            np.random.randint(1, p+1, p)
        )

        # Try to initialize
        def init_unsorted_model():
            model = nonconvex_sdp.FKPrecisionTraceLoss(Sigma, groups)

        self.assertRaisesRegex(
            ValueError, "Sigma and groups must be sorted prior to input",
            init_unsorted_model
        )

    def test_identity_soln(self):
        """ Tests that InvGrahmTrace comes up with the correct
        solution for the identity matrix""" 

        # Calc group sizes 
        p = 50
        groups = knockadapt.utilities.preprocess_groups(
            np.random.randint(1, p+1, p)
        )
        groups = np.sort(groups)
        group_sizes = knockadapt.utilities.calc_group_sizes(groups)

        # Set up model
        Sigma = np.eye(p)
        init_blocks = [0.5*np.eye(gj) for gj in group_sizes]
        model = nonconvex_sdp.FKPrecisionTraceLoss(
            Sigma, groups, init_S=init_blocks
        )

        # Run basic functions (foward, scale_sqrt_S)
        out = model.forward()
        model.scale_sqrt_S(tol=1e-5, num_iter=10)

        # Run optimizer
        opt = nonconvex_sdp.NonconvexSDPSolver(
            Sigma=Sigma,
            groups=groups,
            init_S=init_blocks
        )
        opt_S = opt.optimize(
            sdp_verbose=False,
            tol=1e-5,
            max_epochs=100,
            line_search_iter=10,
        )
        self.check_S_properties(Sigma, opt_S, groups)
        np.testing.assert_almost_equal(
            opt_S, np.eye(p), decimal=1,
            err_msg=f'For identity, SDPgrad_solver returns {opt_S}, expected {np.eye(p)}'
        )


    def test_equicorrelated_soln(self):
        """ Tests that InvGrahmTrace comes up with the correct
        solution for equicorrelated matrices """

        # Main constants 
        p = 50
        groups = np.arange(1, p+1, 1)

        # Construct equicorrelated matrices
        rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
        for rho in rhos:
            
            # Construct Sigma
            Sigma = np.zeros((p, p)) + rho
            Sigma += (1-rho)*np.eye(p)

            # Expected solution
            opt_prop_rec = min(rho, 0.5)
            max_S_val = min(1, 2-2*rho)
            expected = (1-opt_prop_rec)*max_S_val*np.eye(p)

            # Test optimizer
            opt = nonconvex_sdp.NonconvexSDPSolver(
                Sigma=Sigma,
                groups=groups,
                init_S=None
            )
            opt_S = opt.optimize(
                sdp_verbose=False,
                tol=1e-5,
                max_epochs=100,
                line_search_iter=10,
            )
            self.check_S_properties(Sigma, opt_S, groups)
            np.testing.assert_almost_equal(
                opt_S, expected, decimal=1,
                err_msg=f'For equicorrelated cov rho={rho}, SDPgrad_solver returns {opt_S}, expected {expected}'
            )

    def test_ar1_soln(self):

        # Construct AR1 graph + groups
        np.random.seed(110)
        p = 50
        a = 1
        b = 1
        groups = knockadapt.utilities.preprocess_groups(
            np.random.randint(1, p+1, p)
        )
        Sigma = knockadapt.graphs.AR1(a=a, b=b, p=p)

        # Use SDP as baseline
        init_S = knockadapt.knockoffs.solve_group_SDP(Sigma, groups)
        init_loss = nonconvex_sdp.fk_precision_trace(Sigma, init_S)

        # Apply gradient solver
        opt = nonconvex_sdp.NonconvexSDPSolver(
            Sigma,
            groups,
            init_S=init_S
        )
        opt_S = opt.optimize(
            sdp_verbose=False,
            tol=1e-5,
            max_epochs=100,
            line_search_iter=10,
        )
        new_loss = nonconvex_sdp.fk_precision_trace(Sigma, opt_S)

        # Check S matrix
        self.check_S_properties(Sigma, opt_S, groups)
        
        # Check new loss < init_loss
        self.assertTrue(
            new_loss <= init_loss,
            msg=f"For AR1, noncnvx solver has higher loss {new_loss} v. convex baseline {init_loss}"
        )

    def test_ER_solution(self):

        # Create DGP
        np.random.seed(110)
        p=50
        groups = knockadapt.utilities.preprocess_groups(
            np.random.randint(1, int(p/2), p)
        )
        _,_,_,_,Sigma = knockadapt.graphs.sample_data(
            method='ErdosRenyi', p=p,
        )

        # Use SDP as baseline
        init_S = knockadapt.knockoffs.solve_group_SDP(Sigma, groups)
        init_loss = nonconvex_sdp.fk_precision_trace(Sigma, init_S)

        # Apply gradient solver
        opt = nonconvex_sdp.NonconvexSDPSolver(
            Sigma,
            groups,
            init_S=init_S
        )
        opt_S = opt.optimize(
            sdp_verbose=False,
            tol=1e-5,
            max_epochs=100,
            line_search_iter=10,
        )
        new_loss = nonconvex_sdp.fk_precision_trace(Sigma, opt_S)

        # Check S matrix
        self.check_S_properties(Sigma, opt_S, groups)
        
        # Check new loss < init_loss
        self.assertTrue(
            new_loss <= init_loss,
            msg=f"For ERenyi (p={p}), noncnvx has higher loss {new_loss} v. convex baseline {init_loss}"
        )




if __name__ == '__main__':
    unittest.main()