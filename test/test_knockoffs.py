import numpy as np
import scipy as sp
import unittest
from .context import knockadapt
from statsmodels.stats.moment_helpers import cov2corr

from knockadapt import utilities, graphs, knockoffs, nonconvex_sdp


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


class TestEquicorrelated(CheckSMatrix):
    """ Tests equicorrelated knockoffs and related functions """

    def test_eigenvalue_calculation(self):

        # Test to make sure non-group and group versions agree
        # (in the case of no grouping)
        p = 100
        groups = np.arange(0, p, 1) + 1
        for rho in [0, 0.3, 0.5, 0.7]:
            V = np.zeros((p, p)) + rho
            for i in range(p):
                V[i, i] = 1
            expected_gamma = min(1, 2*(1-rho))
            gamma = knockoffs.calc_min_group_eigenvalue(
                Sigma=V, groups=groups, 
            )
            np.testing.assert_almost_equal(
                gamma, expected_gamma, decimal = 3, 
                err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'
            )

        # Test non equicorrelated version
        V = np.random.randn(p, p)
        V = np.dot(V.T, V) + 0.1*np.eye(p)
        V = cov2corr(V)
        expected_gamma = min(1, 2*np.linalg.eigh(V)[0].min())
        gamma = knockoffs.calc_min_group_eigenvalue(
            Sigma=V, groups=groups
        )
        np.testing.assert_almost_equal(
            gamma, expected_gamma, decimal = 3, 
            err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'

        )

    def test_equicorrelated_construction(self):

        # Test S matrix construction
        p = 100
        groups = np.arange(0, p, 1) + 1
        V = np.random.randn(p, p)
        V = np.dot(V.T, V) + 0.1*np.eye(p)
        V = cov2corr(V)

        # Expected construction
        expected_gamma = min(1, 2*np.linalg.eigh(V)[0].min())
        expected_S = expected_gamma*np.eye(p)

        # Equicorrelated
        S = knockoffs.equicorrelated_block_matrix(Sigma=V, groups=groups)

        # Test to make sure the answer is expected
        np.testing.assert_almost_equal(
            S, expected_S, decimal = 3, 
            err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'

        )

        # # Do it again with a block matrix - start by constructing
        # # something we can easily analyze
        # group_sizes = [3, 2, 3, 4, 1]
        # groups = []
        # for i, size in enumerate(group_sizes):
        #   to_add = [i]*size
        #   groups += to_add
        # groups = np.array(groups) + 1

    def test_psd(self):

        # Test S matrix construction
        p = 100
        V = np.random.randn(p, p)
        V = np.dot(V.T, V) + 0.1*np.eye(p)
        V = cov2corr(V)

        # Create random groups
        groups = np.random.randint(1, p, size=(p))
        groups = utilities.preprocess_groups(groups)
        S = knockoffs.equicorrelated_block_matrix(Sigma=V, groups=groups)

        # Check S properties
        self.check_S_properties(V, S, groups)




class TestSDP(CheckSMatrix):
    """ Tests an easy case of SDP and ASDP """

    def test_easy_sdp(self):


        # Test non-group SDP first
        n = 200
        p = 50
        X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
            n = n, p = p, gamma = 0.3
        )

        # S matrix
        trivial_groups = np.arange(0, p, 1) + 1
        S_triv = knockoffs.solve_group_SDP(corr_matrix, trivial_groups)
        np.testing.assert_array_almost_equal(
            S_triv, np.eye(p), decimal = 2,
            err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_triv, trivial_groups)

        # Repeat for gaussian_knockoffs method
        _, S_triv2 = knockoffs.gaussian_knockoffs(
            X = X, Sigma = corr_matrix, groups = trivial_groups, 
            return_S = True, sdp_verbose = False, verbose = False,
            method = 'sdp'
        )
        np.testing.assert_array_almost_equal(
            S_triv2, np.eye(p), decimal = 2, 
            err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_triv2, trivial_groups)

        # Test slightly harder case
        _,_,_,_, expected_out, _ = graphs.daibarber2016_graph(
            n = n, p = p, gamma = 0
        )
        _, S_harder = knockoffs.gaussian_knockoffs(
            X = X, Sigma = corr_matrix, groups = groups, 
            return_S = True, sdp_verbose = False, verbose = False,
            method = 'sdp'
        )
        np.testing.assert_almost_equal(
            S_harder, expected_out, decimal = 2,
            err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_harder, groups)

        # Repeat for ASDP
        _, S_harder_ASDP = knockoffs.gaussian_knockoffs(
            X = X, Sigma = corr_matrix, groups = groups, method = 'ASDP',
            return_S = True, sdp_verbose = False, verbose = False
        )
        np.testing.assert_almost_equal(
            S_harder_ASDP, expected_out, decimal = 2,
            err_msg = 'solve_group_ASDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_harder_ASDP, groups)


    def test_equicorr_SDP(self):

        # Test non-group SDP on equicorrelated cov matrix
        p = 100
        rho = 0.8
        V = rho*np.ones((p,p)) + (1-rho)*np.eye(p)
        S = knockoffs.solve_group_SDP(V, verbose=True)
        expected = (2 - 2*rho) * np.eye(p)
        np.testing.assert_almost_equal(
            S, expected, decimal = 2,
            err_msg = 'solve_SDP does not produce optimal S matrix (equicorrelated graph)'
        )

    def test_sdp_tolerance(self):

        # Get graph
        np.random.seed(110)
        Q = graphs.ErdosRenyi(p=50, tol=1e-1)
        V = cov2corr(utilities.chol2inv(Q))
        groups = np.concatenate([np.zeros(10) + j for j in range(5)]) + 1
        groups = groups.astype('int32')

        # Solve SDP
        for tol in [1e-3, 0.01, 0.02]:
            S = knockoffs.solve_group_SDP(
                Sigma=V, 
                groups=groups, 
                sdp_verbose=False, 
                objective="pnorm",  
                num_iter=10,
                tol=tol
            )
            G = np.hstack([np.vstack([V, V-S]), np.vstack([V-S, V])])
            mineig = np.linalg.eig(G)[0].min()
            self.assertTrue(
                tol - mineig < 1e3,
                f'sdp solver fails to control minimum eigenvalues: tol is {tol}, val is {mineig}'
            )
            self.check_S_properties(V, S, groups)


    def test_corrmatrix_errors(self):
        """ Tests that SDP raises informative errors when sigma is not scaled properly"""

        # Get graph
        np.random.seed(110)
        Q = graphs.ErdosRenyi(p=50, tol=1e-1)
        V = utilities.chol2inv(Q)
        groups = np.concatenate([np.zeros(10) + j for j in range(5)]) + 1
        groups = groups.astype('int32')


        # Helper function
        def SDP_solver():
            return knockoffs.solve_group_SDP(V, groups)

        # Make sure the value error increases 
        self.assertRaisesRegex(
            ValueError, "Sigma is not a correlation matrix",
            SDP_solver
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

    def test_equicorrelated_soln_recycled(self):

        # Main constants 
        p = 50
        groups = np.arange(1, p+1, 1)

        # Test separately with the recycling proportion param
        rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
        true_rec_props = [0.5, 0.25, 0.8, 0.5, 0.5]
        for true_rec_prop, rho in zip(true_rec_props, rhos):
            
            # Construct Sigma
            Sigma = np.zeros((p, p)) + rho
            Sigma += (1-rho)*np.eye(p)

            # Expected solution
            opt_prop_rec = min(rho, 0.5)
            max_S_val = min(1, 2-2*rho)
            normal_opt = (1-opt_prop_rec)*max_S_val
            new_opt = min(2-2*rho, normal_opt/(1-true_rec_prop))
            expected = new_opt*np.eye(p)

            # Test optimizer
            opt = nonconvex_sdp.NonconvexSDPSolver(
                Sigma=Sigma,
                groups=groups,
                init_S=None,
                rec_prop=true_rec_prop
            )
            opt_S = opt.optimize(
                sdp_verbose=False,
                tol=1e-5,
                max_epochs=300,
                line_search_iter=10,
            )
            self.check_S_properties(Sigma, opt_S, groups)
            np.testing.assert_almost_equal(
                opt_S, expected, decimal=1,
                err_msg=f'For equicorrelated cov rho={rho} rec_prop={true_rec_prop}, SDPgrad_solver returns {opt_S}, expected {expected}'
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

class TestKnockoffGen(unittest.TestCase):
    """ Tests whether knockoffs have correct distribution empirically"""

    def test_method_parser(self):

        # Easiest test
        method1 = 'hello'
        out1 = knockoffs.parse_method(method1, None, None)
        self.assertTrue(
            out1 == method1, 
            "parse method fails to return non-None methods"
        )

        # Default is mcv
        p = 1000
        groups = np.arange(1, p+1, 1)
        out2 = knockoffs.parse_method(None, groups, p)
        self.assertTrue(
            out2 == 'mcv', 
            "parse method fails to return mcv by default"
        )

        # Otherwise SDP
        groups[-1] = 1
        out2 = knockoffs.parse_method(None, groups, p)
        self.assertTrue(
            out2 == 'sdp', 
            "parse method fails to return SDP for grouped knockoffs"
        )

        # Otherwise ASDP
        p = 1001
        groups = np.ones(p)
        out2 = knockoffs.parse_method(None, groups, p)
        self.assertTrue(
            out2 == 'asdp', 
            "parse method fails to return asdp for large p"
        )

    def test_error_raising(self):

        # Generate data
        n = 10
        p = 100
        X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
            n = n, p = p, gamma = 1, rho = 0.8
        )
        S_bad = np.eye(p)

        def fdr_vio_knockoffs():
            knockoffs.gaussian_knockoffs(
                X=X, 
                Sigma=corr_matrix,
                S=S_bad,
                sdp_verbose=False, 
                verbose=False
            )

        self.assertRaisesRegex(
            np.linalg.LinAlgError,
            "meaning FDR control violations are extremely likely",
            fdr_vio_knockoffs, 
        )

        # Test FX knockoff violations
        def fx_knockoffs_low_n():
            knockoffs.gaussian_knockoffs(
                X=X,
                Sigma=corr_matrix,
                S=None,
                fixedX=True,
            )

        self.assertRaisesRegex(
            np.linalg.LinAlgError,
            "FX knockoffs can't be generated with n",
            fx_knockoffs_low_n, 
        )

        # Test unsupplied Sigma
        def mx_nosigma():
            knockoffs.gaussian_knockoffs(
                X=X, fixedX=False,
            )

        self.assertRaisesRegex(
            ValueError,
            "When fixedX is False, Sigma must be provided",
            mx_nosigma, 
        )

    def test_MX_knockoff_dist(self):

        # Test knockoff construction for MCV and SDP
        # on equicorrelated matrices
        n = 100000
        copies = 3
        p = 10
        for rho in [0.1, 0.9]:
            for gamma in [0.5, 1]:
                for method in ['mcv', 'sdp']:
                    X,_,_,_, corr_matrix,_ = graphs.daibarber2016_graph(
                        n = n, p = p, gamma = gamma, rho = rho
                    )
                    # S matrix
                    all_knockoffs, S = knockoffs.gaussian_knockoffs(
                        X=X,
                        Sigma=corr_matrix,
                        copies=copies,
                        method=method, 
                        return_S=True,
                        sdp_verbose=True, 
                        verbose=True
                    )

                    # Calculate empirical covariance matrix
                    knockoff_copy = all_knockoffs[:, :, 0]
                    features = np.concatenate([X, knockoff_copy], axis = 1)
                    G_hat = np.corrcoef(features, rowvar=False)

                    # Calculate population version
                    G = np.concatenate(
                        [np.concatenate([corr_matrix, corr_matrix-S]),
                        np.concatenate([corr_matrix-S, corr_matrix])],
                        axis=1
                    )

                    # Test G has correct structure
                    msg = f"Feature-knockoff cov matrix has incorrect values"
                    msg += f"for daibarber graph, MX knockoffs, rho = {rho}, gamma = {gamma}"
                    np.testing.assert_array_almost_equal(G_hat, G, 2, msg)


    def test_FX_knockoff_dist(self):
        # Test knockoff construction for MCV and SDP
        # on equicorrelated matrices
        n = 1000
        p = 5
        for rho in [0.1, 0.9]:
            for gamma in [0.5, 1]:
                for method in ['mcv', 'sdp']:
                    # X values
                    X,_,_,_,corr_matrix,_ = graphs.daibarber2016_graph(
                        n = n, p = p, gamma = gamma, rho = rho
                    )
                    # S matrix
                    trivial_groups = np.arange(0, p, 1) + 1
                    all_knockoffs, S = knockoffs.gaussian_knockoffs(
                        X=X, 
                        fixedX=True,
                        copies=int(gamma)+1,
                        method=method, 
                        return_S=True,
                        sdp_verbose=False, 
                        verbose=True
                    )

                    # Scale properly so we can calculate
                    scale = np.sqrt(np.diag(np.dot(X.T, X)).reshape(1, -1))
                    X = X / scale
                    knockoff_copy = all_knockoffs[:, :, 0] / scale

                    # # Compute empirical (scaled) cov matrix
                    features = np.concatenate([X, knockoff_copy], axis = 1)
                    G_hat = np.dot(features.T, features)
                    
                    # Calculate what this should be
                    Sigma = np.dot(X.T, X)
                    G = np.concatenate(
                        [np.concatenate([Sigma, Sigma-S]),
                        np.concatenate([Sigma-S, Sigma])],
                        axis=1
                    )

                    # Test G has correct structure
                    msg = f"Feature-knockoff cov matrix has incorrect values"
                    msg += f"for daibarber graph, FX knockoffs, rho = {rho}, gamma = {gamma}"
                    np.testing.assert_array_almost_equal(G_hat, G, 5, msg)


if __name__ == '__main__':
    unittest.main()