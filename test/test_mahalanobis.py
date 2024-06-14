import numpy as np
from distmetrics.mahalanobis2d import eigh2d, get_spatiotemporal_covar, get_spatiotemporal_mu, get_spatiotemporal_var

np.random.seed(0)


def test_eigh():
    """The inputs of our eigh2d function are 2 x 2 x H X W where we want to perform eigh on all of the H X W entries.

    Generates random symmetric matrices perturbed from 2 x 2 matrix of all ones to ensure rank 2 and still symmetric.
    We perturb the matrix by a random number along the anti-diagonal (i.e. indices (0, 1) and (1, 0)). We use a
    normal RV ~N(0, 3). This test verifies that numpy routine generates same output as ours modulo the direction
    of eigenvectors."""
    M = 5
    N = 6

    cov_mat = np.ones((2, 2, M, N))
    eps = np.random.randn(M, N) * 3
    # perturb along anti-diagonal
    # There is zero probability that eps is precisely 0
    cov_mat[1, 0, ...] += eps
    cov_mat[0, 1, ...] += eps

    eig_val, eig_vec = eigh2d(cov_mat)

    for i in range(M):
        for j in range(N):
            cov_temp = cov_mat[..., i, j]
            np_result = np.linalg.eigh(cov_temp)
            eigval_expected, eigvec_expected = np_result.eigenvalues, np_result.eigenvectors
            np.testing.assert_almost_equal(eig_val[0, i, j], eigval_expected[0], 5)
            np.testing.assert_almost_equal(eig_val[1, i, j], eigval_expected[1], 5)

            # Ensuring same direction of all vectors
            v0 = eigvec_expected[:, 0]
            v1 = eigvec_expected[:, 1]
            v0 = v0 if v0[0] > 0 else -v0
            v1 = v1 if v1[0] > 0 else -v1

            w0 = eig_vec[:, 0, i, j]
            w1 = eig_vec[:, 1, i, j]
            w0 = w0 if w0[0] > 0 else -w0
            w1 = w1 if w1[0] > 0 else -w1

            np.testing.assert_almost_equal(v0, w0, 5)
            np.testing.assert_almost_equal(v1, w1, 5)


def test_spatiotemporal_mean():
    """We are taking the spatiatemporal mean within a small 3 x 3 patch and throughout the time.
    Our sample data has spatial size 3 x 3 so if we consider the center pixel (i.e. index 1, 1),
    then we can use the normal np.mean"""
    T, C, H, W = 10, 2, 3, 3
    N_samples = 10
    for n in range(N_samples):
        sample_data = np.random.randn(T, C, H, W)
        sample_data_np = sample_data.transpose([1, 0, 2, 3]).reshape((C, -1))
        mean_st = get_spatiotemporal_mu(sample_data, kernel_height=3, kernel_width=3)[:, 1, 1]
        mean_ex = np.mean(sample_data_np, axis=1)
        np.testing.assert_almost_equal(mean_st, mean_ex, 7)


def test_spatiotemporal_var():
    """We are taking the spatiatemporal variance within a small 3 x 3 patch and throughout the time.
    Our sample data has spatial size 3 x 3 so if we consider the center pixel (i.e. index 1, 1), then
    we can use the normal np.var"""
    T, C, H, W = 10, 2, 3, 3
    N = T * H * W
    N_samples = 10
    for n in range(N_samples):
        sample_data = np.random.randn(T, C, H, W)
        var_st = get_spatiotemporal_var(sample_data, kernel_height=3, kernel_width=3)[:, 1, 1]
        var_st_unbiased = get_spatiotemporal_var(sample_data, kernel_height=3, kernel_width=3, unbiased=True)[:, 1, 1]
        var_np = np.var(sample_data, axis=(0, 2, 3))
        np.testing.assert_almost_equal(var_st, var_np, 7)
        np.testing.assert_almost_equal(var_st_unbiased, var_np * N / (N - 1), 7)


def test_covariance_generation():
    T, C, H, W = 10, 2, 3, 3
    N_samples = 10
    for n in range(N_samples):
        for unbiased in [True, False]:
            sample_data = np.random.randn(T, C, H, W)
            sample_data_np = sample_data.transpose([1, 0, 2, 3]).reshape((C, -1))
            cov_np = np.cov(sample_data_np, rowvar=True, bias=(not unbiased))
            cov = get_spatiotemporal_covar(sample_data, kernel_height=3, kernel_width=3, unbiased=unbiased)
            cov_test = cov[:, :, 1, 1]
            np.testing.assert_almost_equal(cov_np, cov_test)
