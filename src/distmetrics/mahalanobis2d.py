import numpy as np
from astropy.convolution import convolve
from pydantic import BaseModel, model_validator


class MahalanobisDistance2d(BaseModel):
    dist: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_covariance_shape(cls, values: dict) -> dict:
        """Check that our covariance matrix is of the form 2 x 2 x H x W"""
        cov = values.cov
        cov_inv = values.cov_inv
        dist = values.dist
        expected_shape_cov = (2, 2, dist.shape[0], dist.shape[1])
        for cov_mat in [cov, cov_inv]:
            if expected_shape_cov != cov_mat.shape:
                expected_shape_s = ' x '.join(expected_shape_cov)
                raise ValueError(f'Covariance matrices must be of the form {expected_shape_s}')

        mean = values.mean
        expected_shape_mean = (2, dist.shape[0], dist.shape[1])
        if not (mean.shape == expected_shape_mean):
            raise ValueError(f'Mean array needs to have shape {expected_shape_mean}')
        return values


def get_spatiotemporal_mu_1d(arrs: np.ndarray, kernel_width=3, kernel_height=3) -> np.ndarray:
    k_shape = (1, kernel_height, kernel_width)
    kernel = np.ones(k_shape, dtype=np.float32) / np.prod(k_shape)

    mus_spatial = convolve(arrs, kernel, boundary='extend', nan_treatment='interpolate')
    mu_st = np.mean(mus_spatial, axis=0)
    return mu_st


def get_spatiotemporal_mu(arr_st: np.ndarray, kernel_width=3, kernel_height=3) -> np.ndarray:
    if len(arr_st.shape) != 4:
        raise ValueError('We are expecting array of shape T x 2 x H x W')
    _, C, H, W = arr_st.shape
    mu_st = np.full((C, H, W), np.nan)
    for c in range(C):
        mu_st[c, ...] = get_spatiotemporal_mu_1d(arr_st[:, c, ...])

    return mu_st


def get_spatiotemporal_var(
    arr_st: np.ndarray, mu_st=None, kernel_width=3, kernel_height=3, unbiased: bool = False
) -> np.ndarray:
    if len(arr_st.shape) != 4:
        raise ValueError('We are expecting array of shape T x 2 x H x W')
    T, C, H, W = arr_st.shape
    if mu_st is None:
        mu_st = get_spatiotemporal_mu(arr_st, kernel_width=kernel_width, kernel_height=kernel_height)
    else:
        # ensure there are means for each channel
        if mu_st.shape[0] != C:
            raise ValueError('The mean does not match dimension 1 of input arr_st')

    k_shape = (1, kernel_height, kernel_width)
    N = T * kernel_height * kernel_width

    kernel = np.ones(k_shape, dtype=np.float32) / N

    var_st = np.full((C, H, W), np.nan)
    for c in range(C):
        var_spatial_1d = convolve(
            arr_st[:, c, ...] ** 2, kernel, boundary='extend', nan_treatment='interpolate', fill_value=0
        )
        var_spatial_1d -= mu_st[c, ...] ** 2
        # np.mean vs. np.nanmean - np.mean we exclude pixels where np.nan has occurred anywhere in time series
        var_st[c, ...] = np.mean(var_spatial_1d, axis=0)
        if unbiased:
            var_st[c, ...] *= N / (N - 1)
    return var_st


def get_spatiotemporal_cor(arrs: np.ndarray, mu_st=None, kernel_width=3, kernel_height=3, unbiased=False) -> np.ndarray:
    T, C, _, _ = arrs.shape
    if C != 2:
        raise ValueError('input arrs must have 2 channels!')
    if mu_st is None:
        mu_st = get_spatiotemporal_mu(arrs, kernel_width=kernel_width, kernel_height=kernel_height)
    CC, _, _ = mu_st.shape
    if CC != 2:
        raise ValueError('spatiotemporal mean must be 2!')

    k_shape = (1, kernel_height, kernel_width)
    N = T * kernel_height * kernel_width
    kernel = np.ones(k_shape, dtype=np.float32) / N
    # corr = E(XY) - mu_X mu_Y
    term_0 = convolve(
        (arrs[:, 0, ...] * arrs[:, 1, ...]),
        kernel,
        boundary='extend',
        nan_treatment='interpolate',
    )
    term_1 = mu_st[0, ...] * mu_st[1, ...]
    corr_s = term_0 - term_1

    corr_st = np.mean(corr_s, axis=0)
    if unbiased:
        corr_st *= N / (N - 1)

    return corr_st


def get_spatiotemporal_covar(
    arrs: np.ndarray, mu_st=None, kernel_width=3, kernel_height=3, unbiased=True
) -> np.ndarray:
    if mu_st is None:
        mu_st = get_spatiotemporal_mu(arrs, kernel_width=kernel_width, kernel_height=kernel_height)

    _, C, H, W = arrs.shape
    cov_st = np.full((C, C, H, W), np.nan)
    var = get_spatiotemporal_var(
        arrs, mu_st=mu_st, kernel_width=kernel_width, kernel_height=kernel_height, unbiased=unbiased
    )
    for c in range(C):
        cov_st[c, c, ...] = var[c, ...]
    for c in range(C):
        for d in range(c, C):
            if c != d:
                covar_temp = get_spatiotemporal_cor(
                    arrs[:, [c, d], ...],
                    mu_st=mu_st[[c, d], ...],
                    kernel_width=kernel_width,
                    kernel_height=kernel_height,
                    unbiased=unbiased,
                )
                cov_st[c, d, ...] = covar_temp
                cov_st[d, c, ...] = covar_temp
    return cov_st


def eigh2d(cov_mat: np.ndarray) -> np.ndarray:
    """
    References:
    https://math.stackexchange.com/questions/395698/fast-way-to-calculate-eigen-of-2x2-matrix-using-a-formula
    https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    https://math.stackexchange.com/questions/807166/eigenvalues-in-terms-of-trace-and-determinant-for-matrices-larger-than-2-x-2
    """
    if (len(cov_mat.shape) != 4) or (cov_mat.shape[:2] != (2, 2)):
        raise ValueError('Covariance matrix need to have shape (2, 2, H, W)')

    det = cov_mat[0, 0, ...] * cov_mat[1, 1, ...] - cov_mat[0, 1, ...] * cov_mat[1, 0, ...]
    tr = cov_mat[0, 0, ...] + cov_mat[1, 1, ...]

    eigval = np.zeros((2, cov_mat.shape[2], cov_mat.shape[3]), dtype=np.float32)
    # Formulas for eigenvalues taken from here
    # https://math.stackexchange.com/questions/807166/eigenvalues-in-terms-of-trace-and-determinant-for-matrices-larger-than-2-x-2
    # note that eigval[0, ...] < eigval[1, ...] (this is the consistent with np.linalgeigh)
    eigval[0, ...] = 0.5 * (tr - np.sqrt(tr**2 - 4 * det))
    eigval[1, ...] = 0.5 * (tr + np.sqrt(tr**2 - 4 * det))

    eigvec = np.zeros(cov_mat.shape)

    # Formulas for eigenvectors taken from here:
    # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    # Divid into two cases when antidiagonal is small and not-small
    case_1 = np.abs(cov_mat[0, 1, ...]) >= 1e-7
    case_2 = ~case_1

    case_1_cov = cov_mat[..., case_1]
    case_1_eigval = eigval[..., case_1]

    # Eignvector 1
    eigvec[0, 0, case_1] = case_1_eigval[0, ...] - case_1_cov[1, 1, ...]
    eigvec[1, 0, case_1] = case_1_cov[0, 1, ...]

    # Make sure the eigenvector is normalized so that the matrix of eigenvectors has an inverse that is its transpose
    norm = np.sqrt(eigvec[1, 0, case_1] ** 2 + eigvec[0, 0, case_1] ** 2)
    eigvec[0, 0, case_1] /= norm
    eigvec[1, 0, case_1] /= norm

    # Eigenvector 2
    eigvec[0, 1, case_1] = case_1_eigval[1, ...] - case_1_cov[1, 1, ...]
    eigvec[1, 1, case_1] = case_1_cov[1, 0, ...]

    # Make sure the eigenvector is normalized so that the matrix of eigenvectors has an inverse that is its transpose
    norm = np.sqrt(eigvec[0, 1, case_1] ** 2 + eigvec[1, 1, case_1] ** 2)
    eigvec[0, 1, case_1] /= norm
    eigvec[1, 1, case_1] /= norm

    # Eigenvectors are x/y axes when antidiagnoal is small (< 1e-7)
    eigvec[0, 0, case_2] = 1
    eigvec[1, 1, case_2] = 1
    eigvec[1, 0, case_2] = eigvec[1, 0, case_2] = 0

    return eigval, eigvec


def _compute_mahalanobis_dist(
    pre_arrs: np.ndarray, post_arr: np.ndarray, kernel_width=5, kernel_height=5, eig_lb: float = 0.0001 * np.sqrt(2)
) -> tuple[np.ndarray]:
    mu_st = get_spatiotemporal_mu(pre_arrs, kernel_width=kernel_width, kernel_height=kernel_height)
    covar_st = get_spatiotemporal_covar(pre_arrs, mu_st=mu_st, kernel_width=kernel_width, kernel_height=kernel_height)

    eigval, eigvec = eigh2d(covar_st)
    # This is the floor we discused earlier except this is for the variance matrix so our LB is .01
    # We want the matrix norm to be at least .01 so we make sure each eigenvalue is .01 * \sqrt 2
    eigval_clip = np.maximum(eigval, eig_lb)
    eigval_clip_inv = eigval_clip**-1

    # Diag matrix is the diagonal matrix of eigenvalues
    diag_matrix = np.zeros(eigvec.shape, dtype=np.float32)
    diag_matrix[0, 0, ...] = eigval_clip_inv[0, ...]
    diag_matrix[1, 1, ...] = eigval_clip_inv[1, ...]

    # Matrix multiplication to reconstruct the Sigma^-1  = V^T D V where V is the
    # matrix whose colums are eignevectors and D is the diagonal matrix of eigenvalues
    covar_st_inv_floor_t = np.einsum('ijmn,jkmn->ikmn', diag_matrix, eigvec)
    covar_st_inv_floor = np.einsum('ijmn,jkmn->ikmn', eigvec.transpose([1, 0, 2, 3]), covar_st_inv_floor_t)

    # Compute the Mahalanobis distance!
    vec = post_arr - mu_st
    dist_0 = np.einsum('ijkl,jkl->ikl', covar_st_inv_floor, vec)
    dist_1 = np.einsum('ijk,ijk->jk', vec, dist_0)
    dist = np.sqrt(dist_1)
    return mu_st, covar_st, covar_st_inv_floor, dist


def _transform_pre_arrs(pre_arrs_vv: list[np.ndarray], pre_arrs_vh: list[np.ndarray]) -> np.ndarray:
    dual_pol = [np.stack([vv, vh], axis=0) for (vv, vh) in zip(pre_arrs_vv, pre_arrs_vh)]
    ts = np.stack(dual_pol, axis=0)
    return ts


def compute_mahalonobis_dist(
    pre_arrs_vv: list[np.ndarray],
    pre_arrs_vh: list[np.ndarray],
    post_arr_vv: np.ndarray,
    post_arr_vh: np.ndarray,
    kernel_size: int = 3,
    eig_lb: float = 0.0001 * np.sqrt(2),
) -> np.ndarray:
    # T x 2 x H x C arr
    pre_arrs = _transform_pre_arrs(pre_arrs_vv, pre_arrs_vh)
    # 2 x H x C
    post_arr = np.stack([post_arr_vv, post_arr_vh], axis=0)
    mu_st, cov_st, covar_st_inv, dist = _compute_mahalanobis_dist(
        pre_arrs, post_arr, kernel_height=kernel_size, kernel_width=kernel_size, eig_lb=eig_lb
    )
    distance = MahalanobisDistance2d(dist=dist, mean=mu_st, cov=cov_st, cov_inv=covar_st_inv)
    return distance
