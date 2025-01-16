import numpy as np
from astropy.convolution import convolve
from pydantic import BaseModel, model_validator
from scipy.special import logit


class MahalanobisDistance1d(BaseModel):
    dist: np.ndarray | list
    mean: np.ndarray
    std: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_shape(cls, values: dict) -> dict:
        dist = values.dist if not isinstance(values.dist, list) else values.dist[0]
        mean = values.mean
        std = values.std

        if any([dist.shape != arr.shape for arr in [std, mean]]):
            raise ValueError('All arrays must have same shape')


class MahalanobisDistance2d(BaseModel):
    dist: np.ndarray | list
    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_covariance_shape(cls, values: dict) -> dict:
        """Check that our covariance matrix is of the form 2 x 2 x H x W."""
        cov = values.cov
        cov_inv = values.cov_inv
        dist = values.dist if not isinstance(values.dist, list) else values.dist[0]
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


def get_spatiotemporal_mu_1d(arrs: np.ndarray, window_size: int = 3) -> np.ndarray:
    k_shape = (1, window_size, window_size)
    kernel = np.ones(k_shape, dtype=np.float32) / np.prod(k_shape)

    mu_spatial = convolve(arrs, kernel, boundary='extend', nan_treatment='interpolate')
    mu_st = np.nanmean(mu_spatial, axis=0)
    return mu_st


def get_spatiotemporal_var_1d(
    arrs: np.ndarray, mu: np.ndarray = None, window_size: int = 3, unbiased: bool = True
) -> np.ndarray:
    T = arrs.shape[0]
    if mu is None:
        mu = get_spatiotemporal_mu_1d(arrs, window_size=window_size)

    k_shape = (1, window_size, window_size)
    N = T * window_size * window_size
    kernel = np.ones(k_shape, dtype=np.float32) / N

    var_spatial = convolve(arrs**2, kernel, boundary='extend', nan_treatment='interpolate', fill_value=0)
    var_spatial -= mu**2
    # np.mean vs. np.nanmean - np.mean we exclude pixels where np.nan has occurred anywhere in time series
    var_st = np.nanmean(var_spatial, axis=0)
    if unbiased:
        var_st *= N / (N - 1)
    return var_st


def get_spatiotemporal_mu(arr_st: np.ndarray, window_size: int = 3) -> np.ndarray:
    if len(arr_st.shape) != 4:
        raise ValueError('We are expecting array of shape T x 2 x H x W')
    _, C, H, W = arr_st.shape
    mu_st = np.full((C, H, W), np.nan)
    for c in range(C):
        mu_st[c, ...] = get_spatiotemporal_mu_1d(arr_st[:, c, ...])

    return mu_st


def get_spatiotemporal_var(
    arr_st: np.ndarray, mu_st: np.ndarray = None, window_size: int = 3, unbiased: bool = False
) -> np.ndarray:
    if len(arr_st.shape) != 4:
        raise ValueError('We are expecting array of shape T x 2 x H x W')
    _, C, H, W = arr_st.shape
    if mu_st is None:
        mu_st = get_spatiotemporal_mu(arr_st, window_size=window_size)
    else:
        # ensure there are means for each channel
        if mu_st.shape[0] != C:
            raise ValueError('The mean does not match dimension 1 of input arr_st')

    var_st = np.full((C, H, W), np.nan)
    for c in range(C):
        var_st[c, ...] = get_spatiotemporal_var_1d(
            arr_st[:, c, ...],
            mu=mu_st[c, ...],
            window_size=window_size,
            unbiased=unbiased,
        )
    return var_st


def get_spatiotemporal_cor(
    arrs: np.ndarray, mu_st: np.ndarray | None = None, window_size: int = 3, unbiased: bool = False
) -> np.ndarray:
    T, C, _, _ = arrs.shape
    if C != 2:
        raise ValueError('input arrs must have 2 channels!')
    if mu_st is None:
        mu_st = get_spatiotemporal_mu(arrs, window_size=window_size)
    CC, _, _ = mu_st.shape
    if CC != 2:
        raise ValueError('spatiotemporal mean must be 2!')

    k_shape = (1, window_size, window_size)
    N = T * window_size * window_size
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

    corr_st = np.nanmean(corr_s, axis=0)
    if unbiased:
        corr_st *= N / (N - 1)

    return corr_st


def get_spatiotemporal_covar(
    arrs: np.ndarray, mu_st: np.ndarray = None, window_size: int = 3, unbiased: bool = True
) -> np.ndarray:
    if mu_st is None:
        mu_st = get_spatiotemporal_mu(arrs, window_size=window_size)

    _, C, H, W = arrs.shape
    cov_st = np.full((C, C, H, W), np.nan)
    var = get_spatiotemporal_var(arrs, mu_st=mu_st, window_size=window_size, unbiased=unbiased)
    for c in range(C):
        cov_st[c, c, ...] = var[c, ...]
    for c in range(C):
        for d in range(c, C):
            if c != d:
                covar_temp = get_spatiotemporal_cor(
                    arrs[:, [c, d], ...],
                    mu_st=mu_st[[c, d], ...],
                    window_size=window_size,
                    unbiased=unbiased,
                )
                cov_st[c, d, ...] = covar_temp
                cov_st[d, c, ...] = covar_temp
    return cov_st


def eigh2d(cov_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a 2 x 2 x H x W covariance matrices.

    References
    ----------
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


def _compute_mahalanobis_dist_2d(
    pre_arrs: np.ndarray,
    post_arr: np.ndarray | list,
    window_size: int = 5,
    eig_lb: float = 1e-7 * np.sqrt(2),
    unbiased: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | list]:
    mu_st = get_spatiotemporal_mu(pre_arrs, window_size=window_size)
    covar_st = get_spatiotemporal_covar(pre_arrs, mu_st=mu_st, window_size=window_size, unbiased=unbiased)

    eigval, eigvec = eigh2d(covar_st)
    # This is the floor we discused earlier except this is for the variance matrix so our LB is .01
    # We want the matrix norm to be at least .01 so we make sure each eigenvalue is .01 * \sqrt 2
    eigval_clip = np.maximum(eigval, eig_lb)
    eigval_clip_inv = eigval_clip**-1

    # Diag matrix is the diagonal matrix of eigenvalues
    diag_matrix = np.zeros(eigvec.shape, dtype=np.float32)
    diag_matrix[0, 0, ...] = eigval_clip_inv[0, ...]
    diag_matrix[1, 1, ...] = eigval_clip_inv[1, ...]

    # Matrix multiplication to reconstruct the Sigma^-1  = V D V.T where V is the
    # matrix whose colums are eignevectors and D is the diagonal matrix of eigenvalues
    covar_st_inv_floor_t = np.einsum('ijmn,jkmn->ikmn', diag_matrix, eigvec.transpose([1, 0, 2, 3]))
    covar_st_inv_floor = np.einsum('ijmn,jkmn->ikmn', eigvec, covar_st_inv_floor_t)

    # Compute the Mahalanobis distance!
    def compute_distance(post_arr_t: np.ndarray) -> np.ndarray:
        vec = post_arr_t - mu_st
        dist_0 = np.einsum('ijkl,jkl->ikl', covar_st_inv_floor, vec)
        dist_1 = np.einsum('ijk,ijk->jk', vec, dist_0)
        distance = np.sqrt(dist_1)
        return distance

    if isinstance(post_arr, list):
        dist = [compute_distance(arr) for arr in post_arr]
    else:
        dist = compute_distance(post_arr)

    return mu_st, covar_st, covar_st_inv_floor, dist


def _transform_pre_arrs(
    pre_arrs_vv: list[np.ndarray], pre_arrs_vh: list[np.ndarray], logit_transformed: bool = False
) -> np.ndarray:
    if len(pre_arrs_vh) != len(pre_arrs_vv):
        raise ValueError('Both vv and vh pre-arrays must have the same length')
    dual_pol = [np.stack([vv, vh], axis=0) for (vv, vh) in zip(pre_arrs_vv, pre_arrs_vh)]
    ts = np.stack(dual_pol, axis=0)
    if logit_transformed:
        ts = logit(ts)
    return ts


def _transform_post_arrs(
    post_arr_vv: list | np.ndarray, post_arr_vh: list | np.ndarray, logit_transformed: bool = False
) -> np.ndarray:
    if isinstance(post_arr_vv, list) != isinstance(post_arr_vh, list):
        raise ValueError('Both post arrays must be both lists or arrays')

    if isinstance(post_arr_vv, list):
        if len(post_arr_vh) != len(post_arr_vh):
            raise ValueError('Both post array lists must be the same size')
        post_arr = [np.stack([vv, vh], axis=0) for (vv, vh) in zip(post_arr_vv, post_arr_vh)]
    else:
        post_arr = np.stack([post_arr_vv, post_arr_vh], axis=0)

    if logit_transformed:
        post_arr = logit(post_arr)
    return post_arr


def compute_mahalonobis_dist_2d(
    pre_arrs_vv: list[np.ndarray],
    pre_arrs_vh: list[np.ndarray],
    post_arr_vv: np.ndarray | list,
    post_arr_vh: np.ndarray | list,
    window_size: int = 3,
    eig_lb: float = 1e-4 * np.sqrt(2),
    logit_transformed: bool = False,
    unbiased: bool = True,
) -> MahalanobisDistance2d:
    if (len(pre_arrs_vv) == 0) or (len(pre_arrs_vh) == 0):
        raise ValueError('Both vv and vh pre-image lists must be non-empty!')

    # T x 2 x H x C arr
    pre_arrs = _transform_pre_arrs(pre_arrs_vv, pre_arrs_vh, logit_transformed=logit_transformed)

    # 2 x H x C or list of such arrays
    post_arr = _transform_post_arrs(post_arr_vv, post_arr_vh, logit_transformed=logit_transformed)

    mu_st, cov_st, covar_st_inv, dist = _compute_mahalanobis_dist_2d(
        pre_arrs, post_arr, window_size=window_size, eig_lb=eig_lb, unbiased=unbiased
    )
    distance = MahalanobisDistance2d(dist=dist, mean=mu_st, cov=cov_st, cov_inv=covar_st_inv)
    return distance


def compute_mahalonobis_dist_1d(
    pre_arrs: list[np.ndarray],
    post_arr: np.ndarray | list[np.ndarray],
    window_size: int = 3,
    unbiased: bool = True,
    sigma_lb: float = 1e-4,
    logit_transformed: bool = False,
) -> MahalanobisDistance1d | list[MahalanobisDistance1d]:
    if len(pre_arrs) == 0:
        return []
    pre_arrs_s = np.stack(pre_arrs, axis=0)
    if logit_transformed:
        pre_arrs_s = logit(pre_arrs_s)
    mu = get_spatiotemporal_mu_1d(pre_arrs_s, window_size=window_size)
    sigma = get_spatiotemporal_var_1d(pre_arrs_s, mu=mu, window_size=window_size, unbiased=unbiased)
    sigma = np.sqrt(sigma)
    if isinstance(post_arr, list):
        dists = [np.abs(arr - mu) / np.maximum(sigma, sigma_lb) for arr in post_arr]
        result = MahalanobisDistance1d(dist=dists, mean=mu, std=sigma)
    else:
        dist = np.abs(post_arr - mu) / np.maximum(sigma, sigma_lb)
        result = MahalanobisDistance1d(dist=dist, mean=mu, std=sigma)
    return result
