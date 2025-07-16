import multiprocessing as mp
from functools import partial

import numpy as np
from skimage.restoration import denoise_tv_bregman
from tqdm import tqdm

from distmetrics.nd_tools import iterative_linear_interpolate, nn_interpolate


# Use spawn for multiprocessing
mp.set_start_method('spawn', force=True)


def interpolate_arr(
    arr: np.ndarray,
    interp_method: str,
    preserve_exterior_mask: bool = True,
    n_iter_bilinear: int = 10,
    fill_value: float | None = None,
) -> np.ndarray:
    if interp_method == 'nearest':
        interp_func = partial(nn_interpolate, preserve_exterior_mask=preserve_exterior_mask)
    elif interp_method == 'bilinear':
        interp_func = partial(
            iterative_linear_interpolate, max_iter=n_iter_bilinear, preserve_exterior_mask=preserve_exterior_mask
        )
    else:
        raise ValueError(f'Invalid interpolation method: {interp_method}')

    interp_arr = arr.copy()
    interp_arr = interp_func(interp_arr)
    if fill_value is not None:
        interp_arr[np.isnan(interp_arr)] = fill_value
    else:
        interp_arr[np.isnan(interp_arr)] = np.nan
    return interp_arr


def despeckle_one_rtc_arr_with_tv(
    X: np.ndarray,
    reg_param: float = 5,
    max_valid_value: float = 1.0,
    min_valid_value: float = 1e-7,
    interp_method: str = 'none',
    preserve_exterior_mask: bool = True,
    n_iter_bilinear: int = 10,
    fill_value: float | None = None,
) -> np.ndarray:
    X_c = np.clip(X, min_valid_value, max_valid_value)
    if interp_method not in ['none', 'nearest', 'bilinear']:
        raise ValueError(f'Invalid interpolation method: {interp_method}')
    if interp_method != 'none':
        X_c = interpolate_arr(
            X_c,
            interp_method,
            preserve_exterior_mask=preserve_exterior_mask,
            n_iter_bilinear=n_iter_bilinear,
            fill_value=fill_value,
        )
    # Done after interpolation
    nodata_mask = np.isnan(X_c)

    # Despeckle in dB
    X_db = 10 * np.log10(X_c, out=np.full(X_c.shape, np.nan), where=(~np.isnan(X_c)))
    # Use nearest valid value for despeckling
    if nodata_mask.sum() > 0:
        # X_db = interpolate_arr(X_db, 'nearest', preserve_exterior_mask=False, fill_value=None)
        X_db[nodata_mask] = -23
    weight = 1.0 / reg_param
    X_db_dspkl = denoise_tv_bregman(X_db, weight=weight, isotropic=True, eps=1e-3)
    X_dspkl = np.power(10, X_db_dspkl / 10.0)

    # Preserve nodata mask
    X_dspkl[nodata_mask] = np.nan
    X_dspkl = np.clip(X_dspkl, min_valid_value, max_valid_value)
    return X_dspkl


def _despeckle_worker(args: tuple[np.ndarray, dict]) -> np.ndarray:
    """Worker function for multiprocessing that unpacks arguments and calls despeckle_one_rtc_arr_with_tv."""
    arr, kwargs = args
    return despeckle_one_rtc_arr_with_tv(arr, **kwargs)


def despeckle_rtc_arrs_with_tv(
    arrs: list[np.ndarray],
    reg_param: float = 5,
    max_valid_value: float = 1.0,
    min_valid_value: float = 1e-7,
    interp_method: str = 'none',
    preserve_exterior_mask: bool = True,
    n_iter_bilinear: int = 10,
    fill_value: float | None = None,
    n_jobs: int = 10,
    tqdm_enabled: bool = True,
) -> list[np.ndarray]:
    # Prepare arguments for each array
    kwargs = {
        'reg_param': reg_param,
        'max_valid_value': max_valid_value,
        'min_valid_value': min_valid_value,
        'interp_method': interp_method,
        'preserve_exterior_mask': preserve_exterior_mask,
        'n_iter_bilinear': n_iter_bilinear,
        'fill_value': fill_value,
    }

    # Create argument tuples for multiprocessing
    args_list = [(arr, kwargs) for arr in arrs]

    # Use multiprocessing Pool
    with mp.Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(_despeckle_worker, args_list),
                total=len(arrs),
                desc='Despeckling',
                dynamic_ncols=True,
                disable=not tqdm_enabled,
            )
        )

    return results
