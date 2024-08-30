import numpy as np

from .mahalanobis import get_spatiotemporal_mu_1d


def compute_log_ratio(
    pre_arrs: list, post_arr: np.ndarray, spatial_window_size: int = 1, qual_stat: str = 'median'
) -> np.ndarray:
    """
    Compute the log ratio between pre-images and a single post image. Assumes single channel image.

    Parameters
    ----------
    pre_arrs : list
        List of np.ndarrays
    post_arr : np.ndarray
        Single np.ndarray of the post-scene
    spatial_window_size : int, optional
        Can compute statistics in small spatial window, by default 1
    qual_stat : str, optional
        Which statistic to aggregate preimages by, needs to be either "mean" or "median", by default 'median'

    Returns
    -------
    np.ndarray
        The Log Ratio

    Raises
    ------
    ValueError
        If qual_stat is not specified correctly
    """
    if qual_stat not in ['mean', 'median']:
        ValueError('qualt stat needs to be "mean" or "median"')
    pre_stack = np.stack(pre_arrs, axis=0)
    if spatial_window_size == 1:
        if qual_stat == 'median':
            stat = np.nanmedian
        if qual_stat == 'mean':
            stat == np.nanmean
        pre_stat = stat(pre_stack, axis=0)
    elif (spatial_window_size % 2) == 0:
        raise ValueError('Window size must be odd integer')
    else:
        if qual_stat == 'median':
            raise NotImplementedError('Spatial windows are not available for median')
        if qual_stat == 'mean':
            pre_stat = get_spatiotemporal_mu_1d(pre_stack, window_size=spatial_window_size)
    diff = 10 * np.log10(post_arr) - 10 * np.log10(pre_stat)
    return diff
