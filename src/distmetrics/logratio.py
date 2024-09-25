import numpy as np
from pydantic import BaseModel, model_validator

from .mahalanobis import get_spatiotemporal_mu_1d


class LogRatioDecrease(BaseModel):
    dist: np.ndarray
    pre_img_agg: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_shape(cls, values: dict) -> dict:
        dist = values.dist
        agg = values.pre_img_agg
        if dist.shape != agg.shape:
            raise ValueError('the preimage must be aggreated to the same shape as the metric')


def compute_log_ratio_raw(
    pre_arrs: list, post_arr: np.ndarray, window_size: int = 1, qual_stat_for_pre_imgs: str = 'median'
) -> np.ndarray:
    """
    Compute the log ratio between pre-images and a single post image. Assumes single channel image.

    Parameters
    ----------
    pre_arrs : list
        List of np.ndarrays
    post_arr : np.ndarray
        Single np.ndarray of the post-scene
    window_size : int, optional
        Can compute statistics in small spatial window, by default 1
    qual_stat_for_pre_imgs : str, optional
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
    if qual_stat_for_pre_imgs not in ['mean', 'median']:
        ValueError('qual stat needs to be "mean" or "median"')
    if len(pre_arrs) == 0:
        raise ValueError('No prearrs specified')

    pre_stack = np.stack(pre_arrs, axis=0)
    if window_size == 1:
        if qual_stat_for_pre_imgs == 'median':
            stat = np.nanmedian
        if qual_stat_for_pre_imgs == 'mean':
            stat == np.nanmean
        pre_img_agg = stat(pre_stack, axis=0)
    elif (window_size % 2) == 0:
        raise ValueError('Window size must be odd integer')
    else:
        if qual_stat_for_pre_imgs == 'median':
            raise NotImplementedError('Spatial windows are not available for median')
        if qual_stat_for_pre_imgs == 'mean':
            pre_img_agg = get_spatiotemporal_mu_1d(pre_stack, window_size=window_size)
    diff = 10 * np.log10(post_arr) - 10 * np.log10(pre_img_agg)
    return diff, pre_img_agg


def compute_log_ratio_decrease_metric(
    pre_arrs: list,
    post_arr: np.ndarray,
    window_size: int = 1,
    qual_stat_for_pre_imgs: str = 'median',
    max_db_decrease: float = 10.0,
) -> LogRatioDecrease:
    diff, pre_img_agg = compute_log_ratio_raw(
        pre_arrs, post_arr, window_size=window_size, qual_stat_for_pre_imgs=qual_stat_for_pre_imgs
    )
    dist = np.clip(-diff, 0, max_db_decrease)
    dist_ob = LogRatioDecrease(dist=dist, pre_img_agg=pre_img_agg)
    return dist_ob
