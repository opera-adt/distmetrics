from typing import Optional

import numpy as np
from pydantic import BaseModel, model_validator
from scipy.special import logit
from scipy.stats import norm as normal


class CuSumDist(BaseModel):
    dist: np.ndarray | list
    cusum: Optional[np.ndarray]
    drift: Optional[np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_shape(cls, values: dict) -> dict:
        dist = values.dist
        # Ensure everything has the same shape if it is not specified
        cusum_prev = values.get('cusum_prev', np.zeros(dist.shape))
        drift = values.get('drift', np.zeros(dist.shape))

        if any([dist.shape != arr.shape for arr in [cusum_prev, drift]]):
            raise ValueError('All arrays must have same shape')


def compute_cusum_1d(
    pre_arrs: list[np.ndarray], post_arr: np.ndarray, temporal_drift: str = 'mean', transform_data_to: str = None
):
    """Simple implementation of cumusm.

    Source:
    [1] https://en.wikipedia.org/wiki/CUSUM
    [2] https://gitlab.com/nisar-science-algorithms/ecosystems/forest_disturbance
    [3] https://nisar.jpl.nasa.gov/system/documents/files/26_NISAR_FINAL_9-6-19.pdf

    Parameters
    ----------
    pre_arrs : list[np.ndarray]
        Sequence of pre-images
    post_arr : np.ndarray
        Single post array
    temporal_drift : str, optional
        String that indicates the array by which to compute per pixel residual.
        Can use 'mean', 'median' or 'mean_of_maxmin', by default 'mean'.
    transform_data_to : str, optional
        Can compute cumsum relative to gamma naugt (linear) which is what is
        assumed as input (i.e. using None) - also can transform these arrays to
        logits or dbs, by default None or without transformation.

    Returns
    -------
    CuSumDist :
        Larger distance if the final cumulative sum of residuals is high in either direction

    Raises
    ------
    ValueError
        When drift value is specified incorrectly
    """
    allowable_drift_values = ['mean', 'median', 'mean_of_maxmin']
    if temporal_drift not in allowable_drift_values:
        raise ValueError(f'temporal_drift must be in {", ".join(allowable_drift_values)}')

    pre_arrs = np.stack(pre_arrs, axis=0)

    allowable_transforms = ['logit', 'db']
    if (transform_data_to is not None) and (transform_data_to not in (allowable_transforms)):
        raise ValueError(f'transform_data_to should either be None or in {", ".join(allowable_transforms)}')

    if transform_data_to is not None:
        if transform_data_to == 'db':
            pre_arrs_t = 10 * np.log10(pre_arrs)
            post_arr_t = 10 * np.log10(post_arr)
        elif transform_data_to == 'logit':
            pre_arrs_t = logit(pre_arrs)
            post_arr_t = logit(post_arr)
    else:
        pre_arrs_t = pre_arrs
        post_arr_t = post_arr

    if temporal_drift == 'mean':
        drift = np.nanmean(pre_arrs_t, axis=0)
    elif temporal_drift == 'median':
        drift = np.nanmean(pre_arrs_t, axis=0)
    else:
        drift = (np.nanmax(pre_arrs_t, axis=0) + np.nanmin(pre_arrs, axis=0)) / 2.0

    cusum_prev = np.nansum(pre_arrs_t - drift, axis=0)
    residual_current = post_arr_t - drift

    cusum_current = np.abs(residual_current + cusum_prev)
    dist_ob = CuSumDist(dist=cusum_current, cusum=cusum_prev, drift=drift)
    return dist_ob


def compute_prob_cusum_1d(pre_arrs: list[np.ndarray], post_arr: np.ndarray) -> CuSumDist:
    """This adapted from this blog post:

    [1] https://web.archive.org/web/20240817203837/https://sarem-seitz.com/posts/probabilistic-cusum-for-change-point-detection/  # noqa: E501

    The final distance is (1 - prob), where "prob" is the probability of the magnitude of the cusum of the
    residuals of the normalized time-series at the current time step (i.e. represented in post_arr)


    Parameters
    ----------
    pre_arrs : list[np.ndarray]
    post_arr : np.ndarray

    Returns
    -------
    CuSumDist :
        Will only have a distance given by 1 - prob, as defined in the summary above.
    """
    pre_arrs_logit = logit(np.stack(pre_arrs, axis=0))
    arrs_logit = logit(np.stack(pre_arrs + [post_arr], axis=0))

    mean = np.nanmean(pre_arrs_logit, axis=0)
    std = np.nanstd(pre_arrs_logit, axis=0)

    n_temporal_samples = np.nansum(~np.isnan(arrs_logit), axis=0)
    arrs_norm = (arrs_logit - mean) / (std * np.sqrt(n_temporal_samples))
    cusum_residual_normalized = np.nansum(arrs_norm, axis=0)

    p_cdf = normal(loc=0, scale=1).cdf(np.abs(cusum_residual_normalized))
    prob = 2 * (1 - p_cdf)

    return CuSumDist(dist=(1 - prob))
