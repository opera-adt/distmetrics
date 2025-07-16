from collections.abc import Callable
from typing import Union

import numpy as np
import torch
import torch.mps
from pydantic import BaseModel, ConfigDict, model_validator

from distmetrics.tf_inference import estimate_normal_params


class DiagMahalanobisDistance2d(BaseModel):
    dist: Union[np.ndarray, list]  # noqa: UP007
    mean: np.ndarray
    std: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def check_shapes(cls, values: dict) -> dict:
        """Check the covariance matrix is of the form 2 x 2 x H x W."""
        d = values.dist
        mu = values.mean
        sigma = values.std

        if mu.shape != sigma.shape:
            raise ValueError('mean and std must have the same shape')
        if d.shape != sigma.shape[1:]:
            raise ValueError('The mean/std must have same spatial dimensions as dist')


def compute_transformer_zscore(
    model: torch.nn.Module,
    pre_imgs_copol: list[np.ndarray],
    pre_imgs_crosspol: list[np.ndarray],
    post_arr_copol: np.ndarray,
    post_arr_crosspol: np.ndarray,
    stride: int = 4,
    batch_size: int = 32,
    tqdm_enabled: bool = True,
    agg: str | Callable = 'max',
    memory_strategy: str = 'high',
    device: str | None = None,
) -> DiagMahalanobisDistance2d:
    """
    Compute the transformer z-score.

    Assumes that VV and VH are independent so returns mean, std for each polarizaiton separately (as learned by
    model). The mean and std are returned as 2 x H x W matrices. The two zscores are aggregated by the callable agg.
    Agg defaults to maximum z-score of each polarization.

    Warning: mean and std are in logits! That is logit(gamma_naught)!
    """
    if isinstance(agg, str):
        if agg not in ['max', 'min']:
            raise NotImplementedError('We expect max/min as strings')
        elif agg == 'min':
            agg = np.min
        else:
            agg = np.max

    mu, sigma = estimate_normal_params(
        model,
        pre_imgs_copol,
        pre_imgs_crosspol,
        stride=stride,
        batch_size=batch_size,
        tqdm_enabled=tqdm_enabled,
        memory_strategy=memory_strategy,
        device=device,
    )

    post_arr_s = np.stack([post_arr_copol, post_arr_crosspol], axis=0)
    z_score_dual = np.abs(post_arr_s - mu) / sigma
    z_score = agg(z_score_dual, axis=0)
    m_dist = DiagMahalanobisDistance2d(dist=z_score, mean=mu, std=sigma)
    return m_dist
