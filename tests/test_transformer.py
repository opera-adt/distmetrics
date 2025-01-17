from pathlib import Path

import numpy as np
import rasterio
import torch
from numpy.testing import assert_allclose
from scipy.special import logit
from tqdm import tqdm

from distmetrics.transformer import (
    _transform_pre_arrs,
    estimate_normal_params_of_logits,
    get_device,
    load_transformer_model,
)


@torch.no_grad()
def estimate_normal_params_as_logits_explicit(
    model: torch.nn.Module,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    stride: int = 4,
    max_nodata_ratio: float = 0.1,
) -> tuple[np.ndarray]:
    """
    Estimate the mean and sigma of the normal distribution of the logits of the input images.

    Mean and sigma are in logit units.

    This is the slower application due to the for loop. However, there is additional
    control flow around the application of the transformer:

       - we always have a 16 x 16 patch as an input chip for the model
       - we do not apply the model if the ratio of masked pixels in a chip exceeds max_nodata_ratio
    """
    P = 16
    assert stride <= P
    assert stride > 0
    assert (max_nodata_ratio < 1) and (max_nodata_ratio > 0)

    device = get_device()

    # stack to T x 2 x H x W
    pre_imgs_stack = _transform_pre_arrs(pre_imgs_vv, pre_imgs_vh)
    pre_imgs_stack = pre_imgs_stack.astype('float32')

    # Mask
    mask_stack = np.isnan(pre_imgs_stack)
    # Remove T x 2 dims
    mask_spatial = torch.from_numpy(np.any(mask_stack, axis=(0, 1)))
    assert len(mask_spatial.shape) == 2, 'spatial mask should be 2d'

    # Logit transformation
    pre_imgs_stack[mask_stack] = 1e-7
    pre_imgs_logit = logit(pre_imgs_stack)
    pre_imgs_logit = np.expand_dims(pre_imgs_logit, axis=0)

    # H x W
    H, W = pre_imgs_logit.shape[-2:]

    # Initalize Output arrays
    pred_means = torch.zeros((2, H, W), device=device)
    pred_logvars = torch.zeros_like(pred_means)
    count = torch.zeros_like(pred_means)

    # Sliding window
    n_patches_y = int(np.floor((H - P) / stride) + 1)
    n_patches_x = int(np.floor((W - P) / stride) + 1)

    model.eval()  # account for dropout, etc
    for i in tqdm(range(n_patches_y), desc='Rows Traversed'):
        for j in range(n_patches_x):
            if i == (n_patches_y - 1):
                sy = slice(H - P, H)
            else:
                sy = slice(i * stride, i * stride + P)

            if j == (n_patches_x - 1):
                sx = slice(W - P, W)
            else:
                sx = slice(j * stride, j * stride + P)

            chip = torch.from_numpy(pre_imgs_logit[:, :, :, sy, sx]).to(device)
            chip_mask = mask_spatial[sy, sx]
            # Only apply model if nodata mask is smaller than X%
            if (chip_mask).sum().item() / chip_mask.nelement() <= max_nodata_ratio:
                chip_mean, chip_logvar = model(chip)
                chip_mean, chip_logvar = chip_mean[0, ...], chip_logvar[0, ...]
                pred_means[:, sy, sx] += chip_mean.reshape((2, P, P))
                pred_logvars[:, sy, sx] += chip_logvar.reshape((2, P, P))
                count[:, sy, sx] += 1
            else:
                continue

    pred_means = (pred_means / count).squeeze()
    pred_logvars = (pred_logvars / count).squeeze()

    M_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[M_3d] = torch.nan
    pred_logvars[M_3d] = torch.nan

    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    return pred_means, pred_sigmas


def test_logit_estimation(test_data_dir: Path) -> None:
    all_paths = list(test_data_dir.glob('*.tif'))
    vv_paths = [p for p in all_paths if 'VH' in p.name]
    vh_paths = [p for p in all_paths if 'VV' in p.name]
    assert len(vv_paths) == len(vh_paths)

    def open_arr(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)

    vv_arrs = [open_arr(p) for p in vv_paths]
    vh_arrs = [open_arr(p) for p in vh_paths]

    model = load_transformer_model()
    pred_means_explicit, pred_sigmas_explicit = estimate_normal_params_as_logits_explicit(
        model, vv_arrs, vh_arrs, stride=2
    )
    pred_means_stream, pred_sigmas_stream = estimate_normal_params_of_logits(
        model, vv_arrs, vh_arrs, memory_strategy='low', stride=2
    )
    pred_means_fold, pred_sigmas_fold = estimate_normal_params_of_logits(
        model, vv_arrs, vh_arrs, memory_strategy='high', stride=2
    )

    edge_buffer = 16
    sy_buffer = np.s_[edge_buffer:-edge_buffer]
    sx_buffer = np.s_[edge_buffer:-edge_buffer]
    assert_allclose(
        pred_means_explicit[:, sy_buffer, sx_buffer],
        pred_means_stream[:, sy_buffer, sx_buffer],
        atol=1e-5,
    )
    assert_allclose(
        pred_sigmas_explicit[:, sy_buffer, sx_buffer],
        pred_sigmas_stream[:, sy_buffer, sx_buffer],
        atol=1e-5,
    )
    assert_allclose(
        pred_means_explicit[:, sy_buffer, sx_buffer],
        pred_means_fold[:, sy_buffer, sx_buffer],
        atol=1e-5,
    )
    assert_allclose(
        pred_sigmas_explicit[:, sy_buffer, sx_buffer],
        pred_sigmas_fold[:, sy_buffer, sx_buffer],
        atol=1e-5,
    )
