from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from numpy.testing import assert_allclose
from scipy.special import logit
from tqdm import tqdm

from distmetrics.mahalanobis import _transform_pre_arrs
from distmetrics.model_load import ALLOWED_MODELS, control_flow_for_device, load_transformer_model
from distmetrics.tf_metric import (
    estimate_normal_params,
)


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('model_name', ALLOWED_MODELS)
def test_external_model_loading(device: str, model_name: str) -> None:
    """Test loading a model using explicit config and weights paths instead of library token."""
    # Get the model data directory using the same approach as the library
    import distmetrics.model_load

    model_data_dir = Path(distmetrics.model_load.__file__).parent.resolve() / 'model_data'

    # Construct paths to config and weights for the specified model
    model_dir = model_data_dir / model_name
    config_path = model_dir / 'config.json'
    weights_path = model_dir / 'weights.pth'

    # Verify the files exist
    assert config_path.exists(), f'Config file {config_path} does not exist'
    assert weights_path.exists(), f'Weights file {weights_path} does not exist'

    # Load model using library token (reference)
    model_lib = load_transformer_model(lib_model_token=model_name, device=device, model_compilation=False)

    # Load model using explicit paths
    model_external = load_transformer_model(
        lib_model_token='external',
        model_cfg_path=config_path,
        model_wts_path=weights_path,
        device=device,
        model_compilation=False,
    )

    # Verify both models have the same architecture
    assert isinstance(model_lib, type(model_external))
    assert model_lib.num_patches == model_external.num_patches
    assert model_lib.patch_size == model_external.patch_size
    assert model_lib.data_dim == model_external.data_dim
    assert model_lib.max_seq_len == model_external.max_seq_len

    # Verify both models have the same weights by comparing a few key parameters
    lib_params = dict(model_lib.named_parameters())
    ext_params = dict(model_external.named_parameters())

    assert set(lib_params.keys()) == set(ext_params.keys()), 'Models have different parameter names'

    # Compare a subset of parameters to verify they're identical
    for param_name in list(lib_params.keys())[:5]:  # Check first 5 parameters
        assert_allclose(
            lib_params[param_name].detach().cpu().numpy(),
            ext_params[param_name].detach().cpu().numpy(),
            rtol=1e-7,
            err_msg=f'Parameter {param_name} differs between library and external loading',
        )


@pytest.mark.parametrize('device', ['cpu'])
def test_external_model_loading_error_handling(device: str) -> None:
    """Test error handling when loading models with invalid external paths."""
    # Test with non-existent config file
    with pytest.raises(FileNotFoundError):
        load_transformer_model(
            lib_model_token='external',
            model_cfg_path='/path/that/does/not/exist/config.json',
            model_wts_path='/path/that/does/not/exist/weights.pth',
            device=device,
            model_compilation=False,
        )

    # Test with None paths when using external token
    with pytest.raises(ValueError, match='model_wts_path must be provided'):
        load_transformer_model(
            lib_model_token='external', model_cfg_path=None, model_wts_path=None, device=device, model_compilation=False
        )

    # Test with only one path provided
    with pytest.raises(ValueError, match='model_wts_path must be provided'):
        load_transformer_model(
            lib_model_token='external',
            model_cfg_path='/some/path/config.json',
            model_wts_path=None,
            device=device,
            model_compilation=False,
        )


@torch.no_grad()
def estimate_normal_params_as_logits_explicit(
    model: torch.nn.Module,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    stride: int = 4,
    max_nodata_ratio: float = 0.1,
    device: str | None = None,
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

    device = control_flow_for_device(device)

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
    pre_imgs_stack = np.expand_dims(pre_imgs_stack, axis=0)

    # H x W
    H, W = pre_imgs_stack.shape[-2:]

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

            chip = torch.from_numpy(pre_imgs_stack[:, :, :, sy, sx]).to(device)
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


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('model_name', ALLOWED_MODELS)
@pytest.mark.parametrize('model_compilation', [True])
def test_inference(cropped_despeckled_data_dir: Path, device: str, model_name: str, model_compilation: bool) -> None:
    all_paths = list(cropped_despeckled_data_dir.glob('*.tif'))
    vv_paths = [p for p in all_paths if 'VH' in p.name]
    vh_paths = [p for p in all_paths if 'VV' in p.name]
    assert len(vv_paths) == len(vh_paths)

    def open_arr(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)

    vv_arrs = [open_arr(p) for p in vv_paths]
    vh_arrs = [open_arr(p) for p in vh_paths]

    vv_arrs = [logit(a) for a in vv_arrs]
    vh_arrs = [logit(a) for a in vh_arrs]

    model = load_transformer_model(lib_model_token=model_name, device=device, model_compilation=model_compilation)
    pred_means_explicit, pred_sigmas_explicit = estimate_normal_params_as_logits_explicit(
        model, vv_arrs, vh_arrs, stride=2, device=device
    )
    pred_means_stream, pred_sigmas_stream = estimate_normal_params(
        model, vv_arrs, vh_arrs, memory_strategy='low', stride=2, device=device
    )
    pred_means_fold, pred_sigmas_fold = estimate_normal_params(
        model, vv_arrs, vh_arrs, memory_strategy='high', stride=2, device=device
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
