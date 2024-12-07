from collections.abc import Callable, Generator
from pathlib import Path

import einops
import numpy as np
import torch
import torch.mps
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, model_validator
from scipy.special import logit
from tqdm import tqdm

from .mahalanobis import _transform_pre_arrs


def unfolding_stream(
    image_st: torch.Tensor, kernel_size: int, stride: int, batch_size: int
) -> Generator[torch.Tensor, None, None]:
    T, C, H, W = image_st.shape

    patches = []
    slices = []

    n_patches_y = int(np.floor((H - kernel_size) / stride) + 1)
    n_patches_x = int(np.floor((W - kernel_size) / stride) + 1)

    for i in range(0, n_patches_y):
        for j in range(0, n_patches_x):
            if i == (n_patches_y - 1):
                s_y = slice(H - kernel_size, H)
            else:
                s_y = slice(i * stride, i * stride + kernel_size)

            if j == (n_patches_x - 1):
                s_x = slice(W - kernel_size, W)
            else:
                s_x = slice(j * stride, j * stride + kernel_size)

            patch = image_st[..., s_y, s_x]
            patches.append(patch)
            slices.append((s_y, s_x))

            # Yield patches in batches
            if len(patches) == batch_size:
                yield torch.stack(patches, dim=0), slices
                patches = []
                slices = []

    if patches:
        yield torch.stack(patches, dim=0), slices


WEIGHTS_DIR = Path(__file__).parent.resolve() / 'model_weights'
TRANSFORMER_WEIGHTS_PATH = WEIGHTS_DIR / 'transformer.pth'


class DiagMahalanobisDistance2d(BaseModel):
    dist: np.ndarray | list
    mean: np.ndarray
    std: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_shapes(cls, values: dict) -> dict:
        """Check that our covariance matrix is of the form 2 x 2 x H x W"""
        d = values.dist
        mu = values.mean
        sigma = values.std

        if mu.shape != sigma.shape:
            raise ValueError('mean and std must have the same shape')
        if d.shape != sigma.shape[1:]:
            raise ValueError('The mean/std must have same spatial dimensions as dist')


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['num_encoder_layers']
        self.dim_feedforward = model_config['dim_feedforward']
        self.max_seq_len = model_config['max_seq_len']
        self.dropout = model_config['dropout']
        self.activation = model_config['activation']

        self.num_patches = model_config['num_patches']
        self.patch_size = model_config['patch_size']
        self.data_dim = model_config['data_dim']

        self.embedding = nn.Linear(self.data_dim, self.d_model)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.d_model))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, 1, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)

        self.mean_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),  # reuse dim feedforward here
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )

        self.logvar_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),  # reuse dim feedforward here
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )

    def num_parameters(self):
        """Count the number of trainable parameters in the model."""
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = 0
            for p in self.parameters():
                count = 1
                for s in p.size():
                    count *= s
                self._num_parameters += count

        return self._num_parameters

    def forward(self, src):
        batch_size, seq_len, channels, height, width = src.shape

        assert self.num_patches == (height * width) / (self.patch_size**2)

        src = einops.rearrange(
            src, 'b t c (h ph) (w pw) -> b t (h w) (c ph pw)', ph=self.patch_size, pw=self.patch_size
        )  # batch, seq_len, num_patches, data_dim

        src = (
            self.embedding(src) + self.spatial_pos_embed + self.temporal_pos_embed[:, :seq_len, :, :]
        )  # batch, seq_len, num_patches, d_model

        src = src.view(
            batch_size, seq_len * self.num_patches, self.d_model
        )  # transformer expects (batch_size, sequence, dmodel)

        # Pass through the transformer encoder with causal masking
        # mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        # mask = mask.repeat(self.num_patches, self.num_patches)

        output = self.transformer_encoder(src)  # , mask=mask, is_causal=True) # is_causal makes this masked attention

        mean = self.mean_out(output)  # batchsize, seq_len*num_patches, data_dim
        logvar = self.logvar_out(output)  # batchsize, seq_len*num_patches, 2*data_dim

        mean = mean.view(batch_size, seq_len, self.num_patches, self.data_dim)  # undo previous operation
        logvar = logvar.view(batch_size, seq_len, self.num_patches, self.data_dim)

        # reshape to be the same shape as input batch_size, seq len, channels, height, width
        mean = einops.rearrange(
            mean,
            'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            c=channels,
            h=height // self.patch_size,
            w=width // self.patch_size,
        )

        # reshape so for each pixel we output 4 numbers (ie each entry of cov matrix)
        logvar = einops.rearrange(
            logvar,
            'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            c=channels,
            h=height // self.patch_size,
            w=width // self.patch_size,
        )

        return mean[:, -1, ...], logvar[:, -1, ...]


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available:
        device = 'mps'
    else:
        device = 'cpu'
    return device


def load_trained_transformer_model():
    device = get_device()
    config = {
        'type': 'transformer (space and time pos encoding)',
        'patch_size': 8,
        'num_patches': 4,
        'data_dim': 128,  # 2 * patch_size * patch_size
        'd_model': 256,
        'nhead': 4,
        'num_encoder_layers': 2,
        'dim_feedforward': 512,
        'max_seq_len': 10,
        'dropout': 0.2,
        'activation': 'relu',
    }
    transformer = SpatioTemporalTransformer(config).to(device)
    weights = torch.load(TRANSFORMER_WEIGHTS_PATH, map_location=device, weights_only=True)
    transformer.load_state_dict(weights)
    return transformer


@torch.no_grad()
def estimate_normal_params_as_logits_explicit(
    model, pre_imgs_vv: list[np.ndarray], pre_imgs_vh: list[np.ndarray], stride=4, max_nodata_ratio: float = 0.1
) -> tuple[np.ndarray]:
    """
    Assumes images are in gamma naught and despeckled with TV

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


@torch.no_grad()
def estimate_normal_params_as_logits_stream(
    model,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    stride=2,
    batch_size=32,
    max_nodata_ratio: float = 0.1,
    tqdm_enabled: bool = True,
) -> tuple[np.ndarray]:
    """

    Parameters
    ----------
    model : _type_
        transformer with chip (or patch size) 16
    pre_imgs_vv : list[np.ndarray]
    pre_imgs_vh : list[np.ndarray]
        _description_
    stride : int, optional
        Should be between 1 and 16, by default 2.
    stride : int, optional
        How to batch chips.

    Returns
    -------
    tuple[np.ndarray]
        pred_mean, pred_sigma (as logits)

    Notes
    -----
    - Applied model to images where mask values are assigned 1e-7
    """
    P = 16
    assert stride <= P
    assert stride > 0

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
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit).to(device)

    # C x H x W
    C, H, W = pre_imgs_logit.shape[-3:]

    # Sliding window
    n_patches_y = int(np.floor((H - P) / stride) + 1)
    n_patches_x = int(np.floor((W - P) / stride) + 1)
    n_patches = n_patches_y * n_patches_x

    n_batches = n_patches // batch_size + 1

    target_shape = (C, H, W)
    count = torch.zeros(*target_shape).to(device)
    pred_means = torch.zeros(*target_shape).to(device)
    pred_logvars = torch.zeros(*target_shape).to(device)

    unfold_gen = unfolding_stream(pre_imgs_stack_t, P, stride, batch_size)

    model.eval()
    for patch_batch, slices in tqdm(
        unfold_gen, total=n_batches, desc='Chips Traversed', mininterval=2, disable=(not tqdm_enabled)
    ):
        chip_mean, chip_logvar = model(patch_batch)
        for k, (sy, sx) in enumerate(slices):
            chip_mask = mask_spatial[sy, sx]
            if (chip_mask).sum().item() / chip_mask.nelement() <= max_nodata_ratio:
                pred_means[:, sy, sx] += chip_mean[k, ...]
                pred_logvars[:, sy, sx] += chip_logvar[k, ...]
                count[:, sy, sx] += 1
    pred_means /= count
    pred_logvars /= count

    M_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[M_3d] = torch.nan
    pred_logvars[M_3d] = torch.nan

    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    return pred_means, pred_sigmas


@torch.no_grad()
def estimate_normal_params_as_logits_folding(
    model,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    stride=2,
    batch_size=32,
    tqdm_enabled: bool = True,
) -> tuple[np.ndarray]:
    """
    Parameters
    ----------
    model : _type_
        transformer with chip (or patch size) 16
    pre_imgs_vv : list[np.ndarray]
    pre_imgs_vh : list[np.ndarray]
        _description_
    stride : int, optional
        Should be between 1 and 16, by default 2.
    stride : int, optional
        How to batch chips.

    Returns
    -------
    tuple[np.ndarray]
        pred_mean, pred_sigma (as logits)

    Notes
    -----
    - May apply model to chips of slightly smaller size around boundary
    - Applied model to images where mask values are assigned 1e-7
    """
    P = 16
    assert stride <= P
    assert stride > 0

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

    # H x W
    H, W = pre_imgs_logit.shape[-2:]
    T = pre_imgs_logit.shape[0]
    C = pre_imgs_logit.shape[1]

    # Sliding window
    n_patches_y = int(np.floor((H - P) / stride) + 1)
    n_patches_x = int(np.floor((W - P) / stride) + 1)
    n_patches = n_patches_y * n_patches_x

    # Shape (T x 2 x H x W)
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit)
    # T x (2 * P**2) x n_patches
    patches = F.unfold(pre_imgs_stack_t, kernel_size=P, stride=stride)
    # n_patches x T x (C * P**2)
    patches = patches.permute(2, 0, 1).to(device)
    # n_patches x T x C x P**2
    patches = patches.view(n_patches, T, C, P**2)

    n_batches = n_patches // batch_size + 1

    target_chip_shape = (n_patches, C, P, P)
    pred_means_p = torch.zeros(*target_chip_shape).to(device)
    pred_logvars_p = torch.zeros(*target_chip_shape).to(device)

    model.eval()
    for i in tqdm(range(n_batches), desc='Chips Traversed', mininterval=2, disable=(not tqdm_enabled)):
        # change last dimension from P**2 to P, P; use -1 because won't always have batch_size as 0th dimension
        batch_s = slice(batch_size * i, batch_size * (i + 1))
        patch_batch = patches[batch_s, ...].view(-1, T, C, P, P)
        chip_mean, chip_logvar = model(patch_batch)
        pred_means_p[batch_s, ...] += chip_mean
        pred_logvars_p[batch_s, ...] += chip_logvar
    del patches
    torch.cuda.empty_cache()

    # n_patches x C x P x P -->  (C * P**2) x n_patches
    pred_logvars_p_reshaped = pred_logvars_p.view(n_patches, C * P**2).permute(1, 0)
    pred_logvars = F.fold(pred_logvars_p_reshaped, output_size=(H, W), kernel_size=P, stride=stride)
    del pred_logvars_p

    pred_means_p_reshaped = pred_means_p.view(n_patches, C * P**2).permute(1, 0)
    pred_means = F.fold(pred_means_p_reshaped, output_size=(H, W), kernel_size=P, stride=stride)
    del pred_means_p_reshaped

    input_ones = torch.ones(1, H, W, dtype=torch.float32).to(device)
    count_patches = F.unfold(input_ones, kernel_size=P, stride=stride)
    count = F.fold(count_patches, output_size=(H, W), kernel_size=P, stride=stride)
    del count_patches
    torch.cuda.empty_cache()

    pred_means /= count
    pred_logvars /= count

    mask_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[mask_3d] = torch.nan
    pred_logvars[mask_3d] = torch.nan

    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    return pred_means, pred_sigmas


def get_unfolded_view(X: torch.Tensor, kernel_size, stride):
    unfolded_height = X.unfold(-2, kernel_size, stride)
    patches = unfolded_height.unfold(-2, kernel_size, stride)
    # T x C x n_patches_y x n_patches_x x kernel_size x kernel_size
    # ->  n_patches_y x n_patches_x x T x C x kernel_size x kernel_size
    return patches.permute(2, 3, 0, 1, 4, 5)


@torch.no_grad()
def estimate_normal_params_as_logits_folding_alt(
    model,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    stride=2,
    batch_size=32,
    tqdm_enabled: bool = True,
) -> tuple[np.ndarray]:
    """
    Parameters
    ----------
    model : _type_
        transformer with chip (or patch size) 16
    pre_imgs_vv : list[np.ndarray]
    pre_imgs_vh : list[np.ndarray]
        _description_
    stride : int, optional
        Should be between 1 and 16, by default 2.
    stride : int, optional
        How to batch chips.

    Returns
    -------
    tuple[np.ndarray]
        pred_mean, pred_sigma (as logits)

    Notes
    -----
    - May apply model to chips of slightly smaller size around boundary
    - Applied model to images where mask values are assigned 1e-7
    """
    P = 16
    assert stride <= P
    assert stride > 0

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

    # H x W
    H, W = pre_imgs_logit.shape[-2:]
    # T = pre_imgs_logit.shape[0]
    C = pre_imgs_logit.shape[1]

    # Sliding window
    n_patches_y = int(np.floor((H - P) / stride) + 1)
    n_patches_x = int(np.floor((W - P) / stride) + 1)
    n_patches = n_patches_y * n_patches_x

    # Shape (T x 2 x H x W)
    pre_imgs_stack_t = torch.from_numpy(pre_imgs_logit).to(device)

    # Shape: (T x 2 x N_Patches_y x N_patches_x x P x P)
    pre_patches = get_unfolded_view(pre_imgs_stack_t, P, stride)

    target_unfolded_shape = (n_patches, C, P, P)
    pred_means_p = torch.zeros(*target_unfolded_shape).to(device)
    pred_logvars_p = torch.zeros(*target_unfolded_shape).to(device)

    n_batches_x = int(np.ceil(n_patches_x / batch_size))
    current_patch_start_idx = 0

    model.eval()
    for i in tqdm(range(n_patches_y), desc='Rows traversed', mininterval=2, disable=(not tqdm_enabled)):
        for j in range(n_batches_x):
            start = batch_size * j
            stop = min(n_patches_x, batch_size * (j + 1))
            ts_slice = slice(start, stop)
            batch_size_current = stop - start

            patch_batch = pre_patches[i, ts_slice, ...]
            chip_mean, chip_logvar = model(patch_batch)

            patch_idx = slice(current_patch_start_idx, current_patch_start_idx + batch_size_current)
            pred_means_p[patch_idx, ...] += chip_mean
            pred_logvars_p[patch_idx, ...] += chip_logvar

            current_patch_start_idx += batch_size_current
    del pre_imgs_stack_t
    torch.cuda.empty_cache()

    # Put all reconstruction on the CPU
    input_ones = torch.ones(1, H, W, dtype=torch.float32).to('cpu')
    count_patches = F.unfold(input_ones, kernel_size=P, stride=stride)
    count = F.fold(count_patches, output_size=(H, W), kernel_size=P, stride=stride)

    # n_patches x C x P x P -->  (C * P**2) x n_patches
    pred_logvars_p_reshaped = pred_logvars_p.view(n_patches, C * P**2).permute(1, 0)
    pred_logvars = F.fold(pred_logvars_p_reshaped.to('cpu'), output_size=(H, W), kernel_size=P, stride=stride)

    pred_means_p_reshaped = pred_means_p.view(n_patches, C * P**2).permute(1, 0)
    pred_means = F.fold(pred_means_p_reshaped.to('cpu'), output_size=(H, W), kernel_size=P, stride=stride)

    pred_means /= count
    pred_logvars /= count

    mask_3d = mask_spatial.unsqueeze(dim=0).expand(pred_means.shape)
    pred_means[mask_3d] = torch.nan
    pred_logvars[mask_3d] = torch.nan

    pred_means = pred_means.cpu().numpy().squeeze()
    pred_logvars = pred_logvars.cpu().numpy().squeeze()
    pred_sigmas = np.sqrt(np.exp(pred_logvars))
    return pred_means, pred_sigmas


def compute_transformer_zscore(
    model,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    post_arr_vv: np.ndarray,
    post_arr_vh: np.ndarray,
    stride=4,
    batch_size=32,
    tqdm_enabled: bool = True,
    agg: str | Callable = 'max',
    memory_strategy: str = 'high',
) -> DiagMahalanobisDistance2d:
    """Assumes that VV and VH are independent so returns mean, std for each polarizaiton separately (as learned by
    model). The mean and std are returned as 2 x H x W matrices. The two zscores are aggregated by the callable agg.
    Agg defaults to maximum z-score of each polarization.

    Warning: mean and std are in logits! That is logit(gamma_naught)!
    """
    if memory_strategy not in ['high', 'low']:
        raise ValueError('memory strategy must be high or low')
    if isinstance(agg, str):
        if agg not in ['max', 'min']:
            raise NotImplementedError('We expect max/min as strings')
        elif agg == 'min':
            agg = np.min
        else:
            agg = np.max

        post_arr_logit_s = logit(np.stack([post_arr_vv, post_arr_vh], axis=0))
        compute_logits = (
            estimate_normal_params_as_logits_folding
            if memory_strategy == 'high'
            else estimate_normal_params_as_logits_folding_alt
        )
        mu, sigma = compute_logits(
            model, pre_imgs_vv, pre_imgs_vh, stride=stride, batch_size=batch_size, tqdm_enabled=tqdm_enabled
        )
        z_score_dual = np.abs(post_arr_logit_s - mu) / sigma
        z_score = agg(z_score_dual, axis=0)
        m_dist = DiagMahalanobisDistance2d(dist=z_score, mean=mu, std=sigma)
        return m_dist
