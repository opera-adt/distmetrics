from pathlib import Path
from typing import Callable

import einops
import numpy as np
import torch
import torch.mps
import torch.nn as nn
from scipy.special import logit
from tqdm import tqdm

from .mahalanobis import _transform_pre_arrs

WEIGHTS_DIR = Path(__file__).parent.resolve() / 'model_weights'
TRANSFORMER_WEIGHTS_PATH = WEIGHTS_DIR / 'transformer.pth'


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, model_config):
        super(SpatioTemporalTransformer, self).__init__()

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
    weights = torch.load(TRANSFORMER_WEIGHTS_PATH, map_location=device)
    transformer.load_state_dict(weights)
    return transformer


def estimate_normal_params_as_logits(
    model,
    pre_imgs_vv: list[np.ndarray],
    pre_imgs_vh: list[np.ndarray],
    stride=4,
) -> tuple[np.ndarray]:
    """
    Assumes images are in gamma naught and despeckled with TV

    Mean and sigma are in logit units
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
    with torch.no_grad():  # tells torch it doesn't need to keep track of gradients
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
                if (chip_mask).sum().item() / chip_mask.nelement() <= 0.1:
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
    return pred_means, pred_logvars


def get_1d_transformer_zscore(model,
                              pre_imgs_vv: list[np.ndarray],
                              pre_imgs_vh: list[np.ndarray],
                              post_arr_vv: np.ndarray,
                              post_arr_vh: np.ndarray,
                              stride=4,
                              agg: str | Callable = 'max') -> np.ndarray:
    if isinstance(agg, str):
        if agg not in ['max', 'min']:
            raise NotImplementedError('We expect max/min as strings')
        elif agg == 'min':
            agg = np.min
        else:
            agg = np.max

        post_arr_logit_s = logit(np.stack([post_arr_vv, post_arr_vh], axis=0))
        mu, log_sigma_sq = estimate_normal_params_as_logits(model, pre_imgs_vv, pre_imgs_vh, stride=stride)
        sigma = np.sqrt(np.exp(log_sigma_sq))
        z_score_dual = np.abs(post_arr_logit_s - mu) / sigma
        z_score = agg(z_score_dual, axis=0)
        return z_score
