import einops
import torch
import torch.mps
import torch.nn as nn


class TransformerEncoderWithMask(nn.TransformerEncoder):
    """
    Drop-in replacement for nn.TransformerEncoder that supports per-batch attn_mask. Used to mask out "one way attention" of queries for NaNs
    """
    def forward(self, src, attn_mask=None, src_key_padding_mask=None):
        # src: (B, L, d_model)
        output = src
        for mod in self.layers:
            attn_output, _ = mod.self_attn(
                output, output, output,
                attn_mask=attn_mask,                    # (B*H, L, L) or (L, L)
                key_padding_mask=src_key_padding_mask,  # (B, L) True = ignore key
                need_weights=False
            )
            output = output + mod.dropout1(attn_output)
            output = mod.norm1(output)

            src2 = mod.linear2(mod.dropout(mod.activation(mod.linear1(output))))
            output = output + mod.dropout2(src2)
            output = mod.norm2(output)

        if self.norm is not None:
            output = self.norm(output)
        return output


class MLPTimeEmbedding(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, acq_dts_float):
        """
        Args:
            acq_dts_float: Tensor of shape (B, T) — float days since 2014-01-01
        Returns:
            time_emb: Tensor of shape (B, T, out_dim)
        """
        # Add feature dimension
        t = acq_dts_float.unsqueeze(-1)  # (B, T, 1)
        return self.project(t)            # (B, T, out_dim)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        dropout: float,
        activation: str,
        patch_size: int,
        data_dim: int,
        input_size: int = 16,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.activation = activation

        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2
        self.patch_size = patch_size
        self.data_dim = data_dim

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

    def num_parameters(self) -> float:
        """Count the number of trainable parameters in the model."""
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = 0
            for p in self.parameters():
                count = 1
                for s in p.size():
                    count *= s
                self._num_parameters += count

        return self._num_parameters

    def forward(self, img_baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels, height, width = img_baseline.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f'seq_len must be less than or equal to max_seq_len, got {seq_len} and {self.max_seq_len}')
        assert self.num_patches == (height * width) / (self.patch_size**2)

        img_baseline = einops.rearrange(
            img_baseline, 'b t c (h ph) (w pw) -> b t (h w) (c ph pw)', ph=self.patch_size, pw=self.patch_size
        )  # batch, seq_len, num_patches, data_dim

        img_baseline = (
            self.embedding(img_baseline)
            + self.spatial_pos_embed
            # changed self.temporal_pos_embed[:, :seq_len, :, :]
            # to self.temporal_pos_embed[:, (self.max_seq_len-seq_len):, :, :] to ensure last pre-image always has
            # correct index
            + self.temporal_pos_embed[:, (self.max_seq_len - seq_len) :, :, :]
        )  # batch, seq_len, num_patches, d_model

        img_baseline = img_baseline.view(
            batch_size, seq_len * self.num_patches, self.d_model
        )  # transformer expects (batch_size, sequence, dmodel)

        output = self.transformer_encoder(img_baseline)

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


class SpatioTemporalTransformerRedux(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()

        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['num_encoder_layers']
        self.dim_feedforward = model_config['dim_feedforward']
        self.max_seq_len = model_config['max_seq_len']
        self.dropout = model_config['dropout']
        self.activation = model_config['activation']
        self.input_size = model_config['input_size']
        self.patch_size = model_config['patch_size']
        self.num_patches = int((self.input_size / self.patch_size) ** 2)
        self.data_dim = model_config['data_dim']

        # Learnable tokens - initialize with small random values instead of zeros
        self.nan_token = nn.Parameter(torch.randn(2) * 0.02)  # Small random initialization
        self.pad_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)  # PAD embedding in d_model space

        self.embedding = nn.Linear(self.data_dim, self.d_model)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.d_model))

        # Replace positional temporal embedding with MLP-based one
        self.temporal_embedding = MLPTimeEmbedding(out_dim=self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoderWithMask(encoder_layer, self.num_encoder_layers)

        self.mean_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )
        self.logvar_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )

    def num_parameters(self) -> float:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _neg_large_for_dtype(dtype: torch.dtype) -> float:
        # Use more conservative negative values to prevent overflow
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4  # More conservative for mixed precision
        return -1e6  # More conservative than -1e9

    def replace_nans_only(self, x):
        """Replace NaNs with learned nan_token and clamp to prevent extreme values."""
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            token = self.nan_token.view(1, 1, -1, 1, 1)  # (1,1,C,1,1) broadcasts to (B,T,C,H,W)
            # Clamp token values to prevent extreme values
            token = torch.clamp(token, min=-10.0, max=10.0)
            x = torch.where(nan_mask, token, x)
        # Additional safety: clamp the entire tensor to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        return x

    def forward(self, img_baseline: torch.Tensor, acq_dts_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = img_baseline.shape
        device = img_baseline.device
        dtype = img_baseline.dtype

        #Masks
        frame_pad_mask = (img_baseline == -9999.0).all(dim=(2, 3, 4))         # (B, T)
        nan_mask_pixels = torch.isnan(img_baseline)                           # (B, T, C, H, W)

        #Replace Nans
        x = self.replace_nans_only(img_baseline)

        #Reshape
        x = einops.rearrange(
            x, 'b t c (h ph) (w pw) -> b t (h w) (c ph pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        #Reshape NaN mask and padding mask
        patch_nan_mask = einops.rearrange(
            nan_mask_pixels,
            'b t c (h ph) (w pw) -> b t (h w) (c ph pw)',
            ph=self.patch_size, pw=self.patch_size
        ).all(dim=-1)  # True if entire patch was NaN
        patch_pad_mask = frame_pad_mask.unsqueeze(-1).expand(-1, -1, self.num_patches)  # (B, T, P)

        L = T * self.num_patches
        seq_nan_mask = patch_nan_mask.reshape(B, L)  # (B, L)
        seq_pad_mask = patch_pad_mask.reshape(B, L)  # (B, L)

        # Generate temporal embeddings from acquisition dates
        # Replace any NaN values in acquisition dates with 0
        acq_dts_clamped = torch.where(torch.isnan(acq_dts_float), torch.zeros_like(acq_dts_float), acq_dts_float)
        temporal_emb = self.temporal_embedding(acq_dts_clamped)  # (B, T, d_model)
        temporal_emb = temporal_emb.unsqueeze(2).expand(-1, -1, self.num_patches, -1)  # (B, T, P, d_model)

        #Define inputs
        x = self.embedding(x) + self.spatial_pos_embed + temporal_emb
        x = x.view(B, L, self.d_model)

        # Insert learnable pad embedding at PAD positions -- helps w stability
        if seq_pad_mask.any():
            x = x.masked_scatter(
                seq_pad_mask.unsqueeze(-1),
                self.pad_embed.expand(B, L, self.d_model).masked_select(seq_pad_mask.unsqueeze(-1))
            )

        #Define manual attention mask for NaNs
        neg_large = self._neg_large_for_dtype(dtype)
        attn_mask = torch.zeros(B, L, L, device=device, dtype=dtype)

        if seq_nan_mask.any():
            attn_mask = torch.where(seq_nan_mask.unsqueeze(-1).expand(B, L, L), torch.tensor(neg_large, dtype=dtype, device=device), attn_mask)
        if seq_pad_mask.any():
            attn_mask = torch.where(seq_pad_mask.unsqueeze(-1).expand(B, L, L), torch.tensor(neg_large, dtype=dtype, device=device), attn_mask)

        # ensure diagonal 0 for masked rows
        diag = attn_mask.diagonal(dim1=-2, dim2=-1)
        diag.masked_fill_(seq_nan_mask | seq_pad_mask, 0.0)

        # Expand for MultiheadAttention
        Hh = self.transformer_encoder.layers[0].self_attn.num_heads
        attn_mask = attn_mask.unsqueeze(1).expand(B, Hh, L, L).reshape(B * Hh, L, L).contiguous()

        src_key_padding_mask = seq_pad_mask  # (B, L) True = ignore PAD keys

        output = self.transformer_encoder(x, attn_mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

        # Add numerical stability checks
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf detected in transformer output, applying emergency clipping")
            output = torch.clamp(output, min=-100.0, max=100.0)
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)

        mean = self.mean_out(output)
        logvar = self.logvar_out(output)

        # Clamp outputs to prevent extreme values
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)  # Prevent extremely large variances

        mean = mean.view(B, T, self.num_patches, self.data_dim)
        logvar = logvar.view(B, T, self.num_patches, self.data_dim)

        #reshape
        mean = einops.rearrange(
            mean, 'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size,
            c=C, h=(H // self.patch_size), w=(W // self.patch_size)
        )
        logvar = einops.rearrange(
            logvar, 'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size,
            c=C, h=(H // self.patch_size), w=(W // self.patch_size)
        )

        # Return only final timestep
        return mean[:, -1, ...], logvar[:, -1, ...]
