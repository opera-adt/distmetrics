import json
import math
import platform
from pathlib import Path

import torch
import torch.mps
from einops._torch_specific import allow_ops_in_compiled_graph

from distmetrics.tf_model import SpatioTemporalTransformer


MODEL_DATA = Path(__file__).parent.resolve() / 'model_data'

# Dtype selection
TORCH_DTYPE_MAP = {
    'float32': torch.float32,
    'float': torch.float32,
    'bfloat16': torch.bfloat16,
}

_ALLOWED_MODELS = {
    'transformer_original',
    'transformer_optimized',
    'transformer_optimized_fine',
    'transformer_anniversary_trained',
}


_MODEL = None


def load_library_model(model_name: str) -> tuple[dict, Path]:
    """Load model weights and config from the library directory.

    Parameters
    ----------
    model_name : str
        Name of model directory within model_data/. Must be one of:
        - transformer_original
        - transformer_optimized
        - transformer_optimized_fine
        - transformer_anniversary_trained

    Returns
    -------
    tuple[dict, Path]
        Model config dictionary and path to weights file

    Raises
    ------
    ValueError
        If model_name is not one of the allowed values or if files don't exist
    """
    if model_name not in _ALLOWED_MODELS:
        raise ValueError(f'Model name must be one of: {", ".join(_ALLOWED_MODELS)}, got {model_name}')

    model_dir = MODEL_DATA / model_name
    config_path = model_dir / 'config.json'
    weights_path = model_dir / 'weights.pth'

    if not model_dir.exists():
        raise ValueError(f'Model directory {model_dir} does not exist')
    if not config_path.exists():
        raise ValueError(f'Config file {config_path} does not exist')
    if not weights_path.exists():
        raise ValueError(f'Weights file {weights_path} does not exist')

    with config_path.open() as f:
        config = json.load(f)

    return config, weights_path.resolve()


def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def control_flow_for_device(device: str | None = None) -> str:
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError('device must be one of cpu, cuda, mps')
    return device


def load_transformer_model(
    lib_model_token: str = 'transformer_original',
    model_cfg_path: Path | None = None,
    model_wts_path: Path | None = None,
    device: str | None = None,
    model_compilation: bool = False,
    batch_size: int = 32,
    dtype: str = 'float32',
) -> torch.nn.Module:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if lib_model_token not in ['external'] + list(_ALLOWED_MODELS):
        raise ValueError(
            f'model_token must be one of {", ".join(["external"] + list(_ALLOWED_MODELS))}, got {lib_model_token}'
        )

    if lib_model_token in _ALLOWED_MODELS:
        config, weights_path = load_library_model(lib_model_token)
    else:
        with Path.open(model_cfg_path) as cfg:
            config = json.load(cfg)
        weights_path = model_wts_path

    if dtype not in TORCH_DTYPE_MAP.keys():
        raise ValueError(f'dtype must be one of {", ".join(TORCH_DTYPE_MAP.keys())}, got {dtype}')
    torch_dtype = TORCH_DTYPE_MAP[dtype]

    device = control_flow_for_device(device)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    transformer = SpatioTemporalTransformer(config).to(device)
    transformer.load_state_dict(weights)
    transformer = transformer.eval()

    if model_compilation:
        allow_ops_in_compiled_graph()

        if device == 'cuda':
            import torch_tensorrt

            # Get dimensions
            total_pixels = transformer.num_patches * (transformer.patch_size**2)
            wh = math.isqrt(total_pixels)
            channels = transformer.data_dim // (transformer.patch_size**2)
            expected_dims = (batch_size, transformer.max_seq_len, channels, wh, wh)

            transformer = torch_tensorrt.compile(
                transformer,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1,) + expected_dims[1:],
                        opt_shape=expected_dims,
                        max_shape=expected_dims,
                        dtype=torch_dtype,
                    )
                ],
                enabled_precisions={torch_dtype},  # e.g., {torch.float}, {torch.float16}
                truncate_long_and_double=True,  # Optional: helps prevent type issues
            )

        else:
            transformer = torch.compile(transformer, mode='max-autotune-no-cudagraphs', dynamic=False)

    _MODEL = transformer

    return transformer
