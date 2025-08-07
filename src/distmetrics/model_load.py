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

ALLOWED_MODELS = [
    'transformer_original',
    'transformer_optimized',
    'transformer_optimized_fine',
    'transformer_anniversary_trained',
    'transformer_anniversary_trained_optimized',
    'transformer_anniversary_trained_optimized_fine',
]


def compile_model(
    transformer: torch.nn.Module, dtype: str, device: str, batch_size: int, cuda_latest: bool = False
) -> torch.nn.Module:
    """Optimize model for inference using torch.compile or TensorRT."""
    if allow_ops_in_compiled_graph:
        allow_ops_in_compiled_graph()

    if device == 'cuda' and cuda_latest:
        try:
            import torch_tensorrt

            # Get dimensions for TensorRT
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
                        dtype=dtype,
                    )
                ],
                enabled_precisions={dtype},
                truncate_long_and_double=True,
            )
        except ImportError:
            print('torch_tensorrt not available, using standard compilation')
            transformer = torch.compile(transformer, backend='inductor')
    elif device == 'cuda':
        transformer = torch.compile(transformer, backend='inductor')
    else:
        transformer = torch.compile(transformer, mode='max-autotune-no-cudagraphs', dynamic=False)

    return transformer


def load_library_model_config(model_name: str) -> dict:
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'Model name must be one of: {", ".join(ALLOWED_MODELS)}, got {model_name}')

    model_dir = MODEL_DATA / model_name
    config_path = model_dir / 'config.json'

    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory {model_dir} does not exist')
    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} does not exist')

    with config_path.open() as f:
        config = json.load(f)

    return config


def load_weights_from_path(weights_path: Path | str, device: str | None = None) -> dict:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f'Weights file {weights_path} does not exist')

    device = control_flow_for_device(device)
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle both full checkpoints and direct state dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        weights = checkpoint['model_state_dict']
    else:
        weights = checkpoint

    return weights


def load_library_model_weights(model_name: str, device: str | None = None) -> dict:
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'Model name must be one of: {", ".join(ALLOWED_MODELS)}, got {model_name}')

    model_dir = MODEL_DATA / model_name
    weights_path = model_dir / 'weights.pth'

    if not weights_path.exists():
        raise FileNotFoundError(f'Weights file {weights_path} does not exist')

    return load_weights_from_path(weights_path, device)


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
    model_cfg_path: Path | str | dict | None = None,
    model_wts_path: Path | str | dict | None = None,
    device: str | None = None,
    model_compilation: bool = False,
    batch_size: int = 32,
    dtype: str = 'float32',
) -> torch.nn.Module:
    """Load a transformer model from a library or a custom model.

    Parameters
    ----------
    lib_model_token : str, optional
        Name of model directory within model_data/. Must be one of:
        - transformer_original
        - transformer_optimized
        - transformer_optimized_fine
        - transformer_anniversary_trained
    model_cfg_path : Path | str | dict | None, optional
        Path to model config file or dictionary with config parameters. If None, model is loaded from library.
    model_wts_path : Path | str | dict | None, optional
        Path to model weights file or dictionary. If None, model is loaded from library.
    device : str | None, optional
        Device to load model to. If None, device is selected automatically.
    model_compilation : bool, optional
        Whether to compile the model for faster inference.
    batch_size : int, optional
        Batch size to use for model compilation.
    dtype : str, optional
        Data type to use for model. Must be one of:
        - float32
        - float
        - bfloat16

    Returns
    -------
    torch.nn.Module
        Transformer model.
    """
    if lib_model_token not in ['external'] + list(ALLOWED_MODELS):
        raise ValueError(
            f'model_token must be one of {", ".join(["external"] + list(ALLOWED_MODELS))}, got {lib_model_token}'
        )

    if lib_model_token in ALLOWED_MODELS:
        if (model_cfg_path is not None) or (model_wts_path is not None):
            raise ValueError(
                f'model_cfg_path and model_wts_path must be None when lib_model_token is in {ALLOWED_MODELS}'
            )
        model_config = load_library_model_config(lib_model_token)
        weights = load_library_model_weights(lib_model_token, device)
    elif lib_model_token == 'external' and any(x is None for x in [model_cfg_path, model_wts_path]):
        raise ValueError('model_wts_path must be provided when model_cfg_path is provided')
    else:
        weights = load_weights_from_path(model_wts_path, device)
        with Path.open(model_cfg_path) as cfg:
            model_config = json.load(cfg)

    if dtype not in TORCH_DTYPE_MAP.keys():
        raise ValueError(f'dtype must be one of {", ".join(TORCH_DTYPE_MAP.keys())}, got {dtype}')
    torch_dtype = TORCH_DTYPE_MAP[dtype]

    transformer = SpatioTemporalTransformer(**model_config).to(device)
    transformer.load_state_dict(weights)
    transformer = transformer.eval()

    if model_compilation:
        transformer = compile_model(transformer, torch_dtype, device, batch_size)

    return transformer


def get_model_context_length(model_name: str, model_cfg_path: Path | str | dict | None = None) -> int:
    if model_name not in ALLOWED_MODELS + ['external']:
        raise ValueError(f'Model name must be one of: {", ".join(ALLOWED_MODELS + ["external"])}, got {model_name}')

    if model_name in ALLOWED_MODELS:
        model_cfg = load_library_model_config(model_name)
    else:
        with Path.open(model_cfg_path) as cfg:
            model_cfg = json.load(cfg)

    return model_cfg['max_seq_len']
