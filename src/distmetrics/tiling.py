import math
from collections.abc import Generator

import numpy as np
from tqdm import tqdm


def get_tile_generator(
    array: np.ndarray, win_size: int, step: int
) -> tuple[Generator[tuple[np.ndarray, tuple], None, None], dict]:
    assert len(array.shape) == 4, 'array must be 4D'
    _, _, h, w = array.shape

    n_steps_h = math.ceil((h - win_size) / step) + 1
    n_steps_w = math.ceil((w - win_size) / step) + 1

    target_h = (n_steps_h - 1) * step + win_size
    target_w = (n_steps_w - 1) * step + win_size

    info = {
        'orig_h': h,
        'orig_w': w,
        'padded_h': target_h,
        'padded_w': target_w,
        'win_h': win_size,
        'win_w': win_size,
        'step_h': step,
        'step_w': step,
        'n_steps_h': n_steps_h,
        'n_steps_w': n_steps_w,
        'dtype': array.dtype,
    }

    def generator() -> Generator[tuple[np.ndarray, tuple], None, None]:
        total = n_steps_h * n_steps_w
        with tqdm(total=total, desc='Processing tiles') as pbar:
            for i in range(n_steps_h):
                for j in range(n_steps_w):
                    h_start = i * step
                    w_start = j * step
                    h_end = min(h_start + win_size, h)
                    w_end = min(w_start + win_size, w)

                    # Extract tile from original array
                    tile = array[..., h_start:h_end, w_start:w_end]

                    # Pad only this tile if needed (at boundaries)
                    actual_h, actual_w = tile.shape[-2:]
                    if actual_h < win_size or actual_w < win_size:
                        pad_h = win_size - actual_h
                        pad_w = win_size - actual_w
                        tile = np.pad(tile, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='edge')

                    pbar.update(1)
                    # Yield tile and its coordinates (i, j indices)
                    yield tile, (i, j)

    return generator(), info


def reconstruct_from_generator(
    tile_gen: Generator[tuple[np.ndarray, np.ndarray, tuple], None, None], info: dict
) -> np.ndarray:
    padded_shape = (info['padded_h'], info['padded_w'])
    orig_shape = (info['orig_h'], info['orig_w'])
    win_h, win_w = info['win_h'], info['win_w']
    step_h, step_w = info['step_h'], info['step_w']

    means_acc = None
    var_acc = None
    count_acc = None

    # Iterate through all tiles
    for mean_tile, var_tile, (i, j) in tile_gen:
        if means_acc is None:
            c, _, _ = mean_tile.shape
            out_shape = (c, padded_shape[0], padded_shape[1])

            means_acc = np.zeros(out_shape, dtype=mean_tile.dtype)
            var_acc = np.zeros(out_shape, dtype=mean_tile.dtype)
            count_acc = np.zeros(out_shape, dtype=np.float32)

            # Create a weight mask for a single tile (usually all ones)
            tile_weights = np.ones((c, win_h, win_w), dtype=np.float32)

        h_start = i * step_h
        h_end = h_start + win_h
        w_start = j * step_w
        w_end = w_start + win_w

        means_acc[..., h_start:h_end, w_start:w_end] += mean_tile
        var_acc[..., h_start:h_end, w_start:w_end] += var_tile
        count_acc[..., h_start:h_end, w_start:w_end] += tile_weights

    count_acc[count_acc == 0] = 1.0

    mean_out = means_acc / count_acc
    var_out = var_acc / count_acc

    mean_out = mean_out[..., : orig_shape[0], : orig_shape[1]]
    var_out = var_out[..., : orig_shape[0], : orig_shape[1]]

    return mean_out, var_out
