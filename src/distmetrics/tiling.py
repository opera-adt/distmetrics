import math
from collections.abc import Generator

import numpy as np


def get_tile_generator(array: np.ndarray, win_size: int, step: int) -> tuple[Generator[np.ndarray, None, None], dict]:
    assert len(array.shape) == 5, 'array must be 5D'
    _, _, c, h, w = array.shape

    n_steps_h = math.ceil((h - win_size) / step) + 1
    n_steps_w = math.ceil((w - win_size) / step) + 1

    target_h = (n_steps_h - 1) * step + win_size
    target_w = (n_steps_w - 1) * step + win_size

    pad_h = target_h - h
    pad_w = target_w - w

    array_padded = np.pad(array, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')

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

    def generator() -> Generator[np.ndarray, None, None]:
        for i in range(n_steps_h):
            for j in range(n_steps_w):
                h_start = i * step
                w_start = j * step
                h_end = h_start + win_size
                w_end = w_start + win_size

                tile = array_padded[..., h_start:h_end, w_start:w_end]
                yield tile

    return generator(), info


def reconstruct_from_generator(tile_gen: Generator[np.ndarray, None, None], info: dict) -> np.ndarray:
    padded_shape = (info['padded_h'], info['padded_w'])
    orig_shape = (info['orig_h'], info['orig_w'])
    win_h, win_w = info['win_h'], info['win_w']
    step_h, step_w = info['step_h'], info['step_w']

    output_canvas = None
    count_canvas = None

    for i in range(info['n_steps_h']):
        for j in range(info['n_steps_w']):
            # Get the processed tile from your function
            try:
                processed_tile = next(tile_gen)
            except StopIteration:
                break

            if output_canvas is None:
                b, t, c, _, _ = processed_tile.shape
                full_shape = (b, t, c, padded_shape[0], padded_shape[1])

                output_canvas = np.zeros(full_shape, dtype=processed_tile.dtype)
                count_canvas = np.zeros(full_shape, dtype=np.float32)

                # Create a weight mask for a single tile (usually all ones)
                tile_weights = np.ones((b, t, c, win_h, win_w), dtype=np.float32)

            h_start = i * step_h
            h_end = h_start + win_h
            w_start = j * step_w
            w_end = w_start + win_w

            output_canvas[..., h_start:h_end, w_start:w_end] += processed_tile
            count_canvas[..., h_start:h_end, w_start:w_end] += tile_weights

    count_canvas[count_canvas == 0] = 1.0

    reconstructed = output_canvas / count_canvas

    final_output = reconstructed[..., : orig_shape[0], : orig_shape[1]]

    return final_output
