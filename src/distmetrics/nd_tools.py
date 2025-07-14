import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt
from scipy.ndimage import label as labeler


def nn_interpolate(arr: np.ndarray, preserve_exterior_mask: bool = True) -> np.ndarray:
    """Interpolate nan values in a 2D array using nearest neighbor interpolation.

    Parameters
    ----------
        data (array): A 2D array containing the data to fill.  Void elements
            should have values of np.nan.

    Returns
    -------
        filled (array): The filled data.

    """
    arr_filled = arr.copy()
    ind = nd.distance_transform_edt(np.isnan(arr_filled), return_distances=False, return_indices=True)
    arr_filled = arr_filled[tuple(ind)]
    if preserve_exterior_mask:
        print('preserving exterior mask')

        print(np.sum(np.isnan(arr)))
        exterior_mask = get_exterior_nodata_mask(arr)
        arr_filled[exterior_mask.astype(bool)] = np.nan
    return arr_filled


def iterative_linear_interpolate(
    arr: np.ndarray, max_iter: int = 10, preserve_exterior_mask: bool = True
) -> np.ndarray:
    """Interpolate nan values in a 2D array via iterative (convolutional) linear interpolation.

    Parameters
    ----------
        arr (array): A 2D array containing the data to fill.  Void elements
            should have values of np.nan.

    Returns
    -------
        filled (array): The filled data.
    """
    arr_filled = arr.copy()
    nan_mask = np.isnan(arr)

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)

    for _ in range(max_iter):
        if not np.any(nan_mask):
            break

        # Set NaNs to zero temporarily
        temp = np.where(nan_mask, 0, arr_filled)

        # Count valid neighbors
        valid = (~nan_mask).astype(float)
        weight = convolve(valid, kernel, mode='constant', cval=0)
        smoothed = convolve(temp, kernel, mode='constant', cval=0)

        # Avoid division by zero
        with np.errstate(invalid='ignore'):
            interp_values = smoothed / weight

        # Fill only the nan locations
        updates = (nan_mask) & (weight > 0)
        arr_filled[updates] = interp_values[updates]
        nan_mask = np.isnan(arr_filled)

    if preserve_exterior_mask:
        exterior_mask = get_exterior_nodata_mask(arr)
        arr_filled[exterior_mask.astype(bool)] = np.nan

    return arr_filled


def get_exterior_nodata_mask(image: np.ndarray, nodata_val: float | int = np.nan) -> np.ndarray:
    if len(image.shape) != 2:
        raise ValueError('can only get exterior mask for 2d array')

    def identify_nodata(val: float | int | np.ndarray) -> np.ndarray | float | int:
        if np.isnan(nodata_val):
            return np.isnan(val)
        else:
            return val == nodata_val

    nodata_mask = identify_nodata(image)
    component_labels, _ = labeler(nodata_mask)
    edge_mask = np.zeros_like(nodata_mask, dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    exterior_labels = list(np.unique(component_labels[edge_mask & nodata_mask]))
    exterior_mask = np.isin(component_labels, exterior_labels).astype(np.uint8)
    return exterior_mask.astype(np.uint8)


def generate_dilated_exterior_nodata_mask(
    image: np.ndarray, nodata_val: float | int = np.nan, n_iterations: int = 10
) -> np.ndarray:
    exterior_mask = get_exterior_nodata_mask(image, nodata_val)
    if n_iterations > 0:
        exterior_mask = binary_dilation(exterior_mask, iterations=n_iterations).astype(np.uint8)
    return exterior_mask


def get_distance_from_mask(mask: np.ndarray, mask_val: int = 1) -> np.ndarray:
    # assume we want distance from 1
    if mask.dtype != np.uint8:
        raise ValueError('mask must be uint8')
    dist = distance_transform_edt(mask != 1, return_distances=True, return_indices=False)
    return dist
