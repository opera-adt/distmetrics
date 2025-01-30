import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import label as labeler


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
