import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.draw import polygon

from distmetrics.nd_tools import get_exterior_nodata_mask


def rotated_rectangle(
    mask_shape: tuple[int, int], center: tuple[float, float], rect_size: tuple[float, float], angle: float
) -> np.ndarray:
    """
    Create a binary NumPy array with a rotated rectangle.

    Parameters
    ----------
    mask_shape : tuple of int
        Tuple (height, width) specifying the size of the binary mask.
    center : tuple of float
        Tuple (cy, cx) specifying the center of the rectangle.
    rect_size : tuple of float
        Tuple (width, height) specifying the dimensions of the rectangle.
    angle : float
        Rotation angle in degrees (counterclockwise).

    Returns
    -------
    numpy.ndarray
        Binary NumPy array with the rotated rectangle.
    """
    cy, cx = center
    width, height = mask_shape
    angle = np.deg2rad(angle)  # Convert angle to radians

    # Compute the four corner points of the rectangle (relative to center)
    dx = width / 2
    dy = height / 2

    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])

    # Rotation matrix
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Apply rotation and translation
    rotated_corners = (corners @ rot_matrix.T) + np.array([cx, cy])

    # Get row and column indices
    rr, cc = polygon(rotated_corners[:, 1], rotated_corners[:, 0], mask_shape)

    # Create binary mask
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[rr, cc] = 1

    return mask


@pytest.mark.parametrize(
    'mask_shape, center, rect_size, angle, nodata_val',
    [
        ((100, 130), (50, 65), (40, 40), 45, np.nan),
        ((130, 100), (65, 50), (100, 50), 45, np.nan),
        ((100, 100), (50, 50), (100, 50), 0, np.nan),
        ((100, 130), (50, 65), (40, 40), 45, 255),
        ((130, 100), (65, 50), (100, 50), 45, 255),
        ((100, 100), (50, 50), (100, 50), 0, 255),
        ((100, 100), (50, 50), (110, 110), 0, 255),
    ],
)
def test_get_exterior_nodata_mask(
    mask_shape: tuple[int, int],
    center: tuple[int, int],
    rect_size: tuple[int, int],
    angle: float,
    nodata_val: float | int,
) -> None:
    # create a rotated rectangle
    binary_mask = rotated_rectangle(mask_shape, center, rect_size, angle)
    nodata_mask_no_hole = 1 - binary_mask
    nodata_mask_with_hole = nodata_mask_no_hole.copy()
    nodata_mask_with_hole[center[0] - 5 : center[0] + 5, center[1] - 5 : center[1] + 5] = 1
    if np.isnan(nodata_val):
        img = nodata_mask_with_hole.astype(np.float32)
        img[nodata_mask_with_hole == 1] = np.nan
    elif nodata_val == 255:
        img = nodata_mask_no_hole.astype(np.uint8)
        img[nodata_mask_with_hole == 1] = 255
    else:
        raise ValueError(f'invalid nodata value: {nodata_val}')
    exterior_mask = get_exterior_nodata_mask(img, nodata_val=nodata_val)
    assert_array_equal(exterior_mask, nodata_mask_no_hole)
