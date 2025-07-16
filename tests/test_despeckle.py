from pathlib import Path

import numpy as np
import pytest
import rasterio

from distmetrics.despeckle import despeckle_rtc_arrs_with_tv


@pytest.mark.parametrize('interp_method', ['none', 'nearest', 'bilinear'])
def test_despeckle_with_different_params(
    cropped_vh_data_dir: Path,
    interp_method: str,
) -> None:
    """Test despeckling with different interpolation methods."""
    all_paths = list(cropped_vh_data_dir.glob('*.tif'))

    def open_arr(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            X = ds.read(1)
        return X

    vh_arrs = [open_arr(p) for p in all_paths]

    dspkl_arrs = despeckle_rtc_arrs_with_tv(
        vh_arrs,
        n_jobs=2,  # Use fewer jobs for testing
        interp_method=interp_method,
        preserve_exterior_mask=True,
        n_iter_bilinear=10,
        fill_value=None,
        reg_param=5,  # Default value
        tqdm_enabled=False,  # Disable progress bar for cleaner test output
    )

    # Basic assertions
    assert len(dspkl_arrs) == len(vh_arrs)

    # Check that all arrays have the same shape as input
    for i, (original, despeckled) in enumerate(zip(vh_arrs, dspkl_arrs)):
        assert despeckled.shape == original.shape, f'Shape mismatch for array {i}'

        # Check that despeckled arrays are not all NaN
        assert not np.all(np.isnan(despeckled)), f'All NaN result for array {i} with method {interp_method}'

        # Check that despeckled arrays are within valid range (if not all NaN)
        if not np.all(np.isnan(despeckled)):
            assert np.nanmin(despeckled) >= 1e-7, f'Values below min_valid_value for array {i}'
            assert np.nanmax(despeckled) <= 1.0, f'Values above max_valid_value for array {i}'


def test_invalid_interpolation_method(cropped_vh_data_dir: Path) -> None:
    """Test that invalid interpolation method raises ValueError."""
    all_paths = list(cropped_vh_data_dir.glob('*.tif'))

    def open_arr(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            X = ds.read(1)
        return X

    vh_arrs = [open_arr(p) for p in all_paths]

    with pytest.raises(ValueError, match='Invalid interpolation method'):
        despeckle_rtc_arrs_with_tv(vh_arrs, interp_method='invalid_method', n_jobs=1, tqdm_enabled=False)
