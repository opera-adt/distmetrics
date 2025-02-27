from pathlib import Path

import rasterio
from numpy.testing import assert_allclose

from distmetrics.rio_tools import merge_categorical_arrays, open_one_ds


def test_merge_categorical_arrays(
    categorical_merge_input_data: list[Path], categorical_merge_output_data: Path
) -> None:
    """Test merging categorical arrays."""
    # read the input data
    input_data = [open_one_ds(path) for path in categorical_merge_input_data]
    arrs, profiles = zip(*input_data)

    # merge the input data
    merged_array, merged_profile = merge_categorical_arrays(
        arrs,
        profiles,
        merge_method='min',
        # Different rasterio/gdal versions may treat nodata slightly differently
        exterior_mask_dilation=10,
        # we are merging 2 datasets so need to specify the target crs as it can be random depending on which is chosen
        target_crs=profiles[1]['crs'],
    )
    merged_array = merged_array[0, ...]
    with rasterio.open(categorical_merge_output_data) as ds:
        merged_array_expected = ds.read(1)
        merged_profile_expected = ds.profile

    assert_allclose(merged_array, merged_array_expected)
    assert merged_profile['crs'] == merged_profile_expected['crs']
    assert merged_profile['transform'] == merged_profile_expected['transform']
