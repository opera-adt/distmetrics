import numpy as np
from dem_stitcher.merge import merge_arrays_with_geometadata
from dem_stitcher.rio_tools import reproject_arr_to_match_profile, reproject_profile_to_new_crs
from rasterio.crs import CRS

from distmetrics.nd_tools import generate_dilated_exterior_nodata_mask, get_distance_from_mask


def _most_common(lst: list[CRS]) -> CRS:
    return max(set(lst), key=lst.count)


def most_common_crs(profiles: list[dict]) -> CRS:
    return _most_common([profile['crs'] for profile in profiles])


def merge_with_weighted_overlap(
    arrs: list[np.ndarray],
    profiles: list[dict],
    target_crs: CRS | None = None,
    exterior_mask_dilation: int = 0,
    use_distance_weighting_from_exterior_mask: bool = True,
) -> tuple[np.ndarray, dict]:
    if target_crs is None:
        target_crs = most_common_crs(profiles)
    crs_resampling_required = [p['crs'] != target_crs for p in profiles]
    profiles_target = [
        reproject_profile_to_new_crs(p, target_crs) if crs_resampling_required[k] else p
        for (k, p) in enumerate(profiles)
    ]
    arrs_r = [
        reproject_arr_to_match_profile(arr, profiles[k], profiles_target[k]) if crs_resampling_required[k] else arr
        for (k, arr) in enumerate(arrs)
    ]
    exterior_masks = [
        generate_dilated_exterior_nodata_mask(arr, nodata_val=p['nodata'], n_iterations=exterior_mask_dilation)
        for (arr, p) in zip(arrs_r, profiles_target)
    ]

    if use_distance_weighting_from_exterior_mask:
        weights = [get_distance_from_mask(mask) for mask in exterior_masks]
    else:
        weights = [np.ones_like(arr) for arr in arrs_r]

    arrs_r_weighted = [arr * weight for (arr, weight) in zip(arrs_r, weights)]

    # masking
    nodata_target = profiles_target[0]['nodata']
    for arr, weight, mask in zip(arrs_r_weighted, weights, exterior_masks):
        arr[mask == 1] = nodata_target
        weight[mask == 1] = nodata_target

    arr_weighted_sum_merged, profile_merged = merge_arrays_with_geometadata(
        arrs_r_weighted, profiles_target, method='sum'
    )
    total_weights_merged, _ = merge_arrays_with_geometadata(weights, profiles_target, method='sum')

    arrs_merged = arr_weighted_sum_merged / (total_weights_merged + 1e-10)

    # Make 2d array instead of BIP 3d array with single band in first dimension
    arrs_merged = arrs_merged[0, ...]

    return arrs_merged, profile_merged
