import numpy as np
from dem_stitcher.merge import merge_arrays_with_geometadata
from dem_stitcher.rio_tools import reproject_arr_to_match_profile, reproject_profile_to_new_crs
from rasterio.crs import CRS

from distmetrics.nd_tools import generate_dilated_exterior_nodata_mask, get_distance_from_mask


def _most_common(lst: list[CRS]) -> CRS:
    return max(set(lst), key=lst.count)


def most_common_crs(profiles: list[dict]) -> CRS:
    return _most_common([profile['crs'] for profile in profiles])


def reproject_arrays_to_target_crs(
    arrs: list[np.ndarray], profiles: list[dict], target_crs: CRS | None = None, resampling_method: str = 'bilinear'
) -> tuple[list[np.ndarray], list[dict]]:
    """Reproject arrays to target CRS (if necessary).

    Parameters
    ----------
    arrs : list[np.ndarray]
        List of arrays to reproject
    profiles : list[dict]
        List of profiles of arrays
    target_crs : CRS
        Target CRS
    resampling_method : str, optional
        Resampling method (see rasterio.enums.Resampling), by default 'bilinear'

    Returns
    -------
    tuple[list[np.ndarray], list[dict]]
        Reprojected arrays and profiles
    """
    if target_crs is None:
        target_crs = most_common_crs(profiles)
    crs_resampling_required = [p['crs'] != target_crs for p in profiles]
    profiles_target = [
        reproject_profile_to_new_crs(p, target_crs) if crs_resampling_required[k] else p
        for (k, p) in enumerate(profiles)
    ]
    arrs_r = [
        reproject_arr_to_match_profile(arr, profiles[k], profiles_target[k], resampling=resampling_method)
        if crs_resampling_required[k]
        else arr
        for (k, arr) in enumerate(arrs)
    ]
    return arrs_r, profiles_target


def merge_with_weighted_overlap(
    arrs: list[np.ndarray],
    profiles: list[dict],
    target_crs: CRS | None = None,
    exterior_mask_dilation: int = 0,
    distance_weight_exponent: float = 1.0,
    use_distance_weighting_from_exterior_mask: bool = True,
) -> tuple[np.ndarray, dict]:
    """Merge (float) arrays with geometadata using average over their overlap or distance from exterior nodata mask.

    Parameters
    ----------
    arrs : list[np.ndarray]
        List of arrays to merge (possibly with different CRS)
    profiles : list[dict]
        Associated rasterio profiles of arrays
    target_crs : CRS | None, optional
        Target CRS, if None, use most common CRS amoung profiles provided, by default None
    exterior_mask_dilation : int, optional
        Exterior mask is the nodata mask intersects the edges of the array, by default 0
    distance_weight_exponent : float, optional
        If using distance weighting the exponent to weight distances by, by default 1.0
    use_distance_weighting_from_exterior_mask : bool, optional
        Whether to use distance from exterior mask for weighting, by default True

    Returns
    -------
    tuple[np.ndarray, dict]
        Merged array and its profile
    """
    arrs_r, profiles_target = reproject_arrays_to_target_crs(
        arrs, profiles, target_crs=target_crs, resampling_method='bilinear'
    )

    exterior_masks = [
        generate_dilated_exterior_nodata_mask(arr, nodata_val=p['nodata'], n_iterations=exterior_mask_dilation)
        for (arr, p) in zip(arrs_r, profiles_target)
    ]

    if use_distance_weighting_from_exterior_mask:
        weights = [get_distance_from_mask(mask) ** distance_weight_exponent for mask in exterior_masks]
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


def merge_categorical_arrays(
    arrs: list[np.ndarray],
    profiles: list[dict],
    target_crs: CRS | None = None,
    exterior_mask_dilation: int = 0,
    merge_method: str = 'mode',
) -> tuple[np.ndarray, dict]:
    arrs_r, profiles_target = reproject_arrays_to_target_crs(
        arrs, profiles, target_crs=target_crs, resampling_method='nearest'
    )

    exterior_masks = [
        generate_dilated_exterior_nodata_mask(arr, nodata_val=p['nodata'], n_iterations=exterior_mask_dilation)
        for (arr, p) in zip(arrs_r, profiles_target)
    ]

    # masking
    nodata_target = profiles_target[0]['nodata']
    for arr, mask in zip(arrs_r, exterior_masks):
        arr[mask == 1] = nodata_target

    arr_merged, profile_merged = merge_arrays_with_geometadata(arrs_r, profiles_target, method=merge_method)

    return arr_merged, profile_merged
