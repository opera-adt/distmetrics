from pathlib import Path

import numpy as np
import rasterio

from distmetrics.despeckle import despeckle_rtc_arrs_with_tv


def test_logit_estimation(cropped_vh_data_dir: Path) -> None:
    all_paths = list(cropped_vh_data_dir.glob('*.tif'))

    def open_arr(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            X = ds.read(1)
        return X

    vh_arrs = [open_arr(p) for p in all_paths]

    dspkl_arrs = despeckle_rtc_arrs_with_tv(vh_arrs, n_jobs=5)
    assert len(dspkl_arrs) == len(vh_arrs)
