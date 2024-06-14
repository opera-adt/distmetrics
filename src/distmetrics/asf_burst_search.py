from warnings import warn

import asf_search as asf
import geopandas as gpd
import pandas as pd
from rasterio.crs import CRS
from shapely.geometry import shape


def get_asf_rtc_burst_ts(burst_id: str) -> gpd.GeoDataFrame:
    # make sure JPL syntax is transformed to asf syntax
    burst_id_asf = burst_id.upper().replace('-', '_')
    resp = asf.search(
        operaBurstID=[burst_id_asf],
        processingLevel='RTC',
        polarization=['VV', 'VH'],
    )
    if not resp:
        raise warn('No results - please check burst id and availability.', category=UserWarning)
        return gpd.GeoDataFrame()

    properties = [r.properties for r in resp]
    geometry = [shape(r.geojson()['geometry']) for r in resp]
    properties_f = [
        {
            'opera_id': p['sceneName'],
            'acq_datetime': pd.to_datetime(p['startTime']),
            'polarization': '+'.join(p['polarization']),
            'url_vh': p['url'],
            'url_vv': (p['url'].replace('_VH.tif', '_VV.tif')),
            'track_number': p['pathNumber'],
        }
        for p in properties
    ]

    df_rtc_ts = gpd.GeoDataFrame(properties_f, geometry=geometry, crs=CRS.from_epsg(4326))
    # Ensure dual polarization
    df_rtc_ts = df_rtc_ts[df_rtc_ts.polarization == 'VV+VH'].reset_index(drop=True)
    df_rtc_ts = df_rtc_ts.sort_values(by='acq_datetime').reset_index(drop=True)

    # Remove duplicates from time series
    df_rtc_ts['dedup_id'] = df_rtc_ts.opera_id.map(lambda id_: '_'.join(id_.split('_')[:5]))
    df_rtc_ts = df_rtc_ts.drop_duplicates(subset=['dedup_id']).reset_index(drop=True)
    df_rtc_ts.drop(columns=['dedup_id'])
    return df_rtc_ts
