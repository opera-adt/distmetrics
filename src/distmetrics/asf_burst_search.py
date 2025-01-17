from warnings import warn

import asf_search as asf
import geopandas as gpd
import pandas as pd
from rasterio.crs import CRS
from shapely.geometry import shape


def get_url_tif(row: pd.Series, polarization: str = 'vh') -> str:
    if polarization not in ['vv', 'vh']:
        raise ValueError('polarization not specified correctly')
    urls = [row.url] + row.additionalUrls
    pol_token = '_VH.tif' if polarization == 'vh' else '_VV.tif'
    valid_urls = [url for url in urls if pol_token in url]
    if not valid_urls:
        raise ValueError('No urls found')
    tif_url = valid_urls[0]
    return tif_url


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
            'url': p['url'],
            'additionalUrls': p['additionalUrls'],
            'track_number': p['pathNumber'],
        }
        for p in properties
    ]

    df_rtc_ts = gpd.GeoDataFrame(properties_f, geometry=geometry, crs=CRS.from_epsg(4326))
    df_rtc_ts['url_vh'] = df_rtc_ts.apply(get_url_tif, axis=1, polarization='vh')
    df_rtc_ts['url_vv'] = df_rtc_ts.apply(get_url_tif, axis=1, polarization='vv')
    df_rtc_ts.drop(columns=['url', 'additionalUrls'], inplace=True)
    # Ensure dual polarization
    df_rtc_ts = df_rtc_ts[df_rtc_ts.polarization == 'VV+VH'].reset_index(drop=True)
    df_rtc_ts = df_rtc_ts.sort_values(by='acq_datetime').reset_index(drop=True)

    # Remove duplicates from time series
    df_rtc_ts['dedup_id'] = df_rtc_ts.opera_id.map(lambda id_: '_'.join(id_.split('_')[:5]))
    df_rtc_ts = df_rtc_ts.drop_duplicates(subset=['dedup_id']).reset_index(drop=True)
    df_rtc_ts.drop(columns=['dedup_id'])
    return df_rtc_ts
