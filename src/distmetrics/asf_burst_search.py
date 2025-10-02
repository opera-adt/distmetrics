from warnings import warn

import asf_search as asf
import geopandas as gpd
import pandas as pd
from rasterio.crs import CRS
from shapely.geometry import shape


def get_url_tif(row: pd.Series, polarization: str = 'crosspol') -> str:
    if polarization not in ['copol', 'crosspol']:
        raise ValueError('polarization not specified correctly')
    urls = [row.url] + row.additionalUrls
    if polarization == 'crosspol':
        valid_urls = [url for url in urls if ('_VH.tif' in url or '_HH.tif' in url)]
    elif polarization == 'copol':
        valid_urls = [url for url in urls if ('_VV.tif' in url or '_HH.tif' in url)]
    if not valid_urls:
        warn(f'No urls found for {row.opera_id} and polarization {polarization}')
        tif_url = ''
    else:
        tif_url = valid_urls[0]
    return tif_url


def get_asf_rtc_burst_ts(burst_id: str, select_polarization: str = 'VV+VH') -> gpd.GeoDataFrame:
    # make sure JPL syntax is transformed to asf syntax
    burst_id_asf = burst_id.upper().replace('-', '_')
    resp = asf.search(
        operaBurstID=[burst_id_asf],
        processingLevel='RTC',
    )
    if not resp:
        warn('No results - please check burst id and availability.', category=UserWarning)
        return gpd.GeoDataFrame()

    properties = [r.properties for r in resp]
    geometry = [shape(r.geojson()['geometry']) for r in resp]
    properties_f = [
        {
            'opera_id': p['sceneName'],
            'acq_datetime': pd.to_datetime(p['startTime']),
            'polarizations': '+'.join(p['polarization']),
            'url': p['url'],
            'additionalUrls': p['additionalUrls'],
            'track_number': p['pathNumber'],
        }
        for p in properties
    ]

    df_rtc_ts = gpd.GeoDataFrame(properties_f, geometry=geometry, crs=CRS.from_epsg(4326))
    df_rtc_ts['url_crosspol'] = df_rtc_ts.apply(get_url_tif, axis=1, polarization='crosspol')
    df_rtc_ts['url_copol'] = df_rtc_ts.apply(get_url_tif, axis=1, polarization='copol')
    df_rtc_ts = df_rtc_ts.dropna(subset=['url_crosspol', 'url_copol'], how='any')
    if select_polarization is not None:
        if select_polarization not in ['VV+VH', 'HH+HV']:
            raise ValueError(f'select_polarization must be "VV+VH" or "HH+HV", not {select_polarization}')
        df_rtc_ts = df_rtc_ts[df_rtc_ts.polarizations == select_polarization].reset_index(drop=True)

    df_rtc_ts.drop(columns=['url', 'additionalUrls'], inplace=True)
    # Ensure dual polarization
    df_rtc_ts = df_rtc_ts.sort_values(by='acq_datetime').reset_index(drop=True)

    # Remove duplicates from time series
    df_rtc_ts['dedup_id'] = df_rtc_ts.opera_id.map(lambda id_: '_'.join(id_.split('_')[:5]))
    df_rtc_ts = df_rtc_ts.drop_duplicates(subset=['dedup_id']).reset_index(drop=True)
    df_rtc_ts.drop(columns=['dedup_id'])
    return df_rtc_ts
