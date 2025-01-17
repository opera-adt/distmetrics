import concurrent.futures
import io

import backoff
import numpy as np
import requests
from rasterio.io import MemoryFile
from requests.exceptions import HTTPError
from tqdm import tqdm


@backoff.on_exception(backoff.expo, (HTTPError, ConnectionError), max_tries=10, max_time=60, jitter=backoff.full_jitter)
def read_bytes(
    url: str,
) -> bytes:
    resp = requests.get(url)
    data = io.BytesIO(resp.content)
    return data


def read_one_asf(url: str) -> tuple[np.ndarray, dict]:
    img_bytes = read_bytes(url)
    with MemoryFile(img_bytes, filename=url.split('/')[-1]) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read(1).astype(np.float32)
            prof = dataset.profile
    del img_bytes
    return arr, prof


def read_asf_rtc_image_data(urls: list[str], max_workers: int = 5) -> tuple[list]:
    N = len(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data = list(tqdm(executor.map(read_one_asf, urls), total=N, desc='Loading RTC data'))
    arrs, profiles = zip(*data)
    return arrs, profiles
