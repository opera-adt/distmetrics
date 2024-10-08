# distmetrics 


## Background

This is a python implementation of disturbance metrics for OPERA RTC-S1 data. The intention is to use this library to quantify disturbance in the RTC imagery. Specifically, our "metrics" define distances between a set of dual polarizations "pre-images" and a single dual polarization "post-image". Some of the metrics only work on single polarization imagery.

The following metrics have been implemented in this library:

1. Log-ratio (raw) - this is not a non-negative function just a difference of pre and post images in db [[1]](#1). Only works on single polarization images.
2. Mahalanobis 1d and 2d based on empirically estimated statistics in patches around each pixel [[2]](#2), [[3]](#3).
3. Maholanobis distances for each polarization where mean/std are estimated from a Vision Transformer [[4]](#4) inspired by [[2]](#2).
4. Cumulative Sums for a given polarization from [[5]](#5) and [[6]](#6).

It is worth noting that other metrics can be generated from the above using `+`, `max`, `min` or linear combinations (with positive scalars). As such, when the distmetric has some auxiliary meaning (e.g. as a probability), such combinations are easier as they are more meaningfully comparable.


## Installation for development

Clone this repository and navigate to it in your terminal. We use the python package manager `mamba`. We highly recommend mamba and the package repository `conda-forge` to organize and manage virtual environment required for this library.

1. `mamba env update -f environment.yml`
2. Activate the environment `conda activate dist-s1`
3. Install the library with `pip` via `pip install -e .` (`-e` ensures this is editable for development)
4. Install a notebook kernel with `python -m ipykernel install --user --name dist-s1`.

Python 3.10+ is supported. When using the transformer model, if you have `gpu` available, it adviseable to check that the output from the below snippet is indeed `cuda`:

```
from distmetrics import get_device

get_device() # should be `cuda` if GPU is available or `mps` if using mac M chips
```

# Usage

See the [`notebooks/`](notebooks/).

# References

<a id="1">[1]</a> E. J. M. Rignot and J. J. van Zyl, "Change detection techniques for ERS-1 SAR data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 31, no. 4, pp. 896-906, July 1993, doi: 10.1109/36.239913. https://ieeexplore.ieee.org/document/239913 

<a id=2>[2]</a> O. L. Stephenson et al., "Deep Learning-Based Damage Mapping With InSAR Coherence Time Series," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 5207917, doi: 10.1109/TGRS.2021.3084209. https://arxiv.org/abs/2105.11544 

<a id=3>[3]</a> Deledalle, CA., Denis, L. & Tupin, F. How to Compare Noisy Patches? Patch Similarity Beyond Gaussian Noise. Int J Comput Vis 99, 86–102 (2012). https://doi.org/10.1007/s11263-012-0519-6. https://inria.hal.science/hal-00672357/

<a id=4>[4]</a> H. Hardiman Mostow et al., "Deep Self-Supervised Global Disturbance Mapping with Sentinel-1 OPERA RTC Synthetic Aperture Radar", *in preparation 2024*.

<a id=5>[5]</a> Sarem Seitz, "Probabalistic Cusum for Change Point Detection", Accessed September 2024.

<a id=6>[6]</a> CUSUM on [Wikipedia](https://en.wikipedia.org/wiki/CUSUM), Accessed September 2024