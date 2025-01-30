# distmetrics 

[![PyPI license](https://img.shields.io/pypi/l/distmetrics.svg)](https://pypi.python.org/pypi/distmetrics/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/distmetrics.svg)](https://pypi.python.org/pypi/distmetrics/)
[![PyPI version](https://img.shields.io/pypi/v/distmetrics.svg)](https://pypi.python.org/pypi/distmetrics/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/distmetrics)](https://anaconda.org/conda-forge/distmetrics)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/distmetrics)](https://anaconda.org/conda-forge/distmetrics)

This is a python library for calculating a variety of generic disturbance metrics from input OPERA RTC-S1 time-series including a transformer-based metric proposed in Hardiman-Mostow et al., 2024 [[1]](#1).
Generic land disturbances refer to any land disturbances observable with OPERA RTC-S1 including land-use changes, natural disasters, deforestation, etc.
A disturbance metric is a per-pixel function that quantifies via a radiometric or statistical measures such generic land disturbances between a set of baseline images (pre-images) and a new acquisition (post-image).
This library is specific to the dual-polarization VV $+$ VH OPERA RTC-S1 data and will likely need to be modified for other SAR data.
The user is expected to provide/curate the co-registered pre-imagery and the post-image for the computation of the distmetric.

# Usage

See the [`notebooks/`](notebooks/). 
These notebooks show how to download the necessary, publicly available time series and calculate these disturbance metrics.


## Background

This is a python implementation of disturbance metrics for OPERA RTC-S1 data. The intention is to use this library to quantify disturbance in the RTC imagery. Specifically, our "metrics" define distances between a set of dual polarizations "pre-images" and a single dual polarization "post-image". Some of the metrics only work on single polarization imagery.

The following metrics have been implemented in this library:

1. Transformer metric - mean and std estimated from a Vision Transformer [[1]](#1) inspired by [[2]](#2).
2. Mahalanobis 1d and 2d  - based on mean and std from sample statistics in patches around each pixel [[3]](#3), [[4]](#4).
3. Log-ratio - this is not a non-negative function just a difference of pre and post images in dB [[2]](#1). Only works on single polarization images.
4. CuSum metric - both absolute residuals and normalized residuals are computed in a per-pixel fashion. See [[5]](#5) and [[6]](#6).

It is worth noting that other metrics can be generated from the above using `+`, `max`, `min` or linear combinations (with positive scalars). As such, when the distmetric has some auxiliary meaning (e.g. as a probability), such combinations are easier as they are more meaningfully comparable.

## Installation

We recommend using the `conda/mamba` package manager to install this library.

```
mamba install -c conda-forge distmetrics
```

You can also use `pip`, although this doesn't ensure proper dependencies are installed.

### GPU support

To get the best performance of pytorch, you need to ensure pytorch recognizes the GPU.
Using `conda-forge` distributions, you may require you to ensure that `cudatoolkit` is installed (this is the additional library in `environment_gpu.yml`).
For our servers, we needed to install `cudatoolkit>=11.8` to get pytorch to recognize the GPU.
There are certain libraries that may downgrade `pytorch` to use CPU only (you can check this by looking at the distribution of pytorch before installing the library).
There may be different distributions of pytorch and cuda drivers that are compatible, but providing detailed instructions is beyond the scope of these instructions.


### For development

Clone this repository and navigate to it in your terminal. We use the python package manager `mamba`. We highly recommend mamba and the package repository `conda-forge` to organize and manage virtual environment required for this library.

1. `mamba env update -f environment.yml`
2. Activate the environment `conda activate distmetrics`
3. Install the library with `pip` via `pip install -e .` (`-e` ensures this is editable for development)
4. Install a notebook kernel with `python -m ipykernel install --user --name dist-s1`.

Python 3.10+ is supported. When using the transformer model, if you have `gpu` available, it adviseable to check that the output from the below snippet is indeed `cuda`:

```
from distmetrics import get_device

get_device() # should be `cuda` if GPU is available or `mps` if using mac M chips
```

# References

<a id=1>[1]</a> H. Hardiman Mostow et al., "Deep Self-Supervised Disturbance Mapping with Sentinel-1 OPERA RTC Synthetic Aperture Radar", [https://arxiv.org/abs/2501.09129](https://arxiv.org/abs/2501.09129).

<a id=2>[2]</a> O. L. Stephenson et al., "Deep Learning-Based Damage Mapping With InSAR Coherence Time Series," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 5207917, doi: 10.1109/TGRS.2021.3084209. https://arxiv.org/abs/2105.11544 

<a id="3">[3]</a> E. J. M. Rignot and J. J. van Zyl, "Change detection techniques for ERS-1 SAR data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 31, no. 4, pp. 896-906, July 1993, doi: 10.1109/36.239913. https://ieeexplore.ieee.org/document/239913 

<a id=4>[4]</a> Deledalle, CA., Denis, L. & Tupin, F. How to Compare Noisy Patches? Patch Similarity Beyond Gaussian Noise. Int J Comput Vis 99, 86–102 (2012). https://doi.org/10.1007/s11263-012-0519-6. https://inria.hal.science/hal-00672357/

<a id=5>[5]</a> Sarem Seitz, "Probabalistic Cusum for Change Point Detection", https://web.archive.org/web/20240817203837/https://sarem-seitz.com/posts/probabilistic-cusum-for-change-point-detection/, Accessed September 2024.

<a id=6>[6]</a> Tartakovsky, Alexander, Igor Nikiforov, and Michele Basseville. Sequential analysis: Hypothesis testing and changepoint detection. CRC press, 2014.
