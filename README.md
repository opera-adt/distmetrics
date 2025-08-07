# distmetrics 

[![PyPI license](https://img.shields.io/pypi/l/distmetrics.svg)](https://pypi.python.org/pypi/distmetrics/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/distmetrics.svg)](https://pypi.python.org/pypi/distmetrics/)
[![PyPI version](https://img.shields.io/pypi/v/distmetrics.svg)](https://pypi.python.org/pypi/distmetrics/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/distmetrics)](https://anaconda.org/conda-forge/distmetrics)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/distmetrics)](https://anaconda.org/conda-forge/distmetrics)

The `distmetrics` (or informally `dist-lib`) provides a set of python tools and metrics to identify generic disturbances within OPERA RTC-S1 time-series including a transformer-based metric proposed in Hardiman-Mostow et al., 2024 [[1]](#1).
The transformer metric and its application occupies most of this library in order to effectively, efficiently apply this deep-learning based model (using a visual transformer architecture).
Also, there are:
- tools for despeckling
- GIS tools to merge burst-wise products after metrics have been computed
- downloading burst time series
"Generic land disturbances" refer to any land disturbances observable with OPERA RTC-S1 including land-use changes, natural disasters, deforestation, etc.
A disturbance metric is a per-pixel function that quantifies land disturbances between a set of baseline images (pre-images) and a new acquisition (post-image).
This library is specific to the dual-polarization VV $+$ VH OPERA RTC-S1 data.

# Usage

See the [`notebooks/`](notebooks/). 
These notebooks show how to:

1. download the necessary, publicly available time series of OPERA RTC-S1 data (see setup below)
2. despeckle the time-series and 
3. calculate the disturbance metrics for delineating areas of disturbances

## Setup

Please make sure to ensure you have an ASF account (https://search.asf.alaska.edu/#/) and set up a `~/.netrc` with:

```
machine urs.earthdata.nasa.gov
    login <username>
    password <password>
```


## Provenance of Transformer Models

Currently there are 4 models in this library (see `from distmetrics.model_load import ALLOWED_MODELS`), though you can load your own weights and config assuming the same architecture.
The models in this library are:
   - `transformer_original`
   - `transformer_optimized`
   - `transformer_optimized_fine`
   - `transformer_anniversary_trained`
   - `transformer_anniversary_trained_10`
   - `transformer_anniversary_trained_optimized`
   - `transformer_anniversary_trained_optimized_fine`

Please see the [dist-s1-model](https://github.com/opera-adt/dist-s1-model) [A] for training this transformer model and [dist-s1-training-data](https://github.com/opera-adt/dist-s1-training-data) [B] for curating a dataset from OPERA RTC-S1. There is a link in [A] to the existing dataset that was curated based on time-series with dense preimages and despeckled and masked water. In [B], you can understand how the data is curated. For the models above, here is some additional discussion:
- `transformer_original` - the model trained by Harris Hardiman-Mostow on the training data linked to in [A].
- `transformer_optimized` and `transformer_optimized_fine` are models with the same architecture as `transformer_original` and using the same dataset in [A] by [dmartinez05](https://github.com/dmartinez05) with reduced size. The `fine` refers to the fact that within the input size of the model (`16 x 16`) there are patches used within the input size that are `4 x 4` (as opposed to `8 x 8`).
- `transformer_anniversary_trained` and `transformer_anniversary_trained_10` were trained by [Jungkyo Jung](https://github.com/oberonia78) using a dataset similar to one found in [B] using despeckling and some landcover masking. The context length of the the former is 20 and the latter 10.
- `transformer_anniversary_trained_optimized` and `transformer_anniversary_trained_optimized_fine` were trained by [dmartinez05](https://github.com/dmartinez05) using the same dataset used for `transformer_anniversary_trained` and are optimized in that they use less parameters and the `fine` suffix refers to the internal model patches are `4 x 4`.


## Background on Metrics

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
