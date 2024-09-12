# distmetrics 


## Background

This is a python implementation of disturbance metrics for OPERA RTC-S1 data. Right now, it has the Mahalanobis distance for dual polarization imagary. The intention is to use this library to quantify disturbance in the RTC imagery. Specifically, our "metrics" define distances between a set of dual polarizations "pre-images" and a single dual polarization "post-image". Some of the metrics only work on single polarization imagery.

The following metrics have been implemented:

1. logratio (raw) - this is not a non-negative function just a difference of pre and post images in db.
2. Mahalanobis 1d and 2d based on empirically estimated statistics in patches around each pixel.
3. Maholanobis distances for each polarization where mean/std are estimated from a Vision Transformer.

## Installation for development

Clone this repository and navigate to it in your terminal. We use the python package manager `mamba`.

1. `mamba env update -f environment.yml`
2. Activate the environment `conda activate dist-s1`
3. Install the library with `pip` via `pip install -e .` (`-e` ensures this is editable for development)
4. Install a notebook kernel with `python -m ipykernel install --user --name dist-s1`.

Python 3.10+ is supported.


# Usage

See the [`notebooks/`](notebooks/).