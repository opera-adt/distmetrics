# distmetrics 


## Background

This is a python implementation of disturbance metrics for OPERA RTC-S1 data. Right now, it has the Mahalanobis distance for dual polarization imagary. The intention is to use this library to quantify disturbance in the RTC imagery. Specifically, our "metrics" define distances between a set of dual polarizations "pre-images" and a single dual polarization "post-image". Some of the metrics only work on single polarization imagery.

The following metrics have been implemented in this library:

1. Log-ratio (raw) - this is not a non-negative function just a difference of pre and post images in db [[1]](#1).
2. Mahalanobis 1d and 2d based on empirically estimated statistics in patches around each pixel [[2](#2), [3](#3)].
3. Maholanobis distances for each polarization where mean/std are estimated from a Vision Transformer [[4]](#2) inspired by [[2]](#2).

## Installation for development

Clone this repository and navigate to it in your terminal. We use the python package manager `mamba`.

1. `mamba env update -f environment.yml`
2. Activate the environment `conda activate dist-s1`
3. Install the library with `pip` via `pip install -e .` (`-e` ensures this is editable for development)
4. Install a notebook kernel with `python -m ipykernel install --user --name dist-s1`.

Python 3.10+ is supported.


# Usage

See the [`notebooks/`](notebooks/).

# References

<a id="1">[1]<a> E. J. M. Rignot and J. J. van Zyl, "Change detection techniques for ERS-1 SAR data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 31, no. 4, pp. 896-906, July 1993, doi: 10.1109/36.239913. https://escholarship.org/content/qt02j5r0qf/qt02j5r0qf.pdf </a>

<a id=2>[2] O. L. Stephenson et al., "Deep Learning-Based Damage Mapping With InSAR Coherence Time Series," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 5207917, doi: 10.1109/TGRS.2021.3084209. https://arxiv.org/abs/2105.11544 </a>

<a id=3>[3] Deledalle, CA., Denis, L. & Tupin, F. How to Compare Noisy Patches? Patch Similarity Beyond Gaussian Noise. Int J Comput Vis 99, 86–102 (2012). https://doi.org/10.1007/s11263-012-0519-6. https://inria.hal.science/hal-00672357/</a>

<a id=4>[4] H. Hardiman Mostow et al., "Deep Self-Supervised Global Disturbance Mapping with Sentinel-1 OPERA RTC Synthetic Aperture Radar", *in preparation 2024*.</a>