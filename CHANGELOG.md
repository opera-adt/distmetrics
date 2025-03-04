# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.11] - 2025-03-05

### Added
* Ability to control `device` for transformer model

## [0.0.10] - 2025-02-20

### Added
* Ensures environment.yml and environment_gpu.yml are up to date with the latest dependencies.
* Requires rasterio>=1.4.0 for merging to be consistent.

## [0.0.9] - 2025-02-20

### Fixed
* Bug in `reproject_arrays_to_target_crs`.

### Added
* Tests for merge categorical arrays using inputs that are in different crs.

## [0.0.8] - 2025-02-20

### Fixed
* Pyproject.toml kept dependencies for mpire and dill (those are now removed)


## [0.0.7] - 2025-02-20

### Fixed
* Pyproject.toml and environment.yml to require python 3.12+
* Pydantic model_config to allow arbitrary types (update for v2)

### Removed 
* `mpire` dependency

### Added
* Test for despeckling of cropped data


## [0.0.6] - 2025-01-18

* Fixed default keyword argument for `merge_categorical_arrays` from `mode` (not possible with rasterio) to `min`.


## [0.0.5] - 2025-01-18

### Added
* Added `rio_tools` for merging float arrays (including burst data such as the computed metric)
    * can be used to average over overlapping areas
    * average using the distance from exterior mask (mask that touches one of the four edges)
    * exterior mask can also be dilated to avoid problematic boundary pixels
    * categorical merging of data
* Added `nd_tools` for getting exterior mask and distance from such mask (wrappers around `scipy.ndimage`)
* A notebook for applying the transformer metric to a large area of interest

### Removed
* Removed torch.compile from transformer model loading

## [0.0.4] - 2025-01-18

### Fixed
* Fixed transformer to ensure last pre-image always has correct index

### Added
* Arxiv link to README
* Installation instructions
* `tqdm` description for despeckling
* Allow user to disable `tqdm` for despeckle

### Changed
* Renamed `load_trained_transformer_model` to `load_transformer_model`
* Renamed inputs to reflect (possible) usage with other polarizations: `vv` -> `copol` and `vh` -> `crosspol`. The APIs don't change, just the variable name inputs.


## [0.0.3] - 2025-01-16

### Fixed
* Fixed ASF calls in landslide transformer notebook - more data is becoming available so static slices do not work

### Added

* Consistent linting of DIST-S1 repositories
* Expose estimation of the normal parameters of the logit images with high/low memory usage.
* Added transformer test based on usage of different strategies of computation.
* Uploaded latest model

### Changed
* Data directory structure
* Only support python 3.12+


## [0.0.2]

### Fixed

* Fixes Changelog format to ensure proper automated release
* Fixes secrets in workflows for proper automated release


## [0.0.1]

This was an initial release of the library.

This is a python library for calculating a variety of generic disturbance metrics from input OPERA RTC-S1 time-series including a transformer-based metric from Hardiman-Mostow et al., 2024.
Generic land disturbances refer to any land disturbances observable with OPERA RTC-S1 including land-use changes, natural disasters, deforestation, etc.
A disturbance metric is a per-pixel function that quantifies via a radiometric or statistical measures such generic land disturbances between a set of baseline images (pre-images) and a new acquisition (post-image).
This library is specific to the dual-polarization OPERA RTC-S1 data and will likely need to be modified for other SAR data.
The user is expected to provide/curate the co-registered baseline imagery (pre-images) and the recent acquisition (post-image) for the computation of the distmetrics.

Provides an interface for the following metrics:

1. transformer - this is metric that uses spatiotemporal context to approximate a per-pixel probability of baseline images. This that was proposed in Hardiman-Mostow, 2024.
2. logratio - this is not a non-negative function just a difference of pre and post images in db, but we transform into a metric by only inspecting the *decrease* in a given polarization
3. Mahalanobis 1d and 2d - based on sample statistics computed in patches around each pixel.
4. CuSum - based on the actual residuals and on a normalized time-series assuming normality.