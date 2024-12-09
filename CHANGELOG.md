# 0.0.1
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