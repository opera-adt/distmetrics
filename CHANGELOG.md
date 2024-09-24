# 0.0.1
First release of distmetrics which provides metrics of disturbance for dual-polarization OPERA RTC-S1 data.
More precisely, the "metrics" are functions({set of pre-images}, post-image). These functions are distance functions
from the set of pre-images to post-images. The API currently assumes the inputs are lists of numpy arrays.
This is because each RTC-S1 product is distributed separately and stacking them into a single array requires an 
extra step for the user.

Provides an interface for the following metrics:

1. logratio- this is not a non-negative function just a difference of pre and post images in db, but we transform into a metric by only inspecting the *decrease* in a given polarization
2. Mahalanobis 1d and 2d based on empirically estimated statistics in patches around each pixel.
3. Maholanobis distances for each polarization where mean/std are estimated from a Vision Transformer.
4. CuSum (on the actual residuals and on a normalized time-series)