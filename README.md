# distmetrics 


## Background

This is a python implementation of disturbance metrics for OPERA RTC-S1 data. Right now, it has the Mahalanobis distance for dual polarization imagary. The intention is to use this library to systematically define disturbance in the RTC imagery.


## Installation for development

Clone this repository and navigate to it in your terminal. We use the python package manager `mamba`.

1. `mamba env update -f environment.yml`
2. Activate the environment `conda activate dist-s1`
3. Install the library with `pip` via `pip install -e .` (`-e` ensures this is editable for development)
4. Install a notebook kernel with `python -m ipykernel install --user --name dist-s1`.

Python 3.10+ is supported.