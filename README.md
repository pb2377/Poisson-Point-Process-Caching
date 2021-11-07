# RSS Cache Detection

## Overview
Please see `method.ipynb` for a summary and discussion of my approach. 

## Files
- `main.py` - Main script to run the code for both cases and produce csv of outputs.
- `src/cluster_expofit.py` - Implements the core cache interval estimator for a given feed,
 with the clustering algorithm described (or without if no threshold=0) and fitting an exponential distribution.
- `src/interval_estimator.py` - Adds threshold optimisation to the above cache interval estimator class.
- `src/module.py` - Implements full RSSCacheDetector combining to write predictions to csv.
- `src/utils.py` - Some generic longer functions used in the jupyter notebook to make that less cluttered.

## Python Environment
python=3.7

See `requirements.txt`




