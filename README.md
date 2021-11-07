# RSS Cache Detection

## Overview
Please see `method.ipynb` for a summary and discussion of my approach. In general, this is for estimating the caching interval of a website feed, but modelling the feed as a Poisson Point process. First this is done directly by fittung an exponential distribution to the times feed entriesa are created. Second this is done by using a threshold clustering algorithms to estimate which entries were uploaded in the same cache, but the entry is created after polling at a slightly different time due to any latency/noise. The threshold clustering has its threshold parameter tuning by optimising F1-score on the classification of whether a feed interval is >600s or not.

This was done as a job interview take home task, so the original data and worksheet are not included.

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




