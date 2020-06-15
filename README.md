# Code for our paper, Estimating the Number and Effect Sizes of Non-Null Hypotheses

This repository contains all the code to reproduce the figures in our paper, "Estimating the Number and Effect Sizes of Non-Null Hypotheses," including examples of its use on both real and synthetic data sets. 

Suppose you run an experiment and get many test statistics, with a known distribution (eg, perhaps the measurement noise is normal, or the measurements are binomial). The code in this repository can be used to estimate the fraction of effect sizes (true means) above a user specified threshold (or list of thresholds). The estimate returned by our algorithm is guaranteed not to exceed the true fraction with high probability (the user-specified `alpha`). We provide examples and code for Gaussian, binomial, and poisson test statistics. The estimator supports other single-parameter test statistics; you will need to add code for generating an appropriate matrix A into `utils.py` in order to support a new statistic.

A note of caution: This estimator is sensitive to your specification of the test statistic distribution (gaussian, binomial, poisson, etc). Misspecifying the distribution, for example getting the wrong variance for the gaussian distribution, can produce very wrong answers. Be sure you examine your data to confirm that the hypothesized distribution is actually reasonable.

## Navigating this repository

There is one Jupyter notebook for each figure in the paper. In addition, we provide the notebook `ComparingDiscretizations` to illustrate the time/accuracy tradeoffs when choosing how to discretize the real line for continuous test statistics.

The code for our estimator is provided in `estimator.py`, with utility functions given in `utils.py`. There are several other Python files that implement baselines and provide scaffolding for synthetic experiments.


## Dependencies
### Python
- python 3
- numpy
- cvxpy (1.0 or later, tested with 1.0.21) with ECOS solver (distributed with CVXPY by default)
- scipy
- multiprocessing

### R (to compare against the baseline method "ashr")
- R (3.0.2 or later; tested on 3.6.0)
- plyr
- doParallel
- ashr

## Fruit fly data set

This repository also contains the data from Supplement 1 of Hao et al. (2008), fruit fly experiments, as a text file. This file has its own README. See `FlyPlots.ipynb` for an example of processing this data, as well as code to re-create the Figure ? from our paper.
