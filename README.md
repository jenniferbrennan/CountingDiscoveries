# Code to re-create Figure 2

submission_notebook.ipynb is a Jupyter notebook (https://jupyter.org/) that walks you through the construction of Figure 2 from our paper, "Estimating the number and effect sizes of non-null hypotheses".
Supporting code for this notebook can be found in submission_code.py

The entire notebook takes ~1hr to run on ten cores on a standard Linux machine. Options are provided in the notebook for faster execution, at the expense of a less accurate optimization.

## Dependencies
python 3
numpy
cvxpy (1.0 or later, tested with 1.0.21) with ECOS solver (distributed with CVXPY by default)
scipy
multiprocessing

## Fruit fly data set

This repository also contains the data from Supplement 1 of Hao et al. (2008), fruit fly experiments, as a text file. This file contains its own README.
