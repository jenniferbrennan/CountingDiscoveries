# This python file contains the utility functions necessary to run our estimator.

import numpy as np
import cvxpy
import scipy.stats
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt

def get_counts(mu_hat, ticks):
    # bin mu_hat into bins centered at the values of ticks
    dticks = ticks[1]-ticks[0]
    n = len(ticks)
    counts = np.zeros(n)
    counts[0] = np.sum( (mu_hat<ticks[0]+dticks/2) )
    counts[-1] = np.sum( (mu_hat>=ticks[-1]-dticks/2) )
    for i in range(1,n-1):
        counts[i] = np.sum( (mu_hat>=ticks[i]-dticks/2) * (mu_hat<ticks[i]+dticks/2) ) # wow, so clever
    return counts


def construct_A_gaussian(ticks, sigmaSquared):
    # Regression matrix to regress on the empirical CDF. We should roughly have that eCDF = A*w for the true w
    # Note, in an idea world, our estimtor would work just as well regressing on the pdf. But, if w has extremely small elements,
    # then when we multiply by A' s.t. ePDF = A*w, we will not get their right tails. The optimization sometimes
    # takes advantage of this, which is bad. So we use the CDF instead.
    n = len(ticks)
    dticks = ticks[1] - ticks[0]
    A = np.zeros((n, n))
    for j in range(n):
        # jth column is the normal CDF centered at ticks[j]
        A[:,j] = scipy.stats.norm.cdf(ticks, loc=ticks[j], scale=np.sqrt(sigmaSquared))
    return A

def construct_A_gaussian_pdf(ticks, sigmaSquared):
    # Regression matrix to regress on the empirical PDF. (Need to use the PDF for the MLE)
    # We should have that the true pdf of X_i's = A*w for the true w
    n = len(ticks)
    dticks = ticks[1] - ticks[0]
    A = np.zeros((n, n))
    for j in range(n):
        # jth column is the normal PDF centered at ticks[j]
        A[:,j] = scipy.stats.norm.pdf(ticks, loc=ticks[j], scale=np.sqrt(sigmaSquared))
    return A

def construct_A_Bin(paramGrid, t):
    # Construct the regression matrix for the mixture of binomials
    # t is the number of trials
    n = len(paramGrid)
    dticks = paramGrid[1]-paramGrid[0]
    A = np.zeros((t+1, n))
    for j in range(n):
        for i in range(t+1):
            A[i,j] = scipy.stats.binom.cdf(k=i, p=paramGrid[j], n=t)
    return A

def construct_A_Poi(lamGrid, obsGrid):
    # Construct the regression matrix for the mixture of Poissons
    # lamGrid is the discretization over parameters lambda, and obsGrid is the discretization over the observed X_i's
    # (these may be on very different scales)
    m = len(obsGrid)
    n = len(lamGrid)
    dticks = lamGrid[1]-lamGrid[0]
    A = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            A[i,j] = scipy.stats.poisson.cdf(k=i, mu=lamGrid[j])
    return A