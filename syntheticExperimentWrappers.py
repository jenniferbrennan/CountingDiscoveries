# Code for synthetic experiments: utility functions to
# draw one set of samples and estimate zeta using a variety of estimators.
# These functions can be repeated many times to get confidence intervals on zetaHat

import numpy as np
from sampling_utils import getSamples_gaussianTwoSpike, getSamples_poisson, getSamples_binomial, getSamples_gaussian
from estimator import binarySearch, estimateEntireLine, KS_test
from utils import construct_A_gaussian, construct_A_gaussian_pdf, construct_A_Bin, construct_A_Poi, get_counts
from otherEstimators import mle, boundNumberDiscoveries_KR, boundNumberDiscoveries_DKW, pValCalcNormal

def drawAndEstimate_GaussianTwoSpike(threshold, n, zeta, mu2, discretization, sigma, tolerance, estimator="", alpha=0.05):
    """Estimate the fraction of mass above the threshold. Draw samples from a two-spike Gaussian, estimate,
    and return zetaHat
    
    Estimator: one of "KS" (ours), "MLE" (the plugin MLE), "KR" (Katsevich and Ramdas, 2018), or "DKW" (Meinshausen and Rice, 2006)
    """
    np.random.seed()  # important, b/c if we parallelize with multiprocessing they will all have the same random state
    observations, grid = getSamples_gaussianTwoSpike(n, zeta, mu2, discretization, sigma)
    
    A = construct_A_gaussian(grid, sigma**2)
    
    if estimator == "KS":
        zetaHat = binarySearch(obs=observations, 
                                 gamma=threshold,
                                 tolerance=tolerance,
                                 alpha=alpha,
                                 hypTest=KS_test,
                                 paramGrid=grid,
                               obsGrid=grid,
                                 sigma=sigma,
                               A=A,
                               verbose=False)
    elif estimator == "MLE":
        counts = get_counts(observations, grid)

        b = counts/np.sum(counts)
        A = construct_A_gaussian_pdf(grid, sigma)

        # Compute the unconstrained MLE
        (w_hat_u, ll_u) = mle(A, b, max_iters=5000)

        zetaHat = sum([mass for (mass, loc) in zip(w_hat_u, grid) if loc > threshold])
    elif estimator == "KR":
        zetaHat_count = boundNumberDiscoveries_KR(observations, gamma=threshold, 
                                                         alpha=alpha,
                                                         pValCalc=pValCalcNormal(sigma=sigma))
        zetaHat = max(zetaHat_count/n, 0)
    elif estimator == "DKW":
        zetaHat_count = boundNumberDiscoveries_DKW(observations, gamma=threshold, 
                                                         alpha=alpha,
                                                         pValCalc=pValCalcNormal(sigma=sigma))
        zetaHat = max(zetaHat_count/n, 0)
    else:
        print("Estimator", estimator, "not recognized, expected one of 'KS', 'MLE', 'KR', 'DKW'")
    
    return zetaHat


def drawAndEstimate_continuousDist(distributionComponents, mixingProportions, n, tolerance, alpha, gammas, t=None, sigma=None, padding=5, discretization=1000, distribution=""):
    """ Perform one synthetic experiment. Right now, this only supports our estimator (because that's all I need),
    but it could be extended similarly to the above, to support other estimators."""
    
    if distribution == "normal":
        observations, paramGrid = getSamples_gaussian(n, mixingProportions, distributionComponents, discretization, sigma, padding)
        A = construct_A_gaussian(paramGrid, sigmaSquared=sigma**2)
        obsGrid =  paramGrid
    elif distribution == "binom":
        observations, paramGrid = getSamples_binomial(n, mixingProportions, distributionComponents, t, discretization)
        A = construct_A_Bin(paramGrid, t)
        obsGrid =  np.arange(0, t+1)
    elif distribution == "poi":
        observations, paramGrid = getSamples_poisson(n, mixingProportions, distributionComponents, discretization)
        obsGrid =  np.arange(0, max(observations)+padding)
        A = construct_A_Poi(lamGrid=paramGrid, obsGrid=obsGrid)
    else:
        print("Unknown distribution", distribution)
        
    zetaHats = estimateEntireLine(obs=observations, 
                                     tolerance=tolerance,
                                     alpha=alpha,
                                     hypTest=KS_test,
                                     paramGrid=paramGrid,
                                     obsGrid=obsGrid,
                                     gammas=gammas, 
                                     sigma=None, 
                                     verbose=False,
                                  A=A)

    
    return observations, zetaHats
    