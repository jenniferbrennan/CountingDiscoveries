# Utility functions for Family-Wise Error Rate (FWER) -controlled multiple hypothesis testing.
# This is the "identification-based" estimator

import scipy.stats
from multiprocessing import Pool, freeze_support


def estimateZeta_FWER(obs, gamma, alpha, sigma=None, t=None, distribution=""):
    """Estimate the fraction of means above the threshold gamma by
    testing whether each one is above gamma. Do this in an FWER-controlled way (at level alpha)
    Computation is based on the distribution you specify:
    "normal": normal distribution, must provide sigma
    "binom": binomial distribution, must provide t
    "poi": poisson distribution, no other parameters needed
    """
    if distribution not in ["normal", "binom", "poi"]:
        print("Expected distribution to be one of 'normal', 'binom', 'poi'; got", distribution)
        
    n = len(obs)
    alphaAdjusted = alpha / n  # Bonferroni correction
    
    if distribution == "normal":
        pValues = [scipy.stats.norm.cdf(gamma, loc=x_i, scale=sigma) for x_i in obs]
    elif distribution == "binom":
        pValues = [1 - scipy.stats.binom.cdf(k=x_i, p=gamma, n=t) for x_i in obs]
    elif distribution == "poi":
        pValues = [scipy.stats.poisson.cdf(k=x_i, mu=gamma) for x_i in obs]
    else:
        print("Unrecognized distribution", distribution)
    
    numMeansAboveGamma = sum([1 if p < alphaAdjusted else 0 for p in pValues])
    
    return numMeansAboveGamma/n


def estimateZeta_FWER_manyThresholds(obs, gammas, alpha, sigma=None, t=None, distribution=""):
    """
    Estimate the fraction of means above threshold gamma, for all gamma in gammas, using the FWER-based
    'identification' estimator.
    
    Note we use the (unadjusted) alpha each time, even if we are testing at many thresholds (that is,
    we do not control for multiple testing in the thresholds, so our resulting estimates are valid for each gamma,
    but perhaps not for all gammas simultaneously). This only gives FWER an advantage against our estimator.
    
    Computation is based on the distribution you specify:
    "normal": normal distribution, must provide sigma
    "binom": binomial distribution, must provide t
    "poi": poisson distribution, no other parameters needed
    
    If this is taking too long, try the parallel version, estimateZeta_FWER_manyThresholds_parallel
    """
    zetaHats_FWER = []
    for gamma in gammas:
        zetaHat_FWER = estimateZeta_FWER(obs, gamma, alpha, sigma=sigma, t=t, distribution=distribution)
        zetaHats_FWER.append(zetaHat_FWER)
    return zetaHats_FWER


def estimateZeta_FWER_manyThresholds_parallel(obs, gammas, alpha, sigma=None, t=None, distribution="", numCores=1):
    """A parallel version of estimateZeta_FWER_manyThresholds"""
    jobs = [(obs, gamma, alpha, sigma, t, distribution) for gamma in gammas]
    with Pool(numCores) as p:
        zetaHats_FWER = p.starmap(estimateZeta_FWER, jobs)
    
    return zetaHats_FWER