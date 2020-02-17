# This python file contains the utility functions necessary to run our estimator.

import numpy as np
import cvxpy
import scipy.stats
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt

# Set this for your machine.
# This many cores will be used for parallel computation of threshold estimates
NUM_CORES = 10

def getSamples(n, mixingProportions, distributionComponents, discretization=1000, sigmaSquared=1, padding=5):
    # Sample from a mixture of Gaussians, with given mixing distribution (distribution of means)
    # mixingProportions is a vector that sums to 1, and distributionComponents is a vector of "frozen" distributions
    #   which each have a method ".rvs" for generating random variates
    #
    # Example usage: 
    # dist = [scipy.stats.bernoulli(0), scipy.stats.beta(a=8, b=4, scale=4)]  # Mean is zero in first component
    # prop = [0.85, 0.15]
    # getSamples(n=10000, mixingProportions=prop, distributionComponents=dist, discretization=1000, sigmaSquared=1, padding=5)
    
    # Choose the component to sample from
    distChoices = np.random.choice(a=len(mixingProportions), 
                                   size=n, 
                                   replace=True,
                                   p=mixingProportions)
    
    # Sample from that component with the .rvs() function
    means = []
    for i in range(len(mixingProportions)):
        theseMeans = list(distributionComponents[i].rvs(size=sum(distChoices==i)))
        means = means + theseMeans
    
    grid = np.linspace(min(means)-padding, max(means)+padding, discretization)
    
    means = np.array(means).reshape(n,1)
    
    noise = np.random.randn(n, 1)*sigmaSquared
    observations = means + noise
    
    print(means)
    
    return observations, grid


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


def construct_A(ticks, sigmaSquared):
    # build regression matrix from "true" effect size to noisy observation
    # we should roughly have that b = A*w for the true underlying w
    n = len(ticks)
    dticks = ticks[1]-ticks[0]
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            A[i,j] = np.exp(-.5*(ticks[i]-ticks[j])**2/sigmaSquared)/np.sqrt(2*np.pi*sigmaSquared)
    return A


def estimateEntireLine(obs, tolerance, alpha, hypTest, grid, gammas, sigmaSquared, verbose=False, parallel=True):
    """Provide simultaneous underestimates for zeta over numEstimates thresholds.
    Inputs: obs, observations X_i
            tolerance, the inherent accuracy limitation in binary search, in (0,1)
            alpha, the confidence parameter for simultaneous confidence on all the returned thresholds
            hypTest, the hypothesis test we will use
            grid, the discretization for the hypothesis test,
            gammas, the places we want to plot the curve
            sigmaSquared, the (known) noise in the observations
    Returns: zetaHats, a list of underestimates
             xVals, the corresponding threshold values of these underestimates (so you can plot xVals, zetaHats to get the curve)"""
    A = construct_A(grid, sigmaSquared)
    if parallel:
        jobs = [(obs, gamma, tolerance, alpha, hypTest, grid, sigmaSquared, verbose, A) for gamma in gammas]
        with Pool(NUM_CORES) as p:
            zetaHats = p.starmap(estimateEntireLine_oneJob, jobs)
    else:
        zetaHats = []
        for gamma in gammas:
            zetaHats.append(estimateEntireLine_oneJob(obs, gamma, tolerance, alpha, hypTest, grid, sigmaSquared, verbose, A))
    
    return zetaHats


def estimateEntireLine_oneJob(obs, gamma, tolerance, alphaAdjusted, hypTest, grid, sigmaSquared, verbose, A):
    """Method to parallelize estimateEntireLine"""
    np.random.seed()
    zetaHat = binarySearch(obs, gamma, tolerance, alphaAdjusted, hypTest, grid, sigmaSquared, verbose, A)
    return zetaHat


def binarySearch(obs, gamma, tolerance, alpha, hypTest, grid, sigmaSquared, verbose=False, A=None):
    """Estimate zeta using binary search.
    Inputs: obs, the observations X_i
            gamma, the threshold (we're estimating the mass above gamma)
            tolerance, the inherent accuracy limitation in our binary search, in (0,1) (1/n is a reasonable choice)
            alpha, the confidence parameter; w.p. 1-alpha, the returned zetaHat will be an underestimate of zeta
            hypTest, the hypothesis test we will use
            grid, the discretization
            sigmaSquared, the (known) noise of observations
    Returns: zetaHat, an understimate of zeta - tolerance w.p. 1-alpha"""
    
    zetaHatMin = 0
    zetaHatMax = 1
    numTests = np.ceil(np.log2(1/tolerance))
    
    if verbose:
        print(numTests, "tests")
    
    # We're trying to find the largest value of zetaHatTest that we can reject
    while zetaHatMax - zetaHatMin > tolerance:
        zetaHatTest = (zetaHatMax + zetaHatMin)/2  # Hypothesis test at the average value
        testResult = hypTest(obs, gamma, zeta=zetaHatTest, alpha=alpha, grid=grid, sigmaSquared=sigmaSquared, verbose=verbose, A=A)
        
        if verbose:
            print(testResult, zetaHatTest)
            
        if testResult == 'reject':
            # Conclude there is at least zetaHatTest mass above gamma
            zetaHatMin = zetaHatTest
        elif testResult == 'failToReject':
            # Test at a smaller value to find one we can reject
            zetaHatMax = zetaHatTest
        else:
            print("Unrecognized test result", testResult)
            exit()
    
    return zetaHatMin


def KS_test(obs, gamma, zeta, alpha, grid, sigmaSquared=1, verbose=False, A=None):
    """Use the KS test to test the null hypothesis 'no more than zeta mass is above gamma' at level alpha
    Input: obs, a list of observations X_i
           gamma, the threshold for the test
           zeta, the amount of mass you're testing (between 0 and 1)
           alpha, the desired false positive rate (between 0 and 1)
           grid, the discretization over which the test will be performed
           sigmaSquared, the (known) noise in the observations
    Returns: 'reject' or 'failToReject', depending on the outcome of the test"""
    n = len(obs)
    dkwBound = np.sqrt(np.log(2/alpha)/(2*n))
    
    counts = get_counts(obs, grid)
    if A is None:
        A = construct_A(grid, sigmaSquared)
    (w, loss) = fit_KS(A, b=counts/n, zeta=zeta, gamma=gamma, ticks=grid)
    
    if verbose:
        print("critical value", dkwBound, "vs loss", loss)
    
    return 'reject' if loss > dkwBound else 'failToReject'


def fit_KS(A, b, zeta, gamma, ticks):
    """Solves the optimization min_{nu in H0} ||\hat{F} - F_nu||_infty
    and returns the PDF of best fit as well as the value of the objective function"""
    n = A.shape[1]
    disc = ticks[1] - ticks[0]  # like dt in the intergral
    
    w = cvxpy.Variable(n, nonneg=True)
    wCDF = cvxpy.cumsum(A @ w)*disc
    ECDF = cvxpy.cumsum(b)
    loss = cvxpy.atoms.norm_inf(wCDF - ECDF)
    objective = cvxpy.Minimize(loss)
    
    mask = [int(t <= gamma) for t in ticks]
    idx = sum(mask)-1
    constraints = [w >= 0,
                  sum(w) == 1,
                  cvxpy.cumsum(w)[idx] >= 1-zeta]
    
    prob= cvxpy.Problem(objective, constraints)
    
    result = prob.solve(verbose=False, solver=cvxpy.ECOS)
    
    if prob.status != 'optimal':
        print(prob.status)
    
    return(w.value, loss.value)


def estimateZeta_FWER(obs, gamma, alpha, sigmaSquared, verbose=False):
    """Estimate the number of means above the threshold gamma by
    testing whether each one is above gamma. Do this in a FWER-controlled way."""
    n = len(obs)
    alphaAdjusted = alpha / n  # Bonferroni correction
    pValues = [scipy.stats.norm.cdf(gamma, loc=x_i, scale=np.sqrt(sigmaSquared)) for x_i in obs]
    numMeansAboveGamma = sum([1 if p < alphaAdjusted else 0 for p in pValues])
    return numMeansAboveGamma/n