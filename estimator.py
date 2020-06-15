# Our estimator, and the associated scaffolding we need to run it
# We choose to implement our estimator using binary search over the value of zetaHat
# but we could also have directly solved the optimization problem on a discretized grid.
# These two approaches are equivalent; the fact that we went with binary search is
# an artifact of how we developed this technique, and not a requirement of our estimator.

import numpy as np
from multiprocessing import Pool, freeze_support
from utils import get_counts
import cvxpy


# Set this for your machine.
# This many cores will be used for parallel computation of threshold estimates
NUM_CORES = 10

def estimateEntireLine(obs, tolerance, alpha, hypTest, paramGrid, obsGrid, gammas, A, sigma, verbose=False, parallel=True):
    """Provide simultaneous conservative estimates for zeta over numEstimates thresholds.
    Inputs: obs, observations X_i
            tolerance, the inherent accuracy limitation in binary search, in (0,1)
            alpha, the confidence parameter for simultaneous confidence on all the returned thresholds
            hypTest, the hypothesis test we will use
            paramGrid, a vector with the grid over which we will discretize nu,
            obsGrid, a vector with the grid over which we will discretize X (usually equal to paramGrid for gaussian settings)
            gammas, the places we want to plot the curve
            A, the matrix that satisfies {empirical CDF} = A * {mixing distribution nu}
            sigma, the (known) noise in the observations
    Returns: zetaHats, a list of estimates
             xVals, the corresponding threshold values of these estimates (so you can plot xVals, zetaHats to get the curve)"""
    if parallel:
        jobs = [(obs, gamma, tolerance, alpha, hypTest, paramGrid, obsGrid, sigma, verbose, A) for gamma in gammas]
        with Pool(NUM_CORES) as p:
            zetaHats = p.starmap(estimateEntireLine_oneJob, jobs)
    else:
        zetaHats = []
        for gamma in gammas:
            zetaHats.append(estimateEntireLine_oneJob(obs, gamma, tolerance, alpha, hypTest, paramGrid, obsGrid, sigma, verbose, A))
    
    return zetaHats


def estimateEntireLine_oneJob(obs, gamma, tolerance, alphaAdjusted, hypTest, paramGrid, obsGrid, sigma, verbose, A):
    """Method to parallelize estimateEntireLine"""
    np.random.seed()
    zetaHat = binarySearch(obs, gamma, tolerance, alphaAdjusted, hypTest, paramGrid, obsGrid, sigma, verbose, A)
    return zetaHat


def binarySearch(obs, gamma, tolerance, alpha, hypTest, paramGrid, obsGrid, sigma, A, verbose=False):
    """Estimate zeta using binary search.
    Inputs: obs, the observations X_i
            gamma, the threshold (we're estimating the mass above gamma)
            tolerance, the inherent accuracy limitation in our binary search, in (0,1) (1/n is a reasonable choice)
            alpha, the confidence parameter; w.p. 1-alpha, the returned zetaHat will be an underestimate of zeta
            hypTest, the hypothesis test we will use
            paramGrid, a vector with the grid over which we will discretize nu,
            obsGrid, a vector with the grid over which we will discretize X (usually equal to paramGrid for gaussian settings)
            sigma, the (known) noise parameter of observations
    Returns: zetaHat, an understimate of zeta - tolerance w.p. 1-alpha"""
    
    zetaHatMin = 0
    zetaHatMax = 1
    numTests = np.ceil(np.log2(1/tolerance))
    
    if verbose:
        print(numTests, "tests")
    
    # We're trying to find the largest value of zetaHatTest that we can reject
    while zetaHatMax - zetaHatMin > tolerance:
        zetaHatTest = (zetaHatMax + zetaHatMin)/2  # Hypothesis test at the average value
        testResult = hypTest(obs, gamma, zeta=zetaHatTest, alpha=alpha, paramGrid=paramGrid,
                             obsGrid=obsGrid,
                             sigmaSquared=sigma**2 if sigma is not None else None, 
                             verbose=verbose, A=A)
        
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


def KS_test(obs, gamma, zeta, alpha, paramGrid, obsGrid, sigmaSquared=1, verbose=False, A=None):
    """Use the KS test to test the null hypothesis 'no more than zeta mass is above gamma' at level alpha
    Input: obs, a list of observations X_i
           gamma, the threshold for the test
           zeta, the amount of mass you're testing (between 0 and 1)
           alpha, the desired false positive rate (between 0 and 1)
            paramGrid, a vector with the grid over which we will discretize nu,
            obsGrid, a vector with the grid over which we will discretize X (usually equal to paramGrid for gaussian settings)
           sigmaSquared, the (known) noise in the observations
    Returns: 'reject' or 'failToReject', depending on the outcome of the test"""
    n = len(obs)
    dkwBound = np.sqrt(np.log(2/alpha)/(2*n))
    counts = get_counts(obs, obsGrid)
    
    (w, loss) = fit_KS(A, b=counts/n, zeta=zeta, gamma=gamma, paramGrid=paramGrid)
    
    if verbose:
        print("critical value", dkwBound, "vs loss", loss)
    
    return 'reject' if loss > dkwBound else 'failToReject'


def fit_KS(A, b, zeta, gamma, paramGrid):
    """Solves the optimization min_{nu in H0} ||\hat{F} - F_nu||_infty
    and returns the PDF of best fit as well as the value of the objective function"""
    n = A.shape[1]
    
    w = cvxpy.Variable(n, nonneg=True)
    wCDF = (A @ w)   # The expected CDF if we had this mixing distribution w (recall dfn of A)
    ECDF = cvxpy.cumsum(b)
    loss = cvxpy.atoms.norm_inf(wCDF - ECDF)
    objective = cvxpy.Minimize(loss)
    
    mask = [int(t <= gamma) for t in paramGrid]
    idx = sum(mask)-1
    constraints = [w >= 0,
                  sum(w) == 1,
                  cvxpy.cumsum(w)[idx] >= 1-zeta]
    
    prob= cvxpy.Problem(objective, constraints)
    
    result = prob.solve(verbose=False, solver=cvxpy.ECOS)
    
    if prob.status != 'optimal':
        print(prob.status)
    
    return(w.value, loss.value)

