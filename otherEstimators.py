import numpy as np
import cvxpy
import scipy

def mle(A_pdf, b, max_iters=None, verbose=False):
    """Returns the maximum likelihood mixing distribution, and the log likelihood
    A_pdf: A d-by-n matrix such that the true PDF = A^T nu, where nu\in\R^d is the true probability of seeing each mu_i, over some grid
    b: vector representing the empirical histogram of observations over a predefined grid of d elements; sums to 1
    
    Returns: a vector with n entries, representing the estimated density of nu, on a grid of n elements
    """
    
    d, n = A_pdf.shape
    w = cvxpy.Variable(n, nonneg=True)
    
    loss = 0
    for i in range(d):
        if b[i] > 0: 
            loss += -b[i]*cvxpy.log(A_pdf[i,:] @ w)
            
    objective = cvxpy.Minimize( loss )
    
    # Constrain w to the simplex
    constraints = [w >= 0, sum(w)==1]
    
    prob = cvxpy.Problem(objective, constraints)

    # if having convergence issues, try another solver: https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
    if max_iters is not None:
        result = prob.solve(verbose=verbose, solver=cvxpy.SCS, max_iters=max_iters) 
    else:
        result = prob.solve(verbose=verbose, solver=cvxpy.SCS)
    if verbose:
        print("Finished solving for mixing distribution, problem status", prob.status)
    w_est = w.value
    
    # Return a log likelihood, not a negative log likelihood!
    return (w_est, -1*loss.value)


def boundNumberDiscoveries_KR(data, alpha, gamma, pValCalc):
    '''
    Implement another baseline, which uses FDP control to count the number of discoveries
     We will do this using Katsevich and Ramdas' "Simultaneous high-probability fdp bounds" (2018, ArXiv),
     specifically for the "sorted" path (Eqn 8)
     
    data is a list of observed test statistics
    alpha is the chance that the true number of discoveries exceeds the bound
    gamma is the effect size threshold - we're counting effects at least gamma
    pValCalc takes a test statistic and a threshold gamma, and computes P(X > X_i) when
        the test statistic X_i ~ H0: mu = gamma
        For example, for the normal distribution, pValCalc(t, gamma) = P_{x~N(gamma,1)}(x > t)
    
    We will address the adjustment for different gamma by computing p-values against 
          H0: mu = gamma
    We will use Katsevich and Ramdas' simultaneous high-probability estimator to generate 
    a nested set of rejection sets, get the fdp rates for each, use those to lower bound
    the number of discoveries, and return the sup over those sets.
    
     Example usage:
      boundNumberDiscoveries(data, alpha=0.05, gamma=1, pValCalc=pValCalcNormal(sigma=1))
    '''
    n = len(data)
    
    # Step 1: Compute p-values
    pValues = [pValCalc(testStat, gamma)[0] for testStat in data]
    
    # Step 2: Compute \bar{V} from Thm 1 of Katsevich and Ramdas
    # Uniformly and with high probability, V-bar / |Rejection set| > FDP(rejection set)
    # so that V-bar is an upper bound on the number of false discoveries in the set
    # This means that |set| - vBar is a lower bound on the number of true discoveries in the set,
    # and a lower bound on the number of discoveries in the WHOLE data set
    #pValues = sorted(pValues)  # This is probably the most intensive part
    pValues.sort()  # In-place sort to use less memory
    
    inflationFactor = np.log(1/alpha)/np.log(1 + np.log(1/alpha))
    
    # Idea: For large values of n, this process seems really memory intensive;
    # we could save memory by: not allocation vBar or lowerBoundOnNumDiscoveries,
    # but instead computing the max in a for loop over the sorted pValues
    
    lowerBoundOnNumDisc = 0
    for i in range(n):
        bound = i + 1 - np.floor(inflationFactor * (1 + n*pValues[i]))
        if bound > lowerBoundOnNumDisc:
            lowerBoundOnNumDisc = bound
    
    return lowerBoundOnNumDisc

def boundNumberDiscoveries_DKW(data, alpha, gamma, pValCalc):
    '''
    Data is a list of observed test statistics
    alpha is the chance that the true number of discoveries exceeds the bound
    gamma is the effect size threshold - we're counting effects at least gamma
    pValCalc takes a test statistic and a threshold gamma, and computes P(X > X_i) when
        the test statistic X_i ~ H0: mu = gamma
        For example, for the normal distribution, pValCalc(t, gamma) = P_{x~N(gamma,1)}(x > t)
    
    We will address the adjustment for different gamma by computing p-values against 
          H0: mu = gamma
    We will use the DKW estimator from Meinshausen and Rice, cited in Katsevich and Ramdas, to generate 
    a nested set of rejection sets, get the fdp rates for each, use those to lower bound
    the number of discoveries, and return the sup over those sets.
    '''
    n = len(data)
    
    # Step 1: Compute p-values
    pValues = [pValCalc(testStat, gamma)[0] for testStat in data]
    
    # Step 2: Compute \bar{V} from Thm 1 of Katsevich and Ramdas
    # Uniformly and with high probability, V-bar / |Rejection set| > FDP(rejection set)
    # so that V-bar is an upper bound on the number of false discoveries in the set
    # This means that |set| - vBar is a lower bound on the number of true discoveries in the set,
    # and a lower bound on the number of discoveries in the WHOLE data set
    #pValues = sorted(pValues)  # This is probably the most intensive part
    pValues.sort()  # In-place sort to use less memory
    
    additiveFactor = np.sqrt(n/2*np.log(1/alpha))
    
    # Idea: For large values of n, this process seems really memory intensive;
    # we could save memory by: not allocation vBar or lowerBoundOnNumDiscoveries,
    # but instead computing the max in a for loop over the sorted pValues
    
    lowerBoundOnNumDisc = 0
    for i in range(n):
        bound = i + 1 - np.floor(additiveFactor + n*pValues[i])
        if bound > lowerBoundOnNumDisc:
            lowerBoundOnNumDisc = bound
    
    return lowerBoundOnNumDisc

def pValCalcNormal(sigma=1):
    '''Return a function that takes in a test statistic and gamma, then computes
    p-values for this given sigma'''
    def pValCalcWithSigma(testStat, gamma):
        return scipy.stats.norm.sf(testStat, loc=gamma, scale=sigma)
    return pValCalcWithSigma
