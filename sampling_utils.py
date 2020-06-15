# Utility functions for sampling test statistics X_i from mixture distributions
# For use in synthetic experiments

import numpy as np
import scipy.stats

#############
# Gaussians
#############

def getSamples_gaussian(n, mixingProportions, distributionComponents, discretization=1000, sigma=1, padding=5):
    """ Sample from a mixture of Gaussians, with given mixing distribution (distribution of means)
     mixingProportions: a vector that sums to 1
     distributionComponents: a vector of "frozen" distributions
       which each have a method ".rvs" for generating random variates
    
     Example usage: 
     dist = [scipy.stats.bernoulli(0), scipy.stats.beta(a=8, b=4, scale=4)]  # Mean is zero in first component
     prop = [0.85, 0.15]
     getSamples_gaussian(n=10000, mixingProportions=prop, distributionComponents=dist, discretization=1000, sigma=1, padding=5)
     """
    
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
    
    noise = np.random.randn(n, 1)*sigma
    observations = means + noise
    
    
    return observations, grid

def getSamples_gaussianTwoSpike(n, zeta, mu2, discretization=1000, sigma=1, padding=5):
    """ A convenience function to sample from the two-spike Gaussian distribution,
    with 1-zeta mass at 0 and zeta mass at mu2
    
    Example usage: getSamples_gaussianTwoSpike(n=1000, zeta=0.1, mu2=1)
    """
    grid = np.linspace(0-padding, mu2+padding, discretization)

    mu = np.random.choice([0, mu2], size=(n,1), replace=True, p=[1-zeta, zeta])
    noise = np.random.randn(n, 1)*sigma
    observations = mu + noise
    return observations, grid


############
# Poissons
############

def getSamples_poisson(n, mixingProportions, distributionComponents, discretization=100):
    """
    Sample from a mixture of Poissons, with mixing distribution given as a sum of other distributions
    
    mixingProportions: a vector that sums to 1
    distributionComponents: a vector of "frozen" distributions which each have a method ".rvs" for generating random variates
    
    Example usage
    dist = [scipy.stats.bernoulli(1),  # The null hypothesis; mean is always 1
          scipy.stats.beta(a=6, b=8, scale=6, loc=2)]
    prop = [0.85, 0.15]
    getSamples_poisson(n=1000, mixingProportions=prop, distributionComponents=dist)
    """
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
    
    grid = np.linspace(0, max(means)*3, discretization)
    
    means = np.array(means).reshape(n,1)
    
    observations = np.random.poisson(lam=means)
    
    return observations, grid


################
# Binomials
################

def getSamples_binomial(n, mixingProportions, distributionComponents, t, discretization=100):
    """
    Sample from a mixture of binomials, with mean parameterized by the mixing distribution (and t, the number
    of trials, fixed)
    
    NOTE: The distribution components should be supported on [-1/2, 1/2], so that the "delta spike at zero" corresponds
    to a reasonable 'null hypothesis' of P(success) = 1/2. We will add 1/2 to all of the means after sampling, so they
    end up in the interval [0,1]. We take this approach because we don't know how to return delta spikes anywhere other
    than zero, using scipy's built-in distributions.
    
    mixingProportions: a vector that sums to 1
    distributionComponents: a vector of "frozen" distributions which each have a method ".rvs" for generating random variates
    
    Example usage
    dist = [scipy.stats.bernoulli(0),   # This will be the 'null hypothesis,' mean is 1/2
         scipy.stats.beta(a=4, b=8, scale=0.45, loc=0.05)]   # Alternates, means between 1/2 and 1 (after we add 1/2)
    prop = [0.8, 0.2]
    getSamples_binomial(n=1000, mixingProportions=prop, distributionComponents=dist, t=5)
    """
    
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
    
    grid = np.linspace(0, 1, discretization)
    
    # Add 1/2 to each mean, so they go from [-1/2, 1/2] to [0,1] (see function description)
    means = np.array(means).reshape(n,1) + np.full(n, 0.5).reshape(n, 1)
    
    if any(means < 0) or any(means > 1):
        print("Make sure the input distribution is in [-1/2, 1/2]; please read description of getSamples_binomials for details")
    
    observations = np.random.binomial(n=t, p=means)
    
    return observations, grid