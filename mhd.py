import numpy as np

def multivariate_hypergeometric_expected_vector(n,m):
    """
    Expected value of multivariate hypergeometric distribution.

    ------------PARAMETERS------------
    @param n : Number of draws.
    @param m : Vector containing the number of items in each category.

    @returns : Vector of the same length of m containing the expected values.
    ----------------------------------
    """

    m = np.asarray(m, float)
    return n * (m / m.sum())

def multivariate_hypergeometric_covariance_matrix(n,m):
    """
    Covariant matrix of multivariate hypergeometric distribution.

   ------------PARAMETERS------------
    @param n : Number of draws.
    @param m : Vector containing the number of items in each category.

    @returns : The covariance matrix of the same length of m containing the expected values.
    ----------------------------------
    """
    N = sum(m)
    gamma = n*(N-n)/(N-1)/N/N 

    F = len(m)
    cov = np.ndarray(shape=(F-1,F-1),dtype=float)
    #gamma = 1
    for i in range(F-1):
        for j in range(i+1,F-1):
            cov[i][j]=-m[i]*m[j]
            cov[j][i]=-m[i]*m[j]

    for i in range(F-1):
        cov[i][i]=m[i]*(N-m[i])
        
    return cov*gamma

def multivariate_hypergeometric_sampling(n, m, simul=1, r_seed=False):
    """
    Creating the sampling of unbaised selection process from the multivariate Sampling distribution

    ------------PARAMETERS------------
    @param simul : Number of simulations.

    @param n : Number of draws per simulation.
    @param m : Vector containing the number of items in each category.
    @returns : A list of vectors containing the number of items sampled from each category during the simulations
    ----------------------------------
    """
    if r_seed:
        #random initialization
        np.random.seed(r_seed)
    
    #the number of category
    F = len(m)
    N = sum(m)
    gamma = n*(N-n)/(N-1)/N/N 
    sampled_vects=[]
    # creating the urn with the different items to be sampled
    urn = np.repeat(np.arange(F), m)
    for k in range(simul):
        draw = np.array([urn[i] for i in np.random.permutation(len(urn))[:n]])
        sampled_vect = np.asarray([np.sum(draw == i) for i in range(F)])
        sampled_vects.append(sampled_vect)

    return sampled_vects 
