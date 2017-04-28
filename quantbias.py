import numpy as np

def Mahalanobis_distance(x, mu, M):
    """
    Calculating the Mahalanobis distance between x and mu, MD(x,mu) in a space with metric M.

    ------------PARAMETERS------------
    @param simul : Number of simulations.

    @param x : First vector
    @param mu : Second vector, usually containing the expected values
    @param M : The inverse of the covariance matrix
    @returns : The Mahalanobis distance between x and mu MD(x,mu)
    ----------------------------------
    """

    delta = np.array(x)-np.array(mu)
    #MD(x,mu) = (delta*M*delta)^{1/2}
    return np.sqrt(np.dot(np.dot(delta,M),delta))
  
def multivariate_hypergeometric_Mahalanobis_distance(x, mu, m):
    """
    Calculating the Mahalanobis distance between x and mu, MD(x,mu) in a metric space defined by a multivariate hypergeometric distribution defined using the vector m.

    ------------PARAMETERS------------
    @param simul : Number of simulations.

    @param x : First vector
    @param mu : Second vector, usually containing the expected values
    @param m : Vector containing the number of items in each category.
    @returns : The MD(x,mu) and a vector containg the contributions to the qaure of the MD(x,mu)
    ----------------------------------
    """
    MD=[]
    N = sum(m)
    n = sum(x)
    #if n!=sum(mu):
    #    print("ERROR: The numer of element in each ranking vector must be the same")
    #    print("Please correct")
    #    return None
    gamma = n*(N-n)/(N-1)/N/N
    
    for i in range(len(x)):
        MD.append(pow((x-mu)[i],2)/N/m[i])
        
    return np.sqrt(sum(MD)/gamma), np.array(MD)/gamma