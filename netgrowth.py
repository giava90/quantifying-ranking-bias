import numpy as np
import igraph

def prefAttch(d_seq, min_p):
    """
    Sampling a node to attach using a linear preferential attachment.
  
    ------------PARAMETERS------------
    @param d_seq : A list containing the degree sequence where each index refers to a node
    @returns : The sampled node
    ----------------------------------
    """
    #we have no values for the fitness 
    mysum = sum(d_seq)
    if mysum:
        prob = [i/mysum+min_p for i in d_seq]
        prob = np.array(prob)/sum(prob)
    else:
        prob = [1/len(d_seq) for i in d_seq]
    return np.where(np.random.multinomial(1, prob)==1)[0][0]
  
#cit_net, fitness = createNetworkWoithPrefAttach(no_of_nodes, k=no_of_links)
def createNetworkWoithPrefAttach(N=1000, k=6, fit=False, min_p=5):
    """
    Creating a Graph using the preferential Attachment rule.
    On this grap it is possible to calculate Degree and PageRank.
  
    ------------PARAMETERS------------
    @param N : The number of nodes in the network
    @param k : The numer of outlinks
    @param fit : Default is False. True implies that nodes recieve fitness score uniformly distributed at random btw [0,1)
    @param min_p: Tuens the minum probability with which a node is selected. Default is min_prob = 1/N/min_p.
    @returns : A graph object of igraph
    ----------------------------------
    """
    min_prob = 1/N/min_p #i.e. 1/10 of the probability of a random case
    g=igraph.Graph(k,directed=True)
    if not fit:
        fi = []
        for i in range(k,N):
            edges = set()
            #print(g.degree(mode = "IN"))
            while(len(edges)<k):
                k_seq = g.degree(mode = "IN")
                j = prefAttch(k_seq, min_prob)
                edges.add((i,j))
            g.add_vertex(i)
            g.add_edges(list(edges))
        return (g, fit)
    else:
        fit = [np.random.sample() for i in range(k)]      
        for i in range(k,N):
            edges = set()
            while(len(edges)<k):
                w_k_seq = np.array(g.degree(mode = "IN"))*np.array(fit)
                j = prefAttch(w_k_seq,min_prob)
                edges.add((i,j))
            g.add_vertex(i)
            g.add_edges(list(edges))
            fit.append(np.random.sample())
        return (g, fit)