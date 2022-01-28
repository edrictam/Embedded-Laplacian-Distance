### Import all relevant libraries
import networkx as nx
import numpy as np
import networkx.linalg.laplacianmatrix as lmat
import scipy.sparse.linalg as splinear
import scipy.stats as si

from scipy.linalg import eigh

########## Define All Core Functions Below ################

def getLaplacian(G):
    """
    Input: NetworkX graph G
    Output: The combinatorial Laplacian matrix (already converted to Numpy)
    """
    L_G = lmat.laplacian_matrix(G)
    return L_G.toarray()

def getEigen(S):
    """
    Input: Numpy array S representing the symmetric Laplacian matrix
    Output: A tuple of Numpy objects consisting of the eigenvalues and the eigenvectors
    """
    eigvals, eigvecs = eigh(S)
    return (eigvals, eigvecs)

def compute_ELD(G1, G2, K, weighted = True):
    """
    Input: Two Networkx graphs G1 and G2, as well as a hyperparameter integer K <= min(number of vertices in G1, number of vertices in G2)
    Output: The Embedded Laplacian Distance (ELD) (i.e. \rho_K(G_1, G_2) using the paper's notation)
    """
    L_G1 = getLaplacian(G1)
    L_G2 = getLaplacian(G2)

    eigval1, eigvec1 = getEigen(L_G1)
    eigval2, eigvec2 = getEigen(L_G2)

    if weighted: 
        w1_dist = [si.wasserstein_distance(eigval1[i]*eigvec1[:, i], eigval2[i]*eigvec2[:, i]) for i in range(K)]
    else: 
        w1_dist = [si.wasserstein_distance(eigvec1[:, i], eigvec2[:, i]) for i in range(K)]

    return np.sum(w1_dist)/K

def create_ELD_distance_matrix(graphList, K):
    """
    Input: A list of NetworkX graphs called graphList, as well as a hyperparameter integer K <= the number of vertices in the smallest graph in graphList
    Output: A distance matrix where entry i, j represents the ELD between graph i and graph j in graphList
    """
    dist_matrix = np.zeros([len(graphList), len(graphList)])
    for i, G1 in enumerate(graphList):
        for j, G2 in enumerate(graphList):
            if i < j:
                dist_matrix[i,j] = compute_ELD(G1, G2, K)
    dist_final = dist_matrix + dist_matrix.T
    return dist_final

########## Define the Sparse Versions of Core Functions Below ################
########## Note that sparse versions only offers speedups when the number of vertices of the graphs being compared is large (e.g. > 10000 vertices)###############

## Define all helper functions here
def getLaplacian_sparse(G):
    """
    Input: NetworkX graph G
    Output: The combinatorial Laplacian matrix (in sparse Scipy matrix format)
    """
    L_G = lmat.laplacian_matrix(G)
    return L_G

def getEigen_sparse(S, K):
    """Input: A sparse symmetric Scipy matrix S, and the positive integer K indicating the number of eigenvalues/eigenvectors to return
       Output: A tuple of Numpy objects consisting of the first K eigenvalues and the eigenvectors (sorted by smallest real part)
    """
    eigval_G, eigvec_G = splinear.eigs(S.asfptype(), K, which= "SR")
    return (np.real(eigval_G), np.real(eigvec_G))

def compute_ELD_fast(G1, G2, K, weighted = True):
    """
    Input: Two Networkx graphs G1 and G2, as well as a hyperparameter integer K <= min(number of vertices in G1, number of vertices in G2)
    Output: The Embedded Laplacian Distance (ELD) (i.e. \rho_K(G_1, G_2) using the paper's notation)
    """
    L_G1 = getLaplacian_sparse(G1)
    L_G2 = getLaplacian_sparse(G2)

    eigval1, eigvec1 = getEigen_sparse(L_G1, K)
    eigval2, eigvec2 = getEigen_sparse(L_G2, K)

    if weighted: 
        w1_dist = [si.wasserstein_distance(eigval1[i]*eigvec1[:, i], eigval2[i]*eigvec2[:, i]) for i in range(K)]
    else: 
        w1_dist = [si.wasserstein_distance(eigvec1[:, i], eigvec2[:, i]) for i in range(K)]

    return np.sum(w1_dist)/K

def create_ELD_distance_matrix_fast(graphList, K):
    """
    Input: A list of NetworkX graphs called graphList, as well as a hyperparameter integer K <= the number of vertices in the smallest graph in graphList
    Output: A distance matrix where entry i, j represents the ELD between graph i and graph j in graphList
    """
    dist_matrix = np.zeros([len(graphList), len(graphList)])
    for i, G1 in enumerate(graphList):
        for j, G2 in enumerate(graphList):
            if i < j:
                dist_matrix[i,j] = compute_ELD_fast(G1, G2, K)
    dist_final = dist_matrix + dist_matrix.T
    return dist_final
