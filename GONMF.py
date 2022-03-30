#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Orthognal Nonnegative Matrix Factorization using ADMM

@author: abderrahmane rahiche
email:   arahiche@yahoo.com
"""



import numpy as np
import scipy.sparse.linalg as spla
import time
import matplotlib.pyplot as plt
from scipy import io, isnan, linalg, sparse
import scipy
import pygsp

#from sklearn.decomposition import TruncatedSVD
#from sparsesvd import sparsesvd



def hard_thresh(X, thresh):
    '''
    A thresholding function
    '''
    thresh = 0
    negindx = X < thresh
    X[negindx] = 0
    return X



def svdInit(Y,k):
    '''
    SVD initialization
    
    '''
    U,Sigma,Vt = spla.svds(Y, k)
    print('U shape', U.shape)
    print('sigma shape', Sigma.shape)
    print('Vt shape', Vt.shape)
    # sign flip ambiguity check
    for i in range(k):
        negidx = np.where(U[:,i]<0)
        posidx = np.where(U[:,i]>=0)
        if np.linalg.norm(U[negidx, i], 'fro') > np.linalg.norm(U[posidx, i], 'fro'):
           U[:,i] = -U[:,i]

    for j in range(k):
        negidx1 = np.where(Vt[j, :]<0)
        posidx1 = np.where(Vt[j, :]>=0)
        if np.linalg.norm(Vt[j, negidx1], 'fro') > np.linalg.norm(Vt[j, posidx1], 'fro'):
           Vt[j,:] = -Vt[j,:]

    return Vt.T, U.T



def svdInit3(Y, k):

    U,S,Vt = spla.svds(Y, k)

    print('Y', Y.shape)
    print('U', U.shape)
    print('S', S.shape)
    print('Vt', Vt.shape)

    #n = len(S)
    # reverse the n first columns of u
    U1 = U[:, ::-1]
    # reverse s
    S1 = S[::-1]
    # reverse the n first rows of vt
    Vt1 = Vt[::-1, :]

    #Vt1 = np.diag(S1).dot(Vt1)
    U1= U1 @ np.diag(S1)

    return U1, Vt1




def creat_graph(data_img, num_k, sig, Verbose):
    """ 
    Creat a graph of vectorized MS images
    data_img is a matrix of d x n, with d = nbr of bands and n = nbr of pixels.
    Each pixel is a node and edges are created using similarity between pixels
    """
    # transpose the data matrix from dim x n_sample to n_sample x dim
    # data is [N x d] where N is the number of nodes in the graph and d is the dimension of the feature space

    #data = np.transpose(data_img)

    # creat the corresponding graph
    G = pygsp.graphs.NNGraph(data_img, NNtype = 'knn', k = num_k, sigma = sig,
        use_flann = True, symmetrize_type = 'average', dist_type='euclidean')

    if Verbose == 'True':
        print('Weight matrix shape: ', G.W.shape)
        v_in, v_out, weights = G.get_edge_list()
        print('edge shape', v_in.shape, v_out.shape, weights.shape)
    # Compute the differential operator
    G.compute_differential_operator()
    if Verbose == 'True':
        print('Gradient shape is: Ne ', G.Ne, ', N ', G.N)
    # Compute the Laplacian
    G.compute_laplacian('combinatorial')
    #
    if Verbose == 'True':
        print('Laplacian matrix shape is ', G.L.shape)
    return G




def UpdateX(Y, H, W, U1, rho1):
    """
    
    This function updates W by solving a least square equation AX=B
    
    """    
    HT = H.T
    HHT = H @ HT
    #
    A = HHT + rho1*np.identity(H.shape[0], dtype = np.float32)

    B = Y @ HT + rho1*(W + U1)

    ### Solve the linear equation using the scipy solver
    Wplus = spla.spsolve(A.T, B.T)
    # return the obtained solution
    return Wplus.T



def UpdateW_admm(Y, H, W, rho1, maxiter2, tol2):

    """
    
    This function updates W using the ADMM algorithm
    
    """
    X = W
    U1 = np.zeros(X.shape)
    res =[]
    
    # inner iterations
    for i in range(maxiter2):
        # Updating X
        X = UpdateX(Y, H, W, U1, rho1)

        # Updating W
        W = hard_thresh(X - U1, 0)

        R1 = W - X

        U1 += R1
        
        # To update rho
        #rho1 = 1.01*rho1

        res.append(np.linalg.norm(R1))
        if i > 1 and np.abs(res[i]-res[i-1])/res[i-1] < tol2:
               print('Tol reached')
               break


    return W, R1




def UpdateH(Y, W, L, Z, P, U2, U3, rho2, rho3, lam):
    """
    This function updates the variable H by solving a least square equation.

    """

    WT = W.T

    A1 = WT @ W + (rho2 + rho3)*np.identity(WT.shape[0], dtype = np.float32)

    B1 = 2*lam*L


    C1 = WT @ Y + rho2*(Z - U2) + rho3*(P - U3)

    if not sparse.isspmatrix_csr(A1):
        #print 'A1 not scr'
        A1 = sparse.csr_matrix(A1)

    if not sparse.isspmatrix_csr(B1):
        #print 'A1 not scr'
        B1 = sparse.csr_matrix(B1)


    Btild = np.reshape(C1, (-1, 1), order="F")

    Atild = sparse.kron(sparse.eye(B1.shape[0]), A1) + sparse.kron(B1.T, sparse.eye(A1.shape[0]))

    Hplus = sparse.linalg.minres(Atild, Btild)[0]

    Hplus = np.reshape(Hplus, C1.shape, order="F")

    return Hplus #prox_hard(Hplus)



def UpdatePsparse(aux_p):
    """
    This function update the P variable
    by calculating a close solution to ||P^{T}P - I||
    
    This function uses the sparse svds fucntion
    
    """

    U,S,Vt = spla.svds(aux_p, k=min(aux_p.shape[0], aux_p.shape[1]))

    # reverse the n first columns of u    
    U1 = U[:, ::-1]
    # reverse s
    #S1 = S[::-1]
    # reverse the n first rows of vt
    Vt1 = Vt[::-1, :]

    #U, s, Vh = linalg.svd(aux_z, full_matrices=True)
    eyep = np.eye(U1.shape[1], Vt1.shape[0])
    tmp5 = U1 @ eyep
    P = tmp5 @ Vt1

    return P #prox_hard(V5)



def UpdateP(aux_p):
    """
    This function update the P variable
    by calculating a close solution to ||P^{T}P - I||
    
    This function uses the normal svd fucntion
    
    """
    U,S,Vt =  scipy.linalg.svd(aux_p, full_matrices = False)

    P = U @ Vt
    return P #prox_hard(V5)



def UpdateH_admm(Y, W, H, L, rho2, rho3, maxiter2, tol2, lam):
    """
    
    This function updates H using the ADMM algorithm
    
    """
    Z = H
    P = H
    U2 = np.zeros(Z.shape)
    U3 = np.zeros(P.shape)

    res =[]

    # inner iterations
    for j in range(maxiter2):
        #print "H inner iteration ", j

        # Update H
        H = UpdateH(Y, W, L, Z, P, U2, U3, rho2, rho3, lam)

        # Update Z
        Z = hard_thresh(H + U2, 0)

        # Update P normal SVD
        #tp = time.time()
        P = UpdateP(H + U3)
        #print('update P', time.time() - tp)
        
        # Update P using sparse SVD
        #tp = time.time()
        #P = UpdatePsparse(H + U3)
        #print('update P', time.time() - tp)
        #print 'P shape', P.shape

        # Updating 
        R2 = H - Z
        R3 = H - P

        U2 = U2 + R2

        U3 = U3 + R3

        #rho2 = 1.01*rho2
        #rho3 = 1.01*rho3

        res.append(np.linalg.norm(R2) + np.linalg.norm(R3))
        if j > 1 and np.abs(res[j]-res[j-1])/res[j-1] < tol2:
               print('Tol reached')
               break

    return Z, R2+R3




def gonmf_auadmm_scaled(Y, nbr_components, L, rho1, rho2, rho3, lam, tol1, tol2, maxiter1, maxiter2):

    '''
    Orgthogonal graph NMF
    
    This function minimize the following objective function:
    
    1/2|| Y - WH||^2 + lam*Tr(H'LH) s.t. W>=0, H>=0, HH^T=T
    
    '''
    H, W = svdInit(Y,nbr_components)
    W = W.T
    H = H.T

    W = W.astype('float32')
    H = H.astype('float32')

    print('Y', Y.shape)
    print('W', W.shape)
    print('H', H.shape)


    res=[]
    res.append(np.inf)
    Cost =[]
    res =[]
    Orth=[]

    for i in range(maxiter1):
           print('iter... ', i)
           # Updating primal variables
           
           # Update W
           #t1= time.time()
           W, R1 = UpdateW_admm(Y, H, W, rho1, maxiter2, tol2)
           #print('W update', time.time() -t1)
           
           # Update H
           #t1= time.time()
           H, R2 = UpdateH_admm(Y, W, H, L, rho2, rho3, maxiter2, tol2, lam)
           #print('H update', time.time() -t1)

           # Calculate the reconstruction error
           # NB. The objective contains a regularization term
           Cost.append(0.5*np.linalg.norm(Y - W @ H,'fro')**2)
           
           # compute the orthogonality
           HHT = H @ H.T
           Orth.append(np.linalg.norm(HHT -np.eye(H.shape[0]),'fro')/np.linalg.norm(HHT,'fro'))

           # Compute the residual error
           res.append(np.linalg.norm(R1) + np.linalg.norm(R2))
           if (i >= 1) and (np.abs(res[i]-res[i-1])/res[i-1] < tol1):
               print('Tol reached')
               break


    return W, H, Cost, res, Orth



















