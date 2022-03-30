#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: abderrahmane rahiche
"""

import numpy as np
import time

import matplotlib.pyplot as plt
from scipy import io, isnan, linalg, sparse

import GONMF as GONMF

#import pygsp as pygsp





if __name__ == "__main__":

    imfile = io.loadmat('Blue_1_1-2.mat')['HSI']
    print('imfile shape', imfile.shape)
    data =np.zeros((imfile.shape[2], imfile.shape[0]*imfile.shape[1]))
    for i in range(imfile.shape[2]):
        tmp = imfile[:,:, i]/255.0
        data[i, :] = tmp.flatten()
    
    print('data type', data.dtype, data.shape)
    

    Y = data.astype('float32') 
    print('data shape', Y.shape)
    
    # Create the spacial graph
    # transpose the data matrix from dim x n_sample to n_sample x dim
    # data is [N x d] where N is the number of nodes in the graph and d is the dimension of the feature space

    num_neighbors = 8
    sigma = 0.1
    Verbose = 'False'  

    print("Creating the corresponding Graph ...")
    G1 = GONMF.creat_graph(Y.T, num_neighbors, sigma, Verbose)
    print('Done ! Ne', G1.Ne, 'N', G1.N)
    Lg = G1.L 
    print('Lg shape', Lg.shape)


    ## Optimal parameters values could be obtained using grid search method
    nbr_components = 3
    rho1 = 1e-3
    rho2 = 1e-4
    rho3 = 1e3
    lam  = 200.
    
    iter1 = 30
    iter2 = 20
    tol1 = 1.e-5
    tol2 = 1.e-5

    # Data factorization
    
    t1 = time.time()    
    W_new, H_new, err, res, orth = GONMF.gonmf_auadmm_scaled(Y, nbr_components, Lg, rho1, rho2, rho3,
                                   lam, tol1, tol2, iter1, iter2)    
    print('Done ! Elapsed time = ', time.time() - t1)
    
    # plot residual error
    plt.plot(res)
    plt.show() 
    
    
    H_new[H_new<1.e-4]=0
        
    # show a sample from the original HS images
    im = np.reshape(Y[10,:], (81, 627))
    plt.imshow(im, cmap='gray')
    plt.show()
    
    # show the abundance maps 
    Position = range(1, nbr_components+1)
    fig = plt.figure()
    for i in range(H_new.shape[0]):
            im = np.reshape(H_new[i,:], (81, 627))
            ax = fig.add_subplot(3, 1,Position[i])
            ax.imshow(im, cmap='gray')
            #plt.colorbar()
    plt.show()
    
    



