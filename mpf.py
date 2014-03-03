'''
Minimum Probability Flow
for Ising model
Converted to Python by Kevin Ford (2014)
from original Matlab code (K_dK_ising_allbitflipextension.m) by:
Author: Jascha Sohl-Dickstein (2012)
Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
This software is made available under the Creative Commons
Attribution-Noncommercial License.
(http://creativecommons.org/licenses/by-nc/3.0/)
'''

import numpy as np


def MPFising(J, X, sym=0, allbitflip=0, l2reg=0):
    '''
    usage: MPFising( J, X, sym=0, allbitflit=0)
    performs minimum probability flow objective function to determine cost and gradient
    for solving an interaction matrix J from observations of states in X
    inputs:
    J is weight matrix of ndims x ndims that is squashed
    to ndims^2 x 1 for easier input to gradient solvers
    X is a ndims x nbatch matrix of observations
    sym option forces a symmetric J matrix (i->j == j->i)
    allbitflip option performs an all bit flip operation in addition to single bit flips
    outputs:
    (K,dK)
    K is cost (KL divergence) of MPF objective fxn at values for J
    dK is the gradient of the cost function with respect to J
    '''
    ndims=X.shape[0]
    nbatch=X.shape[1]
    
    J = J.reshape(ndims, ndims)
    if sym:
        J = (J + J.T)/2
    Y = J.dot(X)
    
    diagJ = np.diag(J)
    # XnotX contains (X - [bit flipped X])
    XnotX = 2*X-1
    
    # precompute transposes
    XT=X.T
    
    # Kfull is a ndims x nbatch matrix containing the contribution to the
    # objective function from flipping each bit in the rows, 
    # for each datapoint on the columns
    Kfull = np.exp(XnotX * Y - (1/2)*diagJ.reshape(ndims,1))
    K = Kfull.sum()
    
    lt = Kfull*XnotX
    dJ = lt.dot(XT)
    dJ = dJ - (1/2)*np.diag(Kfull.sum(1))
             
    # calculate the energies for the data states
    EX = np.sum(X*Y)
    
    
    # all bit flip state is useful for sparse data
    if allbitflip:
        # calculate the energies for the states where all bits are flipped relative to data states
        notX = 1-X
        notY = J.dot(notX)
        EnotX = np.sum(notX*notY)
        # calculate the contribution to the MPF objective function from all-bit-flipped states    
        K2full = np.exp((EX - EnotX)/2)
        K2 = np.sum(K2full)
        # calculate the gradient contribution from all-bit-flipped states
        dJ2 = (X * K2full).dot(XT)/2 - (notX * K2full).dot(notX.T)/2
        # add all-bit-flipped contributions on to full objective
        K = K + K2
        dJ = dJ + dJ2
            
    if sym:
        dJ = (dJ + dJ.T)/2 # symmetrize coupling matrix
        
    Kreg=0
    dKreg=0
    if l2reg:
        # regularization cost:
        Kreg=0.5*l2reg*((J**2).sum())
        dKreg=l2reg*J
        dKreg=dKreg.reshape(dKreg.size)

    # average over batch
    K  = Kreg + K  / nbatch
    dK = dJ.reshape(dJ.size)
    dK = dKreg + dK / nbatch
    return (K,dK)