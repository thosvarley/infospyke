#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:15:10 2020

@author: thosvarley
"""
cimport cython
import numpy as np 
cimport numpy as np 
from libc.math cimport log2, log
from copy import deepcopy
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import chi2


##########################################
### SPARSE OBJECT AND NULL-MODEL FUNCTIONS
##########################################

@cython.wraparound(False)
@cython.boundscheck(False)
def raster_to_sparse(np.ndarray raster):
    """
    Given a raster, computes the associated sparse-raster object. 
    In the sparse object, only the timestamps of each spike are stored. 
    Analagous to asdf format. 
    
    Parameters
    ----------
    raster : np.ndarray 
    
    Returns 
    -------
    
    dict of dicts.
    """
    
    cdef dict sparse = {"nbins":raster.shape[1]}
    sparse["channels"] = {}
    cdef int i
    cdef long n
    for i in range(raster.shape[0]):
        sparse["channels"][i] = {n for n in np.where(raster[i] == 1)[0]}
    
    return sparse


@cython.wraparound(False)
@cython.boundscheck(False)
def circular_shift_sparse(dict sparse, int mx_shift):
    """
    A null model that shifts every series in a circle mod len(series).
    Shifts are drawn from a uniform distribution on the range [1,mx_shift]
    Perfectly preserves autocorrelation while disrupting inter-element coupling.
    
    Shuffles in-place. 
    
    Parameters
    ----------
    sparse : dict
        The sparse-raster object. 
    mx_shift : int
        The maximum allowable shift. Shifts will be randomly chosen on the internal [1,mx_shift) 

    Returns
    -------
    None.

    """
    
    cdef int N = len(sparse["channels"])
    cdef double nbins = sparse["nbins"]
    cdef int[:] rand = np.random.randint(1, mx_shift, N)
    
    cdef int i 
    for i in range(N):
        sparse["channels"][i] = {(x + rand[i]) % nbins for x in sparse["channels"][i]}
    
    return None

    
@cython.wraparound(False)
@cython.boundscheck(False)
def jitter_sparse(dict sparse, int std):
    """
    This is a conservative null - each spike is "jittered" by shifting it left or right 
    according to a random draw from a Gaussian distribution with the given std.

    Jitters in-place.
    
    Parameters
    ----------
    dict sparse : sparse-raster object
        The sparse-raster object.
    int std : int
        The standard deviation of the jittering Gaussian.

    Returns
    -------
    None.

    """
    cdef int nbins = sparse["nbins"]
    cdef int x, i, j, counter, newrand
    cdef dict channels = sparse["channels"]
    cdef int N = len(channels)
    cdef list lens = [len(channels[x]) for x in range(N)]
    cdef int mx = max(lens)
    cdef int[:] rand
    cdef set channel
    
    for i in range(N): # For every neuron
        channel = channels[i] # Get the associated set of spike times. 
        rand = (std*np.random.randn(len(channel))).astype("int32") # The proposed jitter for each spike.
        counter = 0 
        for j in frozenset(channel): # For every spike
            # The condition is: the jitter cannot merge too spikes, or push them outside the time series.
            if (j + rand[counter] not in channel) and (j + rand[counter] > 0) and (j + rand[counter] < nbins):
                # If the above criteira are satisfied, add the new timestamp, remove the old one. 
                channel.add(j + rand[counter])
                channel.remove(j)
            else:
                # If the first jitter failed, keep generating jitters until you find one that works.
                newrand = int(np.random.randn()*std)
                while (j + newrand in channel) or (j + newrand < 0) or (j + newrand > nbins):
                    newrand = int(np.random.randn()*std)
                channel.add(j + newrand)
                channel.remove(j)
                
            counter += 1
    
    return None

@cython.wraparound(False)
@cython.boundscheck(False)
def sparse_raster(dict sparse):
    """
    For every timestamp, you want to subtract from it the number of "missing" stamps less than it.
    """
    
    cdef int N = len(sparse["channels"])
    cdef int i, n, k, x 
    cdef set S = set()
    cdef int num_spikes = 0
    
    for i in range(N):
        S.update(sparse["channels"][i])
        num_spikes += len(sparse["channels"][i])
    
    cdef int mx = max(S)
    cdef set total = { x for x in range(mx) }
    cdef set missing = total.difference(S)

    cdef list rows = []
    cdef list cols = []
    
    for i in range(N):
        M = {n - len({x for x in missing if x < n}) for n in sparse["channels"][i]}
       
        rows += [i for k in range(len(M))]
        cols += list(M)
    
    rows_array = np.array(rows)
    cols_array = np.array(cols)
    
    cdef int upper = cols_array.max() + 1
    
    raster = np.zeros((N, upper))
    raster[rows_array, cols_array] = 1
        
    return raster

###################################
### DISTRIBUTION COMPILER FUNCTIONS
###################################

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[:] _pdf_1d(set X, double nbins):
    
    cdef double N = len(X)
    
    cdef double[:] hist_x = np.zeros(2, dtype="double")
    hist_x[1] = N / nbins
    hist_x[0] = 1 - hist_x[1]
    
    return hist_x


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[:,:] _pdf_2d(set X, set Y, double nbins):
    
    cdef double hist_xy[2][2] #This will define the joint probability space.
    hist_xy[0][:] = [0., 0.] # |X = 0, Y = 0 | X = 0, Y = 1|
    hist_xy[1][:] = [0., 0.] # |X = 1, Y = 0 | X = 1, Y = 1|
    
    cdef set red_xy = X.intersection(Y) #Timestamps where the joint state of X and Y is 1, 1
    cdef set unq_x = X.difference(red_xy) #When X = 1 and Y = 0
    cdef set unq_y = Y.difference(red_xy) # When Y = 1 and X = 0
    #We take the difference between X and red_xy so we are only keeping values of X with no associated spike in Y. 
    
    hist_xy[1][1] = len(red_xy) / nbins #The probability that X=1 and Y=1 simultaniously.
    hist_xy[1][0] = len(unq_x) / nbins
    hist_xy[0][1] = len(unq_y) / nbins 
    
    #The only remaining probability is X=0, Y=0. Since that condition has no stamps, it must be inferred,
    #as the probability mass "left over" when all conditions with at least 1 spike are accounted for. 
    hist_xy[0][0] = 1 - (hist_xy[1][1] + hist_xy[0][1] + hist_xy[1][0])
    
    return hist_xy


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[:,:] _pdf_3d(set X, set Y, set Z, double nbins):
        
    cdef set red_xyz = X.intersection(Y,Z)
    cdef set red_xy = X.intersection(Y).difference(red_xyz)
    cdef set red_xz = X.intersection(Z).difference(red_xyz)
    cdef set red_yz = Y.intersection(Z).difference(red_xyz)
    
    cdef set unq_x = X.difference(Y,Z)
    cdef set unq_y = Y.difference(X,Z)
    cdef set unq_z = Z.difference(X,Y)
    
    cdef double hist_xyz[1][8]
    hist_xyz[0][:] = [0.,0.,0.,0.,0.,0.,0.,0.]
    
    hist_xyz[0][1] = len(unq_z) / nbins
    hist_xyz[0][2] = len(unq_y) / nbins
    hist_xyz[0][3] = len(unq_x) / nbins
    hist_xyz[0][4] = len(red_xy) / nbins
    hist_xyz[0][5] = len(red_xz) / nbins  
    hist_xyz[0][6] = len(red_yz) / nbins
    hist_xyz[0][7] = len(red_xyz) / nbins    
    hist_xyz[0][0] = 1 - (hist_xyz[0][1] + hist_xyz[0][2] + hist_xyz[0][3] + hist_xyz[0][4] + hist_xyz[0][5] + hist_xyz[0][6] + hist_xyz[0][7])    
    
    return hist_xyz

############################
### SHANON ENTROPY FUNCTIONS
############################

@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double _entropy(set X, double nbins):
    """
    """
    cdef double N = len(X)
    
    cdef double p1_x = N / nbins #The probability of a spike. Small, for sparse data. 
    cdef double p0_x = 1 - p1_x #The probability of no spike. Typically close to 1.
    
    return -1*((p1_x*log2(p1_x)) + ((p0_x*log2(p0_x))))

@cython.initializedcheck(False)
@cython.cdivision(True)
cdef dict _local_entropy(set X, double nbins):
    """
    """
    cdef double N = len(X)
    
    cdef double p1_x = N / nbins #The probability of a spike. Small, for sparse data. 
    cdef double p0_x = 1 - p1_x #The probability of no spike. Typically close to 1.
    
    return {0 : (p0_x, -log2(p0_x)) , 1 : (p1_x, -log2(p1_x))}


@cython.initializedcheck(False)
def entropy(int x, dict sparse, bint return_locals = False):
    """
    The basic Shannon entropy for one channel.
    A measure of how uncertain are you about what the next state of the neuron will be. 
    
    Parameters
    ----------
    x : int
        The channel number.
    sparse : dict
        The sparse-data dictionary.
    return_locals : bool 
        Whether to return the local values instead of the double. 
    
    Returns
    -------
        double 
            The entropy of the channel. 
        dict 
            The local entropies and the probabilities of each value. 
            Only returned if return_local == True.
    """
    
    cdef double nbins = sparse["nbins"]    
    
    if return_locals == False:
        return _entropy(sparse["channels"][x], nbins)
    else:
        return _local_entropy(sparse["channels"][x], nbins)


#####################################
### BIVARIATE JOINT-ENTROPY FUNCTIONS
#####################################

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef double _joint_entropy(set X, set Y, double nbins):
    """
    """
    cdef double[:,:] hist_xy = _pdf_2d(X, Y, nbins)
    
    cdef int i, j 
    cdef double h_xy = 0.0
    
    for i in range(2):
        for j in range(2):
            if hist_xy[i][j] != 0:
                h_xy += hist_xy[i][j]*log2(hist_xy[i][j]) #Once again, sum(-p log(p)). We just have more conditions now.  
    
    return -1*h_xy
    

@cython.boundscheck(False)
@cython.cdivision(True)
cdef dict _local_joint_entropy(set X, set Y, double nbins):
    """
    """
    cdef double[:,:] hist_xy = _pdf_2d(X, Y, nbins)
    
    return {(0,0) : (hist_xy[0][0], -log2(hist_xy[0][0])), 
            (0,1) : (hist_xy[0][1], -log2(hist_xy[0][1])),
            (1,0) : (hist_xy[1][0], -log2(hist_xy[1][0])),
            (1,1) : (hist_xy[1][1], -log2(hist_xy[1][1]))}



@cython.initializedcheck(False)
def joint_entropy(int x, int y, dict sparse, bint return_locals = False):
    """
    The basic Shannon entropy for the joint state of two channels

    Parameters
    ----------
    x : int
        The channel number.
    y : int
        The channel number.
    sparse : dict
        The sparse-raster object.
    return_locals : bool
        Whether to return the local values instead of the double. 
    
    Returns
    -------
        double 
            The entropy of the channels. 
        dict 
            The local entropies and the probabilities of each value. 
            Only returned if return_local == True.

    """
    
    cdef double nbins = sparse["nbins"]
    
    if return_locals == False:
        return _joint_entropy(sparse["channels"][x], 
                              sparse["channels"][y], 
                              nbins)
    else:
        return _local_joint_entropy(sparse["channels"][x],
                                    sparse["channels"][y],
                                    nbins)


######################################
### TRIVARIATE JOINT-ENTROPY FUNCTIONS
######################################

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _joint_entropy3(set X, set Y, set Z, double nbins):
    
    cdef double[:,:] hist_xyz = _pdf_3d(X, Y, Z, nbins)
    
    cdef double h_xyz = 0.0
    for i in range(8):
        if hist_xyz[0][i] != 0:
            h_xyz += (hist_xyz[0][i])*(log2(hist_xyz[0][i]))
    
    return -1*h_xyz


@cython.boundscheck(False)
@cython.cdivision(True)
cdef dict _local_joint_entropy3(set X, set Y, set Z, double nbins):
    
    cdef double[:,:] hist_xyz = _pdf_3d(X, Y, Z, nbins)
    
    cdef double h_xyz = 0.0
        
    return {(1,1,1) : (hist_xyz[0][7], -log2(hist_xyz[0][7])),
            (0,0,0) : (hist_xyz[0][0], -log2(hist_xyz[0][0])),
            (0,0,1) : (hist_xyz[0][1], -log2(hist_xyz[0][1])),
            (0,1,0) : (hist_xyz[0][2], -log2(hist_xyz[0][2])),
            (1,0,0) : (hist_xyz[0][3], -log2(hist_xyz[0][3])),
            (1,1,0) : (hist_xyz[0][4], -log2(hist_xyz[0][4])),
            (1,0,1) : (hist_xyz[0][5], -log2(hist_xyz[0][5])),
            (0,1,1) : (hist_xyz[0][6], -log2(hist_xyz[0][6])),
            }


@cython.initializedcheck(False)
def joint_entropy3(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    The basic Shannon entropy for the joint state of two channels

    Parameters
    ----------
    x : int
        The channel number.
    y : int
        The channel number.
    z : int 
        The channel number
    sparse : dict
        The sparse-raster object.
    return_locals : bool
        Whether to return the local values instead of the double. 
    
    Returns
    -------
        double 
            The entropy of the channels. 
        dict 
            The local entropies and the probabilities of each value. 
            Only returned if return_local == True.

    """
    
    cdef double nbins = sparse["nbins"]
    
    if return_locals == False:
        return _joint_entropy3(sparse["channels"][x], 
                               sparse["channels"][y], 
                               sparse["channels"][z],
                               nbins)
    else:
        return _local_joint_entropy3(sparse["channels"][x],
                                     sparse["channels"][y],
                                     sparse["channels"][z],
                                     nbins)


#######################
### MUTUAL INFORMATIONS
#######################

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double _mutual_information(set X, set Y, double nbins):
    
    return _entropy(X, nbins) + _entropy(Y, nbins) - _joint_entropy(X, Y, nbins)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef dict _local_mutual_information(set X, set Y, double nbins):
    
    cdef double[:] hist_x = _pdf_1d(X, nbins)
    cdef double[:] hist_y = _pdf_1d(Y, nbins)
    cdef double[:,:] hist_xy = _pdf_2d(X, Y, nbins)
    
     
    return {(0,0) : (hist_xy[0][0], log2(hist_xy[0][0] / (hist_x[0]*hist_y[0]))), 
            (0,1) : (hist_xy[0][1], log2(hist_xy[0][1] / (hist_x[0]*hist_y[1]))),
            (1,0) : (hist_xy[1][0], log2(hist_xy[1][0] / (hist_x[1]*hist_y[0]))),
            (1,1) : (hist_xy[1][1], log2(hist_xy[1][1] / (hist_x[1]*hist_y[1])))}
    
    
@cython.initializedcheck(False)
def mutual_information(int x, int y, dict sparse, bint return_locals = False):
    """
    Computes the mutual information between two series.

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.
    return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef double nbins = sparse["nbins"]
    cdef dict local_x, local_y, local_xy
    
    if return_locals == False:
        return _mutual_information(X, Y, nbins)
    else:
        return _local_mutual_information(X, Y, nbins)


cdef double _conditional_mutual_information(set X, set Y, set Z, double nbins):
    """
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    """
    
    return (_joint_entropy(X, Z, nbins) + _joint_entropy(Y, Z, nbins)
                       - _joint_entropy3(X, Y, Z, nbins) - _entropy(Z, nbins))

cdef dict _local_conditional_mutual_information(set X, set Y, set Z, double nbins):
    """
    """
    
    cdef dict h_xz = joint_entropy(1,3,sparse,True)
    cdef dict h_yz = joint_entropy(2,3, sparse,True)    
    cdef dict h_xyz = joint_entropy3(1,2,3,sparse,True)
    cdef dict h_z = entropy(3,sparse,True)
    
    return {state : (h_xyz[state][0], (h_xz[(state[0], state[2])][1] + h_yz[(state[1], state[2])][1]
                                    - h_xyz[state][1] - h_z[state[2]][1])) for state in h_xyz.keys()}


@cython.initializedcheck(False)
def conditional_mutual_information(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    Computes the mutual information between two series.

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    int z : TYPE
        DESCRIPTION
    dict sparse : TYPE
        DESCRIPTION.
    return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    cdef double nbins = sparse["nbins"]
    
    if return_locals == False:
        return _conditional_mutual_information(X, Y, Z, nbins)
    else:
        return _local_conditional_mutual_information(X, Y, Z, nbins)


@cython.initializedcheck(False)
cdef double _joint_mutual_information(set X, set Y, set Z, double nbins):
    
    return (_joint_entropy(X, Y, nbins) + _entropy(Z, nbins)
                       - _joint_entropy3(X, Y, Z, nbins))

@cython.initializedcheck(False)
cdef double _local_joint_mutual_information(set X, set Y, set Z, double nbins):
    
    cdef dict h_xy = joint_entropy(1,2,sparse,True)
    cdef dict h_xyz = joint_entropy3(1,2,3,sparse,True)
    cdef dict h_z = entropy(3,sparse,True)
    
    return {state : (h_xyz[state][0], (h_xy[(state[0], state[2])][1] + h_z[state[2]][1]
                                    - h_xyz[state][1])) for state in h_xyz.keys()}


@cython.initializedcheck(False)
def joint_mutual_information(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    int z : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.
    bint return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    cdef double nbins = sparse["nbins"]
    
    if return_locals == False:
        return _joint_mutual_information(X, Y, Z, nbins)
    else:
        return _local_joint_mutual_information(X, Y, Z, nbins)
    

@cython.initializedcheck(False)
def co_information(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    Computes the co-information between three variables.
    Redundancy/synergy biase, etc. 

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    int z : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.
    return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    cdef double nbins = sparse["nbins"]
    cdef dict local_x, local_y, local_z, local_xy, local_xz, local_yz, local_xyz
    
    if return_locals == False:
        return (_entropy(X, nbins) + _entropy(Y, nbins) + _entropy(Z, nbins)
                - _joint_entropy(X, Y, nbins) - _joint_entropy(X, Z, nbins) - _joint_entropy(Y, Z, nbins)
                + _joint_entropy3(X, Y, Z, nbins))
    else:
        local_x = _local_entropy(X, nbins)
        local_y = _local_entropy(Y, nbins) 
        local_z = _local_entropy(Z, nbins)
        local_xy  = _local_joint_entropy(X,Y, nbins)
        local_xz  = _local_joint_entropy(X,Z, nbins) 
        local_yz  = _local_joint_entropy(Y,Z, nbins) 
        local_xyz = _local_joint_entropy3(X,Y,Z, nbins)
        
        return {key : (local_xyz[key][0], (local_x[key[0]][1] + local_y[key[1]][1] + local_z[key[2]][1]
                -local_xy[(key[0], key[1])][1] - local_xz[(key[0], key[2])][1] - local_yz[(key[1],key[2])][1]
                + local_xyz[key][1])) for key in local_xyz.keys()}


@cython.initializedcheck(False)
def total_correlation(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    TC(X) = H(X1) + H(X2) + H(X3) - H(X1, X2, X3)

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    int z : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.
    bint return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    cdef double nbins = sparse["nbins"]
    cdef dict local_x, local_y, local_z, local_xyz
    
    if return_locals == False:
        return _entropy(X, nbins) + _entropy(Y, nbins) + _entropy(Z, nbins) - _joint_entropy3(X, Y, Z, nbins)
    else:
        local_x = _local_entropy(X, nbins)
        local_y = _local_entropy(Y, nbins) 
        local_z = _local_entropy(Z, nbins)
        local_xyz = _local_joint_entropy3(X, Y, Z, nbins)
        
        return {key : (local_xyz[key][0], (local_x[key[0]][1] + local_y[key[1]][1] + local_z[key[2]][1]
                - local_xyz[key][1])) for key in local_xyz.keys()}
    

@cython.initializedcheck(False)
def dual_total_correlation(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    DTC(X) = H(X) - (H(X1 | X2,X3) + H(X2 | X1, X3) + H(X3 | X1, X2))

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    int z : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.
    bint return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    cdef double nbins = sparse["nbins"]
    cdef dict local_xy, local_xz, local_yz, local_xyz
    cdef double h_xyz
    
    if return_locals == False:
        h_xyz = _joint_entropy3(X, Y, Z, nbins)
        return h_xyz + ( #The joint entropy
            (_joint_entropy(X, Y, nbins) - h_xyz) # Conditional, leave-1-out entropies
            + (_joint_entropy(X, Z, nbins) - h_xyz)
            + (_joint_entropy(Y, Z, nbins) - h_xyz)
            )
    else:
        local_xy = _local_joint_entropy(X, Y, nbins)
        local_xz = _local_joint_entropy(X, Z, nbins) 
        local_yz = _local_joint_entropy(Y, Z, nbins)
        local_xyz = _local_joint_entropy3(X, Y, Z, nbins)
        
        return {key : (local_xyz[key][0], 
                       (local_xyz[key][1] + ((local_xy[(key[0], key[1])][1] - local_xyz[key][1]) 
                                             + (local_xz[(key[0], key[2])][1] - local_xyz[key][1])
                                             + (local_yz[(key[1],key[2])][1] - local_xyz[key][1])
                                             )
                        )
                       ) for key in local_xyz.keys()}
    
    
@cython.initializedcheck(False)
def joint_mutual_information(int x, int y, int z, dict sparse, bint return_locals = False):
    """
    

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    int z : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.
    bint return_locals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    cdef double nbins = sparse["nbins"]
    
    if return_locals == False:
        return _joint_entropy(X, Y, nbins) + _entropy(Z, nbins) - _joint_entropy3(X, Y, Z, nbins)
    else:
        local_xy = _local_joint_entropy(X, Y, nbins)
        local_z = _local_entropy(Z, nbins)
        local_xyz = _local_joint_entropy3(X, Y, Z, nbins)        
        
        return {key : (local_xyz[key][0], 
                       (local_xy[(key[0], key[1])][1] + local_z[key[2]][1] - local_xyz[key][1])
                       ) for key in local_xyz.keys()}


########################
### INFORMATION DYNAMICS
########################


@cython.initializedcheck(False)
@cython.cdivision(True)
def active_information_storage(int x, int lag, dict sparse, bint return_values=False):
    """
    6.17 ms ± 87.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    cdef int nbins_full = sparse["nbins"]
    cdef int i 
    cdef set X = {i for i in sparse["channels"][x] if i < (nbins_full - lag)}
    cdef set Y = {i-lag for i in sparse["channels"][x] if i > lag}
    cdef double nbins = nbins_full - lag
    
    if return_values == False:
        return _mutual_information(X, Y, nbins)
    else:
        return _local_mutual_information(X, Y, nbins)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def transfer_entropy(int x, int y, 
                     int source_lag, int target_lag, 
                     dict sparse, bint return_locals):
    """
    Computes the transfer entropy from a source neuron x and the target neuron y.

    Parameters
    ----------
    x : int
        The index of the source cell.
    y : int
        The index of the target cell.
    source_lag : int
        The lag between the source and the target's future.
    target_lag : int
        The lag between the target's past and future.
    sparse : dict
        The sparse-raster object.

    Returns
    -------
    double
        The transfer entropy in bit.
    """
    
    cdef double nbins = sparse["nbins"]
    
    cdef set source = sparse["channels"][x]
    cdef set target = sparse["channels"][y]
    
    cdef set source_past = {x + source_lag for x in source if x + source_lag <= nbins}
    cdef set target_past = {x + target_lag for x in target if x + target_lag <= nbins}
    
    if return_locals == False:
        return _conditional_mutual_information(source_past, target, target_past, nbins)
    else:
        return _local_conditional_mutual_information(source_past, target, target_past, nbins)


###############################
### FC and EC NETWORK INFERENCE
###############################

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def mutual_information_matrix(dict sparse, bint norm = False):
    """
    Given a sparse data dictionary, calculates the pairwise mutual information for each set of two channels. 


    Parameters
    ----------
    sparse : dict
        The sparse-raster object. 
    norm : bool, optional
        If True, the MI is divided by the joint entropy to give a value
        bounded between 0-1. The default is False.

    Returns
    -------
    double[:,:]
        The symmetric, mutual information matrix.
    """

    cdef double nbins = sparse["nbins"]
    cdef int nchannels = len(sparse["channels"])    
    cdef set X, Y
    cdef double hx, hy, h_joint, mi
    cdef int i, j
    cdef double[:,:] mat = np.zeros((nchannels, nchannels), dtype="double")
    
    # Stores the entropies of each y so we don't have to recompute them 
    # over and over again.
    cdef dict cache = {i : None for i in range(nchannels)}
    
    for i in range(nchannels):
        X = sparse["channels"][i]
        if len(X) > 0:
            hx = _entropy(X, nbins)
            
            for j in range(i):
                
                Y = sparse["channels"][j]
                if len(Y) > 0:
                    if cache[j] is None:
                        hy = _entropy(Y, nbins)
                        cache[j] = hy
                    else:
                        hy = cache[j]
                    
                    h_joint = _joint_entropy(X, Y, nbins)
                    
                    if norm == False:
                        mi = hx + hy - h_joint
                    else:
                        mi = (hx + hy - h_joint) / h_joint
                    
                    mat[i][j] = mi
                    mat[j][i] = mi
            
    return np.array(mat)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def mutual_information_matrix(dict sparse, bint norm = False):
    """
    Given a sparse data dictionary, calculates the pairwise mutual information for each set of two channels. 


    Parameters
    ----------
    sparse : dict
        The sparse-raster object. 
    norm : bool, optional
        If True, the MI is divided by the joint entropy to give a value
        bounded between 0-1. The default is False.

    Returns
    -------
    double[:,:]
        The symmetric, mutual information matrix.
    """

    cdef double nbins = sparse["nbins"]
    cdef int nchannels = len(sparse["channels"])    
    cdef set X, Y
    cdef double hx, hy, h_joint, mi
    cdef int i, j
    cdef double[:,:] mat = np.zeros((nchannels, nchannels), dtype="double")
    
    # Stores the entropies of each y so we don't have to recompute them 
    # over and over again.
    cdef dict cache = {i : None for i in range(nchannels)}
    
    for i in range(nchannels):
        X = sparse["channels"][i]
        if len(X) > 0:
            hx = _entropy(X, nbins)
            
            for j in range(i):
                
                Y = sparse["channels"][j]
                if len(Y) > 0:
                    if cache[j] is None:
                        hy = _entropy(Y, nbins)
                        cache[j] = hy
                    else:
                        hy = cache[j]
                    
                    h_joint = _joint_entropy(X, Y, nbins)
                    
                    if norm == False:
                        mi = hx + hy - h_joint
                    else:
                        mi = (hx + hy - h_joint) / h_joint
                    
                    mat[i][j] = mi
                    mat[j][i] = mi
            
    return np.array(mat)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def transfer_entropy_matrix(dict sparse, 
                            int[:] source_lags, 
                            int[:] target_lags):
    """
    

    Parameters
    ----------
    dict sparse : TYPE
        DESCRIPTION.
    int[ : ] source_lags
        DESCRIPTION.
    int[ : ] target_lags
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    cdef double nbins = sparse["nbins"]
    cdef int nchannels = len(sparse["channels"])
    
    cdef double[:,:] mat = np.zeros((nchannels, nchannels))
    
    cdef int i, j, source_lag, target_lag 
    cdef double te, h_tp, h_tp_tf
    cdef set source_past, target_past
    
    cdef dict cache_h_tp = {i : None for i in range(nchannels)}
    cdef dict cache_h_tp_tf = {i : None for i in range(nchannels)}
    
    for i in range(nchannels):
        
        source = sparse["channels"][i]
        if len(source) > 0:
            
            source_lag = source_lags[i]
            source_past = {x + source_lag for x in source if x + source_lag <= nbins}
            
            for j in range(nchannels):
                if i != j:
                    target = sparse["channels"][j]
                    if len(target) > 0:
                        target_lag = target_lags[j]
                        target_past = {x + target_lag for x in target if x + target_lag <= nbins}
                        
                        if cache_h_tp[j] is None:

                            h_tp = _entropy(target_past, nbins)
                            cache_h_tp[j] = h_tp
                            
                            h_tp_tf = _joint_entropy(target_past, target, nbins)
                            cache_h_tp_tf[j] = h_tp_tf
                            
                        else:
                            h_tp = cache_h_tp[j]
                            h_tp_tf = cache_h_tp_tf[j]
                        
                        te = (_joint_entropy(source_past, target_past, nbins)
                              + h_tp_tf
                              - _joint_entropy3(source_past, target, target_past, nbins)
                              - h_tp)
                        
                        mat[i][j] = te

    return np.array(mat)     


#################
### ANALYTIC NULL
#################

@cython.cdivision(True)
@cython.initializedcheck(False)
def analytic_null(double estimate, double nbins) -> (double, double):
    """
    Computes the analytic null for the MI between two discrete, 
    binary spiking processes. 
    
    If X and Y are independent, I(X;Y) is distributed according to 
    a Chi^2 distribution w/ d.f. = 1, divided by a constant log(2)*N*2.0,
    Where n is the number of observations. 
    
    This is only correct in the asymptoptic limit. For small nbins
    the estimator doesn't do well. Similarly, it works best when 
    the entropy of the series is maximal - highly skewed series take 
    longer to converge. 
    
    Use with caution.
    
    See https://arxiv.org/pdf/1408.3270.pdf Appendix A5
    
    Parameters
    ----------
    estimate : double
        The empirical MI
    nbins : double
        The number of bins in the time series.

    Returns
    -------
    double
        The expected mean of the null distribution.
    double
        The pvalue of the estimate.
    """
    
    cdef double C = log(2)*nbins*2.0 # The normalizing fact. 
    cdef double mean = chi2.mean(df=1) / C # Mean of the Chi2 dist. 
    cdef double pval = 1-chi2.cdf(estimate*C, df=1) # Compute pval w/ CDF. 
    
    return (mean, pval)

sparse = np.load("sparse.npz", allow_pickle=True)["arr_0"].item()
