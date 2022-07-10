#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:08:03 2020

@author: thosvarley
"""
#Prototype infospyke functions here!

from copy import deepcopy
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

from infospyke import mutual_information, mutual_information_matrix

sparse = np.load("sparse.npz", allow_pickle=True)["arr_0"].item()
N = sparse["nbins"]

mat = mutual_information_matrix(sparse, False)

#%%
"""
I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
"""

h_xz = joint_entropy(1,3,sparse,True)
h_yz = joint_entropy()    
h_xyz = joint_entropy3(1,2,3,sparse,True)


#%%


'''


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double _mutual_information(set X, set Y, double nbins):
    
    return _entropy(X, nbins) + _entropy(Y, nbins) - _joint_entropy(X, Y, nbins)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef dict _local_mutual_information(set X, set Y, double nbins):
    
    cdef dict local_x = _local_entropy(X, nbins)
    cdef dict local_y = _local_entropy(Y, nbins)
    cdef dict local_xy = _local_joint_entropy(X, Y, nbins)
    
    return {key : local_x[key[0]] + local_y[key[1]] - local_xy[key] for key in local_xy.keys()}


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)






def local_entropy(int x, dict sparse) -> dict:
    """
    Returns a dictionary with the local entropy values of every state.

    Parameters
    ----------
    x : int
        The channel number.
    sparse : dict
        The sparse-data dictionary.

    Returns
    -------
    dict
        The local entropy of every state x can adopt.

    """
    
    cdef double nbins = sparse["nbins"]    
    
    return _local_entropy(sparse["channels"][x], 
                    nbins)
    
@cython.initializedcheck(False)
def joint_entropy(int x, int y, dict sparse) -> double:
    """
    

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.

    Returns
    -------
    double
        DESCRIPTION.

    """
    
    cdef double nbins = sparse["nbins"]
    
    return _joint_entropy(sparse["channels"][x], 
                          sparse["channels"][y], 
                          nbins)


@cython.initializedcheck(False)
def local_joint_entropy(int x, int y, dict sparse) -> dict:
    """
    

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    
    cdef double nbins = sparse["nbins"]
    
    return _local_joint_entropy(sparse["channels"][x], 
                                sparse["channels"][y], 
                                nbins)


def local_joint_entropy3(int x, int y, int z, dict sparse) -> dict:
    """
    

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    
    cdef double nbins = sparse["nbins"]
    
    return _local_joint_entropy3(sparse["channels"][x], 
                                 sparse["channels"][y],
                                 sparse["channels"][z],
                                 nbins)

@cython.initializedcheck(False)
def conditional_entropy(int x, int y, dict sparse):
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    
    return _joint_entropy(X, Y, nbins) - _entropy(X, nbins)


@cython.initializedcheck(False)
def mutual_information(int x, int y, dict sparse):
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    
    return _mutual_information(X, Y, nbins)


@cython.initializedcheck(False)
def local_mutual_information(int x, int y, dict sparse) -> dict:
    """
    

    Parameters
    ----------
    int x : TYPE
        DESCRIPTION.
    int y : TYPE
        DESCRIPTION.
    dict sparse : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    
    return _local_mutual_information(X, Y, nbins)

@cython.initializedcheck(False)
def joint_mutual_information(int x, int y, int z, dict sparse):
    """
    Computes I(X,Y ; Z).

    Parameters
    ----------
    x : int
        The index of the source cell.
    y : int
        The index of the target cell.
    z : int
        The index of the conditioning cell.
    sparse : dict
        The sparse-raster object.

    Returns
    -------
    double
        The joint mutual information in bits.
    """
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    return _joint_entropy(X, Y, nbins) + _entropy(Z, nbins) - _joint_entropy3(X, Y, Z, nbins)


def local_joint_mutual_information(int x, int y, int z, dict sparse) -> dict:
    """
    Computes the local joint mutual information.

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

    Returns
    -------
    dict
        DESCRIPTION.

    """

    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    cdef dict local_xy = _local_joint_entropy(X, Y, nbins)
    cdef dict local_z = _local_entropy(Z, nbins)
    cdef dict local_xyz = _local_joint_entropy3(X,Y,Z,nbins)
    
    return {key : local_xy[(key[0], key[1])] + local_z[key[2]] - local_xyz[key] for key in local_xyz.keys()}

@cython.initializedcheck(False)
def co_information(int x, int y, int z, dict sparse):
    """
    Computes the triadic co-information between three neurons.
    Also known as the redundancy/synergy balance. 
    If Co(X, Y, Z) > 0, then the system is redundancy-dominated,
    If Co(X, Y, Z) > 0, then the system is synergy-dominated. 
    
    Parameters
    ----------
    x : int
        The index of the source cell.
    y : int
        The index of the target cell.
    z : int
        The index of the conditioning cell.
    sparse : dict
        The sparse-raster object.

    Returns
    -------
    double
        The conditional mutual information in bits.
    """
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    cdef double co = (_entropy(X, nbins) + _entropy(Y, nbins) + _entropy(Z, nbins) 
                      - _joint_entropy(X, Y, nbins) - _joint_entropy(X, Z, nbins) - _joint_entropy(Y, Z, nbins)
                      + _joint_entropy3(X, Y, Z, nbins))
    
    return co



@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def conditional_mutual_information(int x, int y, int z, dict sparse):
    """
    Computes the mutual information between cell x and cell y, conditioned on 
    the activity in cell z.

    Parameters
    ----------
    x : int
        The index of the source cell.
    y : int
        The index of the target cell.
    z : int
        The index of the conditioning cell.
    sparse : dict
        The sparse-raster object.

    Returns
    -------
    double
        The conditional mutual information in bits.
    """
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    cdef set Z = sparse["channels"][z]
    
    return _conditional_mutual_information(X, Y, Z, nbins)


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


@cython.initializedcheck(False)
@cython.cdivision(True)
def active_information_storage(int x, int lag, dict sparse):
    """
    5.64 ms ± 57 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    cdef int nbins_full = sparse["nbins"]
    cdef int i 
    cdef set X = {i for i in sparse["channels"][x] if i < (nbins_full - lag)}
    cdef set Y = {i-lag for i in sparse["channels"][x] if i > lag}
    cdef double nbins = nbins_full - lag
    
    return _mutual_information(X, Y, nbins)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def transfer_entropy(int x, int y, 
                     int source_lag, int target_lag, 
                     dict sparse) -> double:
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
    
    return _conditional_mutual_information(source_past, target, target_past, nbins)
    

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def transfer_entropy_matrix(dict sparse, 
                            int[:] source_lags, 
                            int[:] target_lags) -> double[:,:]:
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
'''