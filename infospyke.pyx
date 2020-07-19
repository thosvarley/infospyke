#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:15:10 2020

@author: thosvarley
"""
cimport cython
import numpy as np 
cimport numpy as np 
from libc.math cimport log2
from copy import deepcopy
"""
import pandas as pd 
import matplotlib.pyplot as plt
raster_dir = '/home/thosvarley/Documents/indiana_university/research/wenzel_anes/v1/results/rasters/'

raster = pd.read_csv(raster_dir + "raster_awake.csv",header=None).values

sparse = {"nbins":raster.shape[1]}
sparse["channels"] = {}
for i in range(raster.shape[0]):
    sparse["channels"][i] = {x for x in np.where(raster[i] == 1)[0]}

copy_sparse = deepcopy(sparse)
"""
@cython.wraparound(False)
@cython.boundscheck(False)
def raster_to_sparse(raster):
    
    cdef dict sparse = {"nbins":raster.shape[1]}
    sparse["channels"] = {}
    cdef int i
    cdef long n
    for i in range(raster.shape[0]):
        sparse["channels"][i] = {n for n in np.where(raster[i] == 1)[0]}
    
    return sparse


@cython.wraparound(False)
@cython.boundscheck(False)
def shuffle_sparse(dict sparse):
    
    cdef int nbins = sparse["nbins"]
    cdef int i
    cdef dict channels = sparse["channels"]
    cdef int N = len(channels)
    cdef list lens = [len(channels[x]) for x in range(N)]
    cdef int mx = max(lens)
    
    rand = np.tile(np.arange(nbins), (N,1))
    np.random.shuffle(rand.T)
    
    for i in range(N):
        channels[i] = { n for n in rand[i][:lens[i]] }
        
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

@cython.initializedcheck(False)
@cython.cdivision(True)
def entropy(int x, dict sparse):
    """
    The basic Shannon entropy for one channel.
    
    Arguments:
        x:
            The channel number.
        sparse:
            The sparse-data dictionary.
    
    Returns:
        The entropy of the channel. 
    """
    
    cdef double nbins = sparse["nbins"]
    cdef set X = sparse["channels"][x]
    cdef double Nx = len(X)
    
    cdef double p1_x = Nx / nbins #The probability of a spike. Small, for sparse data. 
    cdef double p0_x = 1 - p1_x #The probability of no spike. Typically close to 1.
    cdef double hx = -1*((p1_x*log2(p1_x)) + ((p0_x*log2(p0_x)))) #Sum(-p log(p))
    
    return hx

@cython.initializedcheck(False)
@cython.cdivision(True)
def joint_entropy(int x, int y, dict sparse):
    """
    Bivariate joint Shannon entropy for one channel.
    
    Arguments:
        x:
            A channel number.
        y:
            A channel number.
        sparse:
            The sparse-data dictionary.
    
    Returns:
        The joint entropy of the two channels.. 
    """    
    cdef double nbins = sparse["nbins"]
    
    cdef set X = sparse["channels"][x]
    cdef double Nx = len(X)
    cdef set Y = sparse["channels"][y]
    cdef double Ny = len(Y)
    
    cdef double hist_xy[2][2] #This will define the joint probability space.
    hist_xy[0][:] = [0., 0.] #X = 0, Y = 0 | X = 0, Y = 1
    hist_xy[1][:] = [0., 0.] #X = 1, Y = 0 | X = 1, Y = 1
    
    red_xy = X.intersection(Y) #Timestamps where the joint state of X and Y is 1, 1
    unq_x = X.difference(red_xy) #When X = 1 and Y = 0
    unq_y = Y.difference(red_xy) # When Y = 1 and X = 0
    #We take the difference between X and red_xy so we are only keeping values of X with no associated spike in Y. 
    
    hist_xy[1][1] = len(red_xy) / nbins #The probability that X=1 and Y=1 simultaniously.
    hist_xy[0][1] = len(unq_x) / nbins
    hist_xy[1][0] = len(unq_y) / nbins 
    
    #The only remaining probability is X=0, Y=0. Since that condition has no stamps, it must be inferred,
    #as the probability mass "left over" when all conditions with at least 1 spike are accounted for. 
    hist_xy[0][0] = 1 - (hist_xy[1][1] + hist_xy[0][1] + hist_xy[1][0])
        
    cdef int i, j 
    cdef double hxy = 0.0
    
    for i in range(2):
        for j in range(2):
            if hist_xy[i][j] != 0:
                hxy -= hist_xy[i][j]*log2(hist_xy[i][j]) #Once again, sum(-p log(p)). We just have more conditions now.  
    
    return hxy

@cython.initializedcheck(False)
@cython.cdivision(True)
def conditional_entropy(int x, int y, dict sparse):
    """
    H(Y|X) = H(X,Y) - H(X)
    """
    cdef double nbins = sparse["nbins"]
    
    cdef set X = sparse["channels"][x]
    cdef double Nx = len(X)
    cdef set Y = sparse["channels"][y]
    cdef double Ny = len(Y)
    
    #Calculating H(X)
    #See the entropy() function for the logic. 
    cdef double p1_x = Nx / nbins
    cdef double p0_x = 1 - p1_x
    cdef double hx = -1*((p1_x*log2(p1_x)) + ((p0_x*log2(p0_x))))
    
    #Calculating H(X,Y)
    #See the joint_entropy() function for the logic. 
    cdef double hist_xy[2][2]
    hist_xy[0][:] = [0., 0.]
    hist_xy[1][:] = [0., 0.]
    
    red_xy = X.intersection(Y)
    unq_x = X.difference(red_xy)
    unq_y = Y.difference(red_xy)    
    
    hist_xy[1][1] = len(red_xy) / nbins
    hist_xy[0][1] = len(unq_x) / nbins
    hist_xy[1][0] = len(unq_y) / nbins 
    hist_xy[0][0] = 1 - (hist_xy[1][1] + hist_xy[0][1] + hist_xy[1][0])
        
    cdef int i, j 
    cdef double hxy = 0.0
    for i in range(2):
        for j in range(2):
            if hist_xy[i][j] != 0:
                hxy -= hist_xy[i][j]*log2(hist_xy[i][j])
    
    return hxy - hx

@cython.initializedcheck(False)
@cython.cdivision(True)
def mutual_information(int x, int y, dict sparse):
    """
    I(X,Y) = H(X) + H(Y) - H(X,Y)
    """
    cdef double nbins = sparse["nbins"]
    cdef int n 
    cdef set X = sparse["channels"][x]
    cdef set Y = sparse["channels"][y]
    
    cdef double Nx = len(X)
    cdef double Ny = len(Y)
    
    cdef double p1_x = Nx / nbins
    cdef double p1_y = Ny / nbins
    cdef double p0_x = 1 - p1_x
    cdef double p0_y = 1 - p1_y
    
    cdef double hx = -1*((p1_x*log2(p1_x)) + ((p0_x*log2(p0_x))))
    cdef double hy = -1*((p1_y*log2(p1_y)) + ((p0_y*log2(p0_y))))
    
    red_xy = X.intersection(Y)
    unq_x = X.difference(red_xy)
    unq_y = Y.difference(red_xy)
    
    cdef double hist_xy[2][2]
    hist_xy[0][:] = [0., 0.]
    hist_xy[1][:] = [0., 0.]
    
    hist_xy[1][1] = len(red_xy) / nbins
    hist_xy[0][1] = len(unq_x) / nbins
    hist_xy[1][0] = len(unq_y) / nbins 
    hist_xy[0][0] = 1 - (hist_xy[1][1] + hist_xy[0][1] + hist_xy[1][0])
        
    cdef int i, j 
    cdef double hxy = 0.0
    for i in range(2):
        for j in range(2):
            if hist_xy[i][j] != 0:
                hxy -= hist_xy[i][j]*log2(hist_xy[i][j])
        
    return hx + hy - hxy

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def mutual_information_matrix(dict sparse):
    
    cdef double nbins = sparse["nbins"]
    cdef int nchannels = len(sparse["channels"])
    cdef int i, j, x, y
    cdef double p1_i, p0_i, hi
    cdef double p1_j, p0_j, hj
    cdef double hij
    
    cdef double[:,:] mat = np.zeros((nchannels, nchannels), dtype="single")
    cdef double hist_ij[2][2]
    hist_ij[0][:] = [0., 0.]
    hist_ij[1][:] = [0., 0.]
    
    for i in range(nchannels):
        
        I = sparse["channels"][i]
        p1_i = len(I) / nbins
        p0_i = 1 - p1_i
        hi = -1*((p1_i*log2(p1_i)) + (p0_i*log2(p0_i)))
        
        mat[i][i] = hi
        
        for j in range(i):
            
            J = sparse["channels"][j]
            p1_j = len(J) / nbins
            p0_j = 1 - p1_j
            hj = -1*((p1_j*log2(p1_j)) + (p0_j*log2(p0_j)))
            
            red_ij = I.intersection(J)
            unq_i = I.difference(red_ij)
            unq_j = J.difference(red_ij)
            
            hist_ij[1][1] = len(red_ij) / nbins
            hist_ij[0][1] = len(unq_i) / nbins
            hist_ij[1][0] = len(unq_j) / nbins
            hist_ij[0][0] = 1 - (hist_ij[1][1] + hist_ij[0][1] + hist_ij[1][0])
            
            hij = 0.0
           
            for x in range(2):
                for y in range(2):
                    if hist_ij[x][y] != 0:
                        hij -= hist_ij[x][y]*log2(hist_ij[x][y])
                    
            mat[i][j] = hi + hj - hij
            mat[j][i] = hi + hj - hij
    
    return mat


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def conditional_mutual_information(int x, int y, int z, dict sparse):
    """
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    """
    cdef double nbins = sparse["nbins"]
    
    X = sparse["channels"][x]
    Y = sparse["channels"][y]
    Z = sparse["channels"][z]
    
    cdef double Nz = len(Z)
    
    #Calculate H(Z)
    cdef double p1_z = Nz / nbins
    cdef double p0_z = 1 - p1_z
    cdef double hz = -1*((p1_z*log2(p1_z)) + ((p0_z*log2(p0_z))))
    
    #Calculate H(X,Y,Z)
    
    red_xyz = X.intersection(Y,Z)
    red_xy = X.intersection(Y).difference(red_xyz)
    red_xz = X.intersection(Z).difference(red_xyz)
    red_yz = Y.intersection(Z).difference(red_xyz)
    
    unq_x = X.difference(Y,Z)
    unq_y = Y.difference(X,Z)
    unq_z = Z.difference(X,Y)
    
    cdef double hist_xyz[1][8]
    hist_xyz[0][:] = [0.,0.,0.,0.,0.,0.,0.,0.]
    
    hist_xyz[0][1] = len(unq_z)
    hist_xyz[0][2] = len(unq_y)
    hist_xyz[0][3] = len(unq_x)
    hist_xyz[0][4] = len(red_xy)
    hist_xyz[0][5] = len(red_xz)  
    hist_xyz[0][6] = len(red_yz)
    hist_xyz[0][7] = len(red_xyz)    
    hist_xyz[0][0] = nbins - (hist_xyz[0][1] + hist_xyz[0][2] + hist_xyz[0][3] + hist_xyz[0][4] + hist_xyz[0][5] + hist_xyz[0][6] + hist_xyz[0][7])    
    
    cdef int i
    cdef double hxyz = 0.0
    for i in range(8):
        if hist_xyz[0][i] != 0:
            hxyz += (hist_xyz[0][i] / nbins)*(log2(hist_xyz[0][i] / nbins))
    
    hxyz *= -1
    
    #Calculate H(X,Z) & H(Y,Z)
    cdef double hist_xz[2][2]
    hist_xz[0][:] = [0., 0.]
    hist_xz[1][:] = [0., 0.]
    
    red_xz = X.intersection(Z)
    unq_x = X.difference(red_xz)
    unq_z = Z.difference(red_xz)  
    
    hist_xz[1][1] = len(red_xz)
    hist_xz[0][1] = len(unq_x)
    hist_xz[1][0] = len(unq_z) 
    hist_xz[0][0] = nbins - (hist_xz[1][1] + hist_xz[0][1] + hist_xz[1][0])
    
    cdef double hist_yz[2][2]
    hist_yz[0][:] = [0., 0.]
    hist_yz[1][:] = [0., 0.]
    
    red_yz = Y.intersection(Z)
    unq_y = Y.difference(red_yz)
    unq_z = Z.difference(red_yz)  
    
    hist_yz[1][1] = len(red_yz)
    hist_yz[0][1] = len(unq_y) 
    hist_yz[1][0] = len(unq_z) 
    hist_yz[0][0] = nbins - (hist_yz[1][1] + hist_yz[0][1] + hist_yz[1][0])
    
    cdef double hyz = 0.0
    cdef double hxz = 0.0
    for i in range(2):
        for j in range(2):
            if hist_xz[i][j] != 0:
                hxz -= (hist_xz[i][j] / nbins)*log2(hist_xz[i][j] / nbins)
                
            if hist_yz[i][j] != 0:
                hyz -= (hist_yz[i][j] / nbins)*log2(hist_yz[i][j] / nbins)

    return hxz + hyz - hxyz - hz


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def transfer_entropy(int x, int y, int lag, dict sparse):
    """
    TE(X -> Y) = I(Yt ; Xp | Yp )
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    """
    cdef double nbins = sparse["nbins"]
    cdef int n
    
    cdef set X = {n - lag for n in sparse["channels"][y]} #Here X serves as Y's past.
    cdef set Y = sparse["channels"][x] #Y serves as X
    cdef set Z = sparse["channels"][y] #Z serves as Y
    #I know, it's a nightmare, I will refactor it...eventually. 
    
    cdef double Nz = len(Z)
    
    #Calculate H(Z)
    cdef double p1_z = Nz / nbins
    cdef double p0_z = 1 - p1_z
    cdef double hz = -1*((p1_z*log2(p1_z)) + ((p0_z*log2(p0_z))))
    
    #Calculate H(X,Y,Z)
    
    red_xyz = X.intersection(Y,Z)
    red_xy = X.intersection(Y).difference(red_xyz)
    red_xz = X.intersection(Z).difference(red_xyz)
    red_yz = Y.intersection(Z).difference(red_xyz)
    
    unq_x = X.difference(Y,Z)
    unq_y = Y.difference(X,Z)
    unq_z = Z.difference(X,Y)
    
    cdef double hist_xyz[1][8]
    hist_xyz[0][:] = [0.,0.,0.,0.,0.,0.,0.,0.]
    
    hist_xyz[0][1] = len(unq_z)
    hist_xyz[0][2] = len(unq_y)
    hist_xyz[0][3] = len(unq_x)
    hist_xyz[0][4] = len(red_xy)
    hist_xyz[0][5] = len(red_xz)  
    hist_xyz[0][6] = len(red_yz)
    hist_xyz[0][7] = len(red_xyz)    
    hist_xyz[0][0] = nbins - (hist_xyz[0][1] + 
                    hist_xyz[0][2] + 
                    hist_xyz[0][3] + 
                    hist_xyz[0][4] + 
                    hist_xyz[0][5] + 
                    hist_xyz[0][6] + 
                    hist_xyz[0][7])#000  
    
    cdef int i
    cdef double hxyz = 0.0
    for i in range(8):
        if hist_xyz[0][i] != 0:
            hxyz += (hist_xyz[0][i] / nbins)*(log2(hist_xyz[0][i] / nbins))
    
    hxyz *= -1
    
    #Calculate H(X,Z) & H(Y,Z)
    cdef double hist_xz[2][2]
    hist_xz[0][:] = [0., 0.]
    hist_xz[1][:] = [0., 0.]
    
    red_xz = X.intersection(Z)
    unq_x = X.difference(red_xz)
    unq_z = Z.difference(red_xz)  
    
    hist_xz[1][1] = len(red_xz)
    hist_xz[0][1] = len(unq_x)
    hist_xz[1][0] = len(unq_z) 
    hist_xz[0][0] = nbins - (hist_xz[1][1] + hist_xz[0][1] + hist_xz[1][0])
    
    cdef double hist_yz[2][2]
    hist_yz[0][:] = [0., 0.]
    hist_yz[1][:] = [0., 0.]
    
    red_yz = Y.intersection(Z)
    unq_y = Y.difference(red_yz)
    unq_z = Z.difference(red_yz)  
    
    hist_yz[1][1] = len(red_yz)
    hist_yz[0][1] = len(unq_y) 
    hist_yz[1][0] = len(unq_z) 
    hist_yz[0][0] = nbins - (hist_yz[1][1] + hist_yz[0][1] + hist_yz[1][0])
    
    cdef double hyz = 0.0
    cdef double hxz = 0.0
    for i in range(2):
        for j in range(2):
            if hist_xz[i][j] != 0:
                hxz -= (hist_xz[i][j] / nbins)*log2(hist_xz[i][j] / nbins)
                
            if hist_yz[i][j] != 0:
                hyz -= (hist_yz[i][j] / nbins)*log2(hist_yz[i][j] / nbins)

    return hxz + hyz - hxyz - hz


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def transfer_entropy_matrix_sparse(dict sparse, int lag, bint norm=False):
    """
    Creates a TE matrix from node row_idx to node column_idx
    """
    
    cdef int N = len(sparse["channels"])
    cdef double nbins = sparse["nbins"]
    cdef int i, j
    cdef double Nj, p1_j, p0_j
    cdef double hj = 1.0
    cdef set J
    
    cdef double[:,:] mat = np.zeros((N,N))
    
    for j in range(N):   
        if norm == True:
            J = sparse["channels"][j]
            Nj = len(J)
    
            p1_j = Nj / nbins
            p0_j = 1 - p1_j
            hj = -1*((p1_j*log2(p1_j)) + ((p0_j*log2(p0_j))))
    
        for i in range(N):
            if j != i:
                mat[i][j] = transfer_entropy_sparse(i, j, lag, sparse)
                
                if norm == True:
                    mat[i][j] = mat[i][j] / hj
    
    return mat