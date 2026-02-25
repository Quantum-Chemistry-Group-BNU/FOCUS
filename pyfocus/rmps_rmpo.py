import numpy as np
import scipy 
from opt_einsum import contract as einsum

# input is rmps,rwfun and rmpo

def gen_nalpha_mpo(nsite):
    mpo = [None]*(2*nsite)
    mpo[0] = np.zeros((1,2,2,2)) # d=2
    mpo[0][0,0] = np.identity(2)
    mpo[0][0,1] = np.array([[1,0],[0,0]])
    mpo[2*nsite-1] = np.zeros((2,1,2,2))
    mpo[2*nsite-1][1,0] = np.identity(2)
    for i in range(1,2*nsite-1):
        mpo[i] = np.zeros((2,2,2,2))
        mpo[i][0,0] = np.identity(2)
        mpo[i][1,1] = np.identity(2)
        if i%2 == 0: mpo[i][0,1] = np.array([[1,0],[0,0]])
    mpo_spatial = [None]*nsite
    for i in range(nsite):
        tmp = einsum('lrnm,rxab->lxnamb',mpo[2*i],mpo[2*i+1])
        s = tmp.shape
        # count from right
        tmp = tmp.reshape(s[0],s[1],s[2]*s[3],s[4]*s[5])
        mpo_spatial[nsite-1-i] = tmp[np.ix_(range(s[0]),range(s[1]),[0,3,1,2],[0,3,1,2])] # permute to {0,2,a,b}
    return mpo_spatial

def gen_nbeta_mpo(nsite):
    mpo = [None]*(2*nsite)
    mpo[0] = np.zeros((1,2,2,2)) # d=2
    mpo[0][0,0] = np.identity(2)
    mpo[2*nsite-1] = np.zeros((2,1,2,2))
    mpo[2*nsite-1][0,0] = np.array([[1,0],[0,0]])
    mpo[2*nsite-1][1,0] = np.identity(2)
    for i in range(1,2*nsite-1):
        mpo[i] = np.zeros((2,2,2,2))
        mpo[i][0,0] = np.identity(2)
        mpo[i][1,1] = np.identity(2)
        if i%2 == 1: mpo[i][0,1] = np.array([[1,0],[0,0]])
    mpo_spatial = [None]*nsite
    for i in range(nsite):
        tmp = einsum('lrnm,rxab->lxnamb',mpo[2*i],mpo[2*i+1])
        s = tmp.shape
        # count from right
        tmp = tmp.reshape(s[0],s[1],s[2]*s[3],s[4]*s[5])
        mpo_spatial[nsite-1-i] = tmp[np.ix_(range(s[0]),range(s[1]),[0,3,1,2],[0,3,1,2])] # permute to {0,2,a,b}
    return mpo_spatial

def compute_expectation(rmps_sites,rwfuns,mpo_spatial):
    nsite = len(rmps_sites)
    boundary = np.ones(1).reshape(1,1,1) # lnr
    for i in range(nsite):
        boundary = einsum('lnr,cdnm,amb,rdb->lca',rmps_sites[i].conj(),mpo_spatial[i],rmps_sites[i],boundary)
    value = einsum('il,lnr,jr->ij',rwfuns.conj(),boundary,rwfuns)
    return value
