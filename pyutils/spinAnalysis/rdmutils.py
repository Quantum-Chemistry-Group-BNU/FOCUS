import numpy as np

def get_onerdm_spinorbital_matrix(fname='rdm1mps.0.0.txt'):
    mat = np.loadtxt(fname)
    return mat

def get_twordm_spinorbital_matrix(fname='rdm2mps.0.0.txt'):
    mat = np.loadtxt(fname)
    return mat

def get_twordm_spinorbital_tensor(norb,fname='rdm2mps.0.0.txt'):
    mat = np.loadtxt(fname)
    sorb = 2*norb
    twordm = np.zeros((sorb,sorb,sorb,sorb)) # rdm2[i,j,k,l] = <i^+j^+kl> (i>j,l>k in matrix)
    for i in range(sorb):
        for j in range(sorb):
            if i == j: continue
            if i>j:
                ij = i*(i-1)//2+j
                sgnij = 1
            else:
                ij = j*(j-1)//2+i
                sgnij = -1
            for k in range(sorb):
                for l in range(sorb):
                    if k == l: continue
                    if l>k:
                        lk = l*(l-1)//2+k
                        sgnlk = 1
                    else:
                        lk = k*(k-1)//2+l
                        sgnlk = -1
                    twordm[i,j,k,l] = mat[ij,lk]*sgnij*sgnlk
    return twordm

# rdm2[i,j,k,l] = <is1^+js2^+ks2ls1> = G[aaaa] + G[abba] + G[baab] + G[bbbb]
def get_twordm_spatialorbital_tensor(norb,fname='rdm2mps.0.0.txt'):
    twordm_spin = get_twordm_spinorbital_tensor(norb,fname)
    twordm = twordm_spin[::2,::2,::2,::2] \
            + twordm_spin[::2,1::2,1::2,::2] \
            + twordm_spin[1::2,::2,::2,1::2] \
            + twordm_spin[1::2,1::2,1::2,1::2]
    return twordm

# interface to oodmrg
def twordm_backtransform(twordm_spatial,urot):
    urot2 = urot.T.copy() # back to original basis
    tmp = np.einsum('ijkl,ip->pjkl',twordm_spatial,urot2)
    tmp = np.einsum('pjkl,jq->pqkl',tmp,urot2)
    tmp = np.einsum('pqkl,kr->pqrl',tmp,urot2)
    tmp = np.einsum('pqrl,ls->pqrs',tmp,urot2)
    return tmp
