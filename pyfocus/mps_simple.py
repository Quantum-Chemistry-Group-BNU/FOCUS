import numpy as np
import scipy
import copy
from opt_einsum import contract as einsum

# Some functions for MPS

def random(n, dcut):
    sites = [None]*n
    sites[0] = np.random.uniform(-1,1,size=(1,4,dcut))
    sites[n-1] = np.random.uniform(-1,1,size=(dcut,4,1))
    for i in range(1,n-1):
        sites[i] = np.random.uniform(-1,1,size=(dcut,4,dcut))
    return sites

def shapes(sites):
    shape = []
    for i in range(len(sites)):
        shape.append(sites[i].shape)
    return shape

def overlap(sites1,sites2):
    assert len(sites1) == len(sites2)
    env = np.ones((1,1),dtype=sites1[0].dtype)
    for i in range(len(sites1)-1,-1,-1):
        tmp = einsum('lnr,dr->lnd',sites2[i],env)
        env = einsum('lnr,mnr->lm',sites1[i].conj(),tmp)
    return env

def get_SvN(pop,thresh=1.e-100):
    s = 0.0
    for p in pop:
        if p < thresh: continue
        s += -p*np.log(p)
    return s

#=========
# Entropy
#=========
svd_driver='gesvd'

# Method1 based on density matrix 
def slstFromDM(sites, iroot=0):
    nsite = len(sites)
    sites_tmp = copy.deepcopy(sites)
    slst = []
    for i in range(1,nsite):
        if i == 1:
            site0 = sites_tmp[i-1][iroot]
            shape = site0.shape
            site0 = site0.reshape(1,shape[0],shape[1])
        else:
            site0 = sites_tmp[i-1]
        site1 = sites_tmp[i]
        psi = einsum('lnr,rmx->lnmx',site0,site1) # twodot wavefunction
        shape = psi.shape
        psi = psi.reshape(shape[0]*shape[1],shape[2]*shape[3])
        u,s,vt = scipy.linalg.svd(psi, full_matrices=False, lapack_driver=svd_driver)
        vt = einsum('l,lr->lr',s,vt)
        sites_tmp[i-1] = u.reshape(shape[0],shape[1],s.shape[0])
        sites_tmp[i] = vt.reshape(s.shape[0],shape[2],shape[3])
        slst.append(s**2)
    return slst

def bipartiteEntropy(sites,iroot=0):
    slst = slstFromDM(sites)
    return [get_SvN(x) for x in slst]

def singleSiteEntropy(sites,iroot=0):
    nsite = len(sites)
    sp = np.zeros(nsite)
    nroots = sites[0].shape[0]
    env = np.zeros((nroots,nroots),dtype=sites[0].dtype)
    env[iroot,iroot] = 1.0
    for i in range(nsite):
        tmp = einsum('lr,rnk->lnk',env,sites[i])
        env = einsum('knl,knr->lr',sites[i],tmp)
        # compute rho
        rho = einsum('lmr,lnr->mn',sites[i],tmp)
        sp[i] = get_SvN(np.diag(rho))
    return sp

def twoSiteEntropy(sites,iroot=0):
    nsite = len(sites)
    spq = np.zeros((nsite,nsite))
    nroots = sites[0].shape[0]
    env = np.zeros((nroots,nroots),dtype=sites[0].dtype)
    env[iroot,iroot] = 1.0
    for i in range(nsite):
        tmp = einsum('lr,rnd->lnd',env,sites[i])
        env = einsum('lnu,lnd->ud',sites[i],tmp)
        # compute uncontracted rho
        rho1 = einsum('lmu,lnd->mnud',sites[i],tmp)
        for j in range(i+1,nsite):
            tmp = einsum('mnud,dor->mnour',rho1,sites[j])
            rho1 = einsum('uol,mnour->mnlr',sites[j],tmp)
            # compute rho2
            rho2 = einsum('upr,mnour->mpno',sites[j],tmp)
            rho2 = rho2.reshape(16,16)
            # Note that the physical indices in CTNS are [0,2,a,b]
            #print(i,j,np.linalg.norm(rho2-rho2.T))
            e,v = scipy.linalg.eigh(rho2)
            spq[i,j] = get_SvN(e)
            spq[j,i] = spq[i,j]
    return spq

def mutualInformation(spq,sp):
    norb = sp.shape[0]
    Ipq = np.zeros_like(spq)
    for i in range(norb):
        for j in range(i):
            Ipq[i,j] = sp[i] + sp[j] - spq[i,j]
            Ipq[j,i] = Ipq[i,j]
    return Ipq

#================
# Canonical Form
#================
def checkRCF(sites,thresh=1.e-10):
    nsite = len(sites)
    diff = 0
    for i in range(nsite-1,-1,-1):
        ova = einsum('unr,dnr->ud',sites[i],sites[i])
        d = ova.shape[0]
        diff_i = np.linalg.norm(ova-np.identity(d))
        diff += diff_i
        print('check i=',i,' site.shape=',sites[i].shape,' |S-I|=',diff_i)
    if diff > nsite*thresh:
        print('MPS is not in RCF: diff=',diff)
        return 1
    else:
        print('MPS is in RCF')
        return 0

def checkLCF(sites,thresh=1.e-10):
    nsite = len(sites)
    diff = 0
    for i in range(0,nsite):
        ova = einsum('lnu,lnd->ud',sites[i],sites[i])
        d = ova.shape[0]
        diff_i = np.linalg.norm(ova-np.identity(d))
        diff += diff_i
        print('check i=',i,' site.shape=',sites[i].shape,' |S-I|=',diff_i)
    if diff > nsite*thresh:
        print('MPS is not in LCF: diff=',diff)
        return 1
    else:
        print('MPS is in LCF')
        return 0
       
# We assume MPS is only for a single state!
def leftCanonicalization(sites, dcut=-1, svd_driver='gesvd'):
    nsite = len(sites)
    sites_tmp = copy.deepcopy(sites)
    cpsi = sites_tmp[0]
    shape = cpsi.shape
    cpsi = cpsi.reshape(shape[0],1,shape[1],shape[2]) # ilnr
    for i in range(nsite-1):
        psi2 = cpsi.transpose(1,2,3,0) # ilnr->ln|ri
        shape = psi2.shape
        psi2 = psi2.reshape(shape[0]*shape[1],shape[2]*shape[3])
        u,s,vt = scipy.linalg.svd(psi2, full_matrices=False, lapack_driver=svd_driver)
        d = s.shape[0]
        u = u.reshape(shape[0],shape[1],d)
        vt = np.einsum('l,lr->lr',s,vt)
        vt = vt.reshape(d,shape[2],shape[3])
        if dcut>0: d = min(d,dcut)
        sites_tmp[i] = u[:,:,:d]
        # update cpsi for the next site
        cpsi = np.einsum('lci,cnr->ilnr',vt[:d,:,:],sites_tmp[i+1])
    # construct the last site
    shape = cpsi.shape
    assert shape[3] == 1
    cpsi = cpsi.reshape(shape[0],shape[1],shape[2])
    sites_tmp[nsite-1] = np.einsum('iln->lni',cpsi)/np.linalg.norm(cpsi)
    return sites_tmp

def rightCanonicalization(sites, dcut=-1, svd_driver='gesvd'):
    nsite = len(sites)
    sites_tmp = copy.deepcopy(sites)
    cpsi = sites_tmp[-1]
    shape = cpsi.shape
    cpsi = cpsi.reshape(shape[0], shape[1], shape[2], 1)  # lnir
    for i in range(nsite-1, 0, -1):
        psi2 = cpsi.transpose(2, 0, 1, 3)  # lnir -> il|nr
        shape = psi2.shape
        psi2 = psi2.reshape(shape[0]*shape[1], shape[2]*shape[3])
        u, s, vt = scipy.linalg.svd(psi2, full_matrices=False, lapack_driver=svd_driver)
        d = s.shape[0]
        vt = vt.reshape(d, shape[2], shape[3]) # cnr
        u = np.einsum('lr,r->lr', u, s)
        u = u.reshape(shape[0], shape[1], d) # ilc
        if dcut > 0: d = min(d, dcut)
        sites_tmp[i] = vt[:d, :, :]
        # update cpsi for the next site
        cpsi = np.einsum('lnr,irc->lnic', sites_tmp[i-1], u[:, :, :d])
    # construct the first site
    shape = cpsi.shape
    assert shape[0] == 1
    cpsi = cpsi.reshape(shape[1], shape[2], shape[3])
    sites_tmp[0] = np.einsum('nic->inc', cpsi)/np.linalg.norm(cpsi)
    return sites_tmp

def getLCFStateFromLCF(sites_lcf, iroot):
    sites_tmp = copy.deepcopy(sites_lcf)
    site = sites_tmp[-1]
    sites_tmp[-1] = site[:,:,[iroot]]
    return sites_tmp

def getLCFStateFromRCF(sites_rcf, iroot):
    sites_lcf = leftCanonicalization(sites_rcf)
    sites_tmp = copy.deepcopy(sites_lcf)
    site = sites_tmp[-1]
    sites_tmp[-1] = site[:,:,[iroot]]
    return sites_tmp

def getRCFStateFromRCF(sites_rcf, iroot):
    sites_tmp = copy.deepcopy(sites_rcf)
    site = sites_tmp[0]
    sites_tmp[0] = site[[iroot],:,:]
    return sites_tmp

# to abab ordering
def to_abab(sites,thresh=1.e-14):
    nsite = len(sites)
    sites_new = [None]*(2*nsite)
    for i in range(nsite):
        site = sites[i] # lnr (n=0,2,a,b) => (n=0,b,a,2)
        site_new = np.zeros_like(site)
        site_new[:,0,:] = site[:,0,:]
        site_new[:,1,:] = site[:,3,:]
        site_new[:,2,:] = site[:,2,:]
        site_new[:,3,:] = site[:,1,:]
        shape = site_new.shape
        assert shape[1] == 4
        site_new = site_new.reshape(shape[0]*2,2*shape[2])
        u,s,vt = scipy.linalg.svd(site_new, full_matrices=False)
        idx = np.argwhere(s>thresh).flatten()
        vtnew = vt[idx,:]
        wnew = np.einsum('ij,j->ij',u[:,idx],s[idx])
        d = len(idx)
        sites_new[2*i] = wnew.reshape(shape[0],2,d)
        sites_new[2*i+1] = vtnew.reshape(d,2,shape[2])
    return sites_new

