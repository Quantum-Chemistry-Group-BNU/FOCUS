#!/usr/bin/env python
import numpy

def cc_tau(t1,t2,fac):
    t1t1 = numpy.einsum('ia,jb->ijab', t1, t1)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t2 + fac*t1t1 
    return tau1

def cc_Fpq(cc, t1, t2):
    tau1 = cc_tau(t1,t2,0.5)	
    # Foo
    Foo  = cc.eris.foo.copy()
    Foo += 0.5 * numpy.einsum('ia,ja->ij', cc.eris.fov, t1)
    Foo += numpy.einsum('ka,ikja->ij', t1, cc.eris.ooov)
    Foo += 0.5 * numpy.einsum('jkab,ikab->ij', tau1, cc.eris.oovv)
    # Fov
    Fov  = cc.eris.fov.copy()
    Fov += numpy.einsum('jb,ijab->ia', t1, cc.eris.oovv)
    # Fvv
    Fvv  = cc.eris.fvv.copy()
    Fvv -= 0.5 * numpy.einsum('ib,ia->ab', cc.eris.fov, t1)
    Fvv += numpy.einsum('ic,iacb->ab', t1, cc.eris.ovvv)
    Fvv -= 0.5 * numpy.einsum('ijac,ijbc->ab', tau1, cc.eris.oovv)
    return Foo, Fov, Fvv

def cc_Woooo(cc, t1, t2, tau2):
    Wijkl  = numpy.einsum('la,ijka->ijkl', t1, cc.eris.ooov)
    Wijkl  = Wijkl - Wijkl.transpose(0,1,3,2)
    Wijkl += cc.eris.oooo.copy()
    Wijkl += 0.25 * numpy.einsum('klab,ijab->ijkl', tau2, cc.eris.oovv)
    return Wijkl

def cc_Wvvvv(cc, t1, t2, tau2):
    Wabcd  = numpy.einsum('ib,iacd->abcd', t1, cc.eris.ovvv)
    Wabcd  = Wabcd - Wabcd.transpose(1,0,2,3)
    Wabcd += cc.eris.vvvv.copy()
    Wabcd += 0.25 * numpy.einsum('ijab,ijcd->abcd', tau2, cc.eris.oovv)
    return Wabcd

def cc_Wovvo(cc, t1, t2, tau2):
    Wiabj  = -cc.eris.ovov.transpose(0,1,3,2)
    Wiabj += numpy.einsum('jc,iabc->iabj', t1, cc.eris.ovvv)
    Wiabj += numpy.einsum('ka,ikjb->iabj', t1, cc.eris.ooov)
    tmp = 0.5*t2 + numpy.einsum('jc,ka->jkca',t1,t1)
    Wiabj -= numpy.einsum('jkca,ikbc->iabj', tmp, cc.eris.oovv)
    return Wiabj

def cc_energy(cc, t1, t2):
    tau2 = cc_tau(t1,t2,1.0)
    ecc = numpy.einsum('ia,ia', cc.eris.fov, t1) \
        + 0.25 * numpy.einsum('ijab,ijab', cc.eris.oovv, tau2)
    return ecc

def ccsd_update_t1(cc, t1, t2, Foo, Fov, Fvv):
    foo = Foo - numpy.diag(cc.eris.foo[range(cc.nocc),range(cc.nocc)])
    fvv = Fvv - numpy.diag(cc.eris.fvv[range(cc.nvir),range(cc.nvir)])
    fov = Fov
    # t1
    t1new = cc.eris.fov.copy() \
          + numpy.einsum('ib,ab->ia', t1, fvv) \
          - numpy.einsum('ja,ji->ia', t1, foo) \
          - numpy.einsum('jb,jaib->ia', t1, cc.eris.ovov)
    # t2
    t1new += numpy.einsum('jb,ijab->ia', fov, t2)
    t1new -= 0.5 * numpy.einsum('ijbc,jabc->ia', t2, cc.eris.ovvv)
    t1new += 0.5 * numpy.einsum('jkab,kjib->ia', t2, cc.eris.ooov)
    t1new /= cc.d1()
    return t1new
 
def ccsd_update_t2(cc, t1, t2, Foo, Fov, Fvv):
    foo = Foo - numpy.diag(cc.eris.foo[range(cc.nocc),range(cc.nocc)])
    fvv = Fvv - numpy.diag(cc.eris.fvv[range(cc.nvir),range(cc.nvir)])
    fov = Fov
    t2new = cc.eris.oovv.copy()
    # Pab
    ft_ab = fvv - 0.5 * numpy.einsum('ia,ib->ab', t1, fov)
    tmp1  = numpy.einsum('bc,ijac->ijab', ft_ab, t2)
    tmp1 -= numpy.einsum('ka,ijkb->ijab', t1, cc.eris.ooov)
    tmp1  = tmp1 - tmp1.transpose(0,1,3,2)
    t2new+= tmp1
    # Pij
    ft_ij = foo + 0.5 * numpy.einsum('ja,ia->ij', t1, fov)
    tmp1  = -numpy.einsum('kj,ikab->ijab', ft_ij, t2)
    tmp1 -= numpy.einsum('ic,jcab->ijab', t1, cc.eris.ovvv)
    tmp1  = tmp1 - tmp1.transpose(1,0,2,3)
    t2new+= tmp1
    # W 
    tau2 = cc_tau(t1,t2,1.0)	
    w = cc_Woooo(cc,t1,t2,tau2)
    t2new+= 0.5 * numpy.einsum('klab,klij->ijab', tau2, w)
    w = cc_Wvvvv(cc,t1,t2,tau2)
    t2new+= 0.5 * numpy.einsum('ijcd,abcd->ijab', tau2, w)
    w  = cc_Wovvo(cc,t1,t2,tau2)
    w  = numpy.einsum('ikac,kbcj->ijab', t2, w)
    w += numpy.einsum('ic,ka,kbjc->ijab', t1, t1, cc.eris.ovov)
    w = w - w.transpose(1,0,2,3)
    w = w - w.transpose(0,1,3,2)
    t2new += w
    # scale
    t2new /= cc.d2()
    return t2new

########
# CISD #
########
def ci_Fpq(cc, t1, t2):
    Foo  = cc.eris.foo.copy()
    Fov  = cc.eris.fov.copy()
    Fvv  = cc.eris.fvv.copy()
    return Foo, Fov, Fvv

def cisdHDiag(cc):
   #--- spin shift ---
   no = cc.nocc
   nv = cc.nvir
   t1 = numpy.ones((no,nv))
   t2 = numpy.ones((no,no,nv,nv))
   t1new,t2new = spinShift(t1,t2) 
   #--- end ---
   ndim = 1+cc.nocc*cc.nvir+cc.nocc*(cc.nocc-1)*cc.nvir*(cc.nvir-1)/4
   diag = numpy.zeros(ndim)
   diag[:] = cc.ehf
   e = cc.eris.fdiag
   ioff = 1
   for i in range(cc.nocc):
      for a in range(cc.nvir):
         diag[ioff] += e[cc.nocc+a]-e[i]+t1new[i,a]
	 ioff += 1
   # i>j
   for i in range(cc.nocc):
      for j in range(i):
         for a in range(cc.nvir):
	    for b in range(a):
#   # i<j
#   for i in range(cc.nocc):
#      for j in range(i+1,cc.nocc):
#         for a in range(cc.nvir):
#	    for b in range(a+1,cc.nocc):
   	       diag[ioff] += e[cc.nocc+a]+e[cc.nocc+b]-e[i]-e[j]\
			   + t2new[i,j,a,b]
	       ioff += 1
   return diag

# Shift the spin-flip states
def spinShift(t1,t2):
   fac = 1.e2
   t1new = numpy.zeros_like(t1)
   nocc = t1new.shape[0]
   ia = 0
   ib = 1
   if nocc%2 == 1: 
      ia = 1
      ib = 0
   t1new[::2,ib::2] = fac*t1[::2,ib::2]
   t1new[1::2,ia::2] = fac*t1[1::2,ia::2]
   t2new = numpy.zeros_like(t2)
   # ijab (AAAA,ABAB,BBBB,BABA) => 2+4=6
   # (ABBB) - 4
   # (BAAA) - 4
   # (AABB),(BBAA)
   t2new[::2,1::2,ib::2,ib::2]   = fac*t2[::2,1::2,ib::2,ib::2] 
   t2new[1::2,::2,ib::2,ib::2]   = fac*t2[1::2,::2,ib::2,ib::2] 
   t2new[1::2,1::2,ia::2,ib::2]  = fac*t2[1::2,1::2,ia::2,ib::2]
   t2new[1::2,1::2,ib::2,ia::2]  = fac*t2[1::2,1::2,ib::2,ia::2]
   t2new[1::2,::2,ia::2,ia::2]   = fac*t2[1::2,::2,ia::2,ia::2] 
   t2new[::2,1::2,ia::2,ia::2]   = fac*t2[::2,1::2,ia::2,ia::2] 
   t2new[::2,::2,ib::2,ia::2]    = fac*t2[::2,::2,ib::2,ia::2]  
   t2new[::2,::2,ia::2,ib::2]    = fac*t2[::2,::2,ia::2,ib::2]  
   t2new[::2,::2,ib::2,ib::2]    = fac*t2[::2,::2,ib::2,ib::2]  
   t2new[1::2,1::2,ia::2,ia::2]  = fac*t2[1::2,1::2,ia::2,ia::2]
   return t1new,t2new

# Heff = P*H*P
def cisdHVec(vec,cc):
   t0,t1,t2 = getT1T2(cc,vec)
   Foo, Fov, Fvv = ci_Fpq(cc, t1, t2)
   # H*Vec
   t1new,t2new = spinShift(t1,t2)
   t1new += cisd_t1(cc, t0, t1, t2, Foo, Fov, Fvv)
   t2new += cisd_t2(cc, t0, t1, t2, Foo, Fov, Fvv)
   hvec = putT1T2(cc,t1new,t2new)
   hvec[0] = ci_energy(cc, t1, t2)
   hvec += cc.ehf*vec
   return hvec

def getT1T2(cc,vec):
   no = cc.nocc
   nv = cc.nvir
   t0 = vec[0]
   t1 = vec[1:1+no*nv].reshape(no,nv).copy()
   ioff = 1+no*nv
   t2 = numpy.zeros((no,no,nv,nv))
   # i>j
   for i in range(cc.nocc):
      for j in range(i):
         for a in range(cc.nvir):
	    for b in range(a):
#   for i in range(cc.nocc):
#      for j in range(i+1,cc.nocc):
#         for a in range(cc.nvir):
#            for b in range(a+1,cc.nvir):
   	       t2[i,j,a,b] = vec[ioff]
   	       t2[j,i,b,a] = vec[ioff]
   	       t2[j,i,a,b] = -vec[ioff]
   	       t2[i,j,b,a] = -vec[ioff]
	       ioff += 1
   return t0,t1,t2

def putT1T2(cc,t1,t2):
   no = cc.nocc
   nv = cc.nvir
   ndim = 1+no*nv+no*(no-1)*nv*(nv-1)/4
   hvec = numpy.zeros(ndim)
   hvec[1:1+no*nv] = t1.reshape(no*nv).copy()
   ioff = cc.nocc*cc.nvir+1
   for i in range(cc.nocc):
      for j in range(i):
         for a in range(cc.nvir):
	    for b in range(a):
#   for i in range(cc.nocc):
#      for j in range(i+1,cc.nocc):
#         for a in range(cc.nvir):
#            for b in range(a+1,cc.nvir):
   	       hvec[ioff] = t2[i,j,a,b] 
	       ioff += 1
   return hvec

# <0|Hn|CI>
def ci_energy(cc, t1, t2):
    #ecc = numpy.einsum('ia,ia', cc.eris.fov, t1) \
    #    + 0.25 * numpy.einsum('ijab,ijab', cc.eris.oovv, t2)
    ecc = numpy.tensordot(cc.eris.fov,t1,axes=([0,1],[0,1])) \
        + 0.25*numpy.tensordot(cc.eris.oovv,t2,axes=([0,1,2,3],[0,1,2,3]))
    return ecc

# <T1|Hn|CI>
def cisd_t1(cc, t0, t1, t2, Foo, Fov, Fvv):
    # t1
    t1new = Fov.copy()*t0 \
    #      + numpy.einsum('ib,ab->ia', t1, Fvv) \
    #      - numpy.einsum('ja,ji->ia', t1, Foo) \
    #      - numpy.einsum('jb,jaib->ia', t1, cc.eris.ovov)
    t1new += numpy.tensordot(t1,Fvv,axes=([1],[1]))
    t1new -= numpy.tensordot(Foo,t1,axes=([0],[0]))
    t1new -= numpy.tensordot(cc.eris.ovov,t1,axes=([1,2],[1,0]))
    # t2
    #t1new += numpy.einsum('jb,ijab->ia', Fov, t2)
    #t1new -= 0.5 * numpy.einsum('ijbc,jabc->ia', t2, cc.eris.ovvv)
    #t1new += 0.5 * numpy.einsum('jkab,kjib->ia', t2, cc.eris.ooov)
    t1new += numpy.tensordot(t2,Fov,axes=([1,3],[0,1]))
    t1new -= 0.5 * numpy.tensordot(t2,cc.eris.ovvv,axes=([1,2,3],[0,2,3]))
    t1new += 0.5 * numpy.tensordot(cc.eris.ooov,t2,axes=([0,1,3],[1,0,3]))
    return t1new
 
# <T2|Hn|CI>
def cisd_t2(cc, t0, t1, t2, Foo, Fov, Fvv):
    t2new = cc.eris.oovv.copy()*t0
    #----------------------------------
    # Disconnected term - P(ij)P(ab)
    tmp1 = numpy.einsum('ia,jb->ijab',Fov,t1)
    tmp1 = tmp1-tmp1.transpose(0,1,3,2)
    tmp1 = tmp1-tmp1.transpose(1,0,2,3)
    t2new+=tmp1
    #----------------------------------
    # Pab
    tmp1  = numpy.einsum('bc,ijac->ijab', Fvv, t2)
    tmp1 -= numpy.einsum('ka,ijkb->ijab', t1, cc.eris.ooov)
    tmp1  = tmp1 - tmp1.transpose(0,1,3,2)
    t2new+= tmp1
    # Pij
    tmp1  = -numpy.einsum('kj,ikab->ijab', Foo, t2)
    tmp1 -= numpy.einsum('ic,jcab->ijab', t1, cc.eris.ovvv)
    tmp1  = tmp1 - tmp1.transpose(1,0,2,3)
    t2new+= tmp1
    # W 
    w = cc.eris.oooo.copy()
    #t2new+= 0.5 * numpy.einsum('klab,klij->ijab', t2, w)
    t2new+= 0.5 * numpy.tensordot(w,t2,axes=([0,1],[0,1]))
    w = cc.eris.vvvv.copy()
    #t2new+= 0.5 * numpy.einsum('ijcd,abcd->ijab', t2, w)
    t2new+= 0.5 * numpy.tensordot(t2,w,axes=([2,3],[2,3]))
    w = -cc.eris.ovov.transpose(0,1,3,2).copy()
    w  = numpy.einsum('ikac,kbcj->ijab', t2, w)
    w = w - w.transpose(1,0,2,3)
    w = w - w.transpose(0,1,3,2)
    t2new += w
    return t2new

###################################
# CISD - iterative update version #
###################################
def cisd_update_t1(cc, t1, t2, Foo, Fov, Fvv, ecorr):
    foo = Foo - numpy.diag(cc.eris.foo[range(cc.nocc),range(cc.nocc)])
    fvv = Fvv - numpy.diag(cc.eris.fvv[range(cc.nvir),range(cc.nvir)])
    fov = Fov.copy()
    # t1
    t1new = cc.eris.fov.copy() \
          + numpy.einsum('ib,ab->ia', t1, fvv) \
          - numpy.einsum('ja,ji->ia', t1, foo) \
          - numpy.einsum('jb,jaib->ia', t1, cc.eris.ovov)
    # t2
    t1new += numpy.einsum('jb,ijab->ia', fov, t2)
    t1new -= 0.5 * numpy.einsum('ijbc,jabc->ia', t2, cc.eris.ovvv)
    t1new += 0.5 * numpy.einsum('jkab,kjib->ia', t2, cc.eris.ooov)
    t1new /= (ecorr + cc.d1())
    return t1new
 
def cisd_update_t2(cc, t1, t2, Foo, Fov, Fvv, ecorr):
    foo = Foo - numpy.diag(cc.eris.foo[range(cc.nocc),range(cc.nocc)])
    fvv = Fvv - numpy.diag(cc.eris.fvv[range(cc.nvir),range(cc.nvir)])
    fov = Fov
    t2new = cc.eris.oovv.copy()
    #----------------------------------
    # Disconnected term - P(ij)P(ab)
    tmp1 = numpy.einsum('ia,jb->ijab',Fov,t1)
    tmp1 = tmp1-tmp1.transpose(0,1,3,2)
    tmp1 = tmp1-tmp1.transpose(1,0,2,3)
    t2new+=tmp1
    #----------------------------------
    # Pab
    tmp1  = numpy.einsum('bc,ijac->ijab', fvv, t2)
    tmp1 -= numpy.einsum('ka,ijkb->ijab', t1, cc.eris.ooov)
    tmp1  = tmp1 - tmp1.transpose(0,1,3,2)
    t2new+= tmp1
    # Pij
    tmp1  = -numpy.einsum('kj,ikab->ijab', foo, t2)
    tmp1 -= numpy.einsum('ic,jcab->ijab', t1, cc.eris.ovvv)
    tmp1  = tmp1 - tmp1.transpose(1,0,2,3)
    t2new+= tmp1
    # W 
    w = cc.eris.oooo.copy()
    t2new+= 0.5 * numpy.einsum('klab,klij->ijab', t2, w)
    w = cc.eris.vvvv.copy()
    t2new+= 0.5 * numpy.einsum('ijcd,abcd->ijab', t2, w)
    w = -cc.eris.ovov.transpose(0,1,3,2).copy()
    w  = numpy.einsum('ikac,kbcj->ijab', t2, w)
    w = w - w.transpose(1,0,2,3)
    w = w - w.transpose(0,1,3,2)
    t2new += w
    # scale
    t2new /= (ecorr + cc.d2())
    return t2new

#
# CCSD codes
#
def cc_cyclic3(t3):
    tmp = t3.copy()    
    tmp = tmp + tmp.transpose(1,2,0,3,4,5) + tmp.transpose(2,0,1,3,4,5)
    tmp = tmp + tmp.transpose(0,1,2,4,5,3) + tmp.transpose(0,1,2,5,3,4)
    return tmp

class CC(object):
    def __init__(self, nb, ne, h1e, int2e):
        self.mixing = 0.5
        self.ehf = 0.
	# (K,N)
        self.nmo = nb
        self.nocc= ne
        self.nvir= nb-ne
        # maxcycle
        self.max_cycle = 500
        # E
        self.conv_tol = 1.e-10
        # T2
        self.conv_tol_rmst2 = 1e-5
        # RESULTS
        self.emp2 = None
        self.ecisd = None
        self.eccsd = None
        self.ept = None
	self.t1 = None
        self.t2 = None
        # INTEGRALS 
        self.eris = lambda:None
        # Fpq=hpq+<pi||qi>
        self.fock = h1e.copy()
        for p in range(self.nmo):
           for q in range(self.nmo):
	      # sum over occupied spin-orbitals
              for i in range(self.nocc):
                 self.fock[p,q] += int2e[p,i,q,i]
	diff = numpy.linalg.norm(self.fock-self.fock.T)
	if diff > 1.e-10: 
           print 'Hermicity=',diff
	   exit()
	self.eris.fdiag = numpy.diag(self.fock)
        # <pq||rs>
        nocc = self.nocc
	self.eris.foo = self.fock[:nocc,:nocc].copy()
	self.eris.fov = self.fock[:nocc,nocc:].copy()
	self.eris.fvv = self.fock[nocc:,nocc:].copy()
        self.eris.oooo = int2e[:nocc,:nocc,:nocc,:nocc].copy()
	self.eris.ooov = int2e[:nocc,:nocc,:nocc,nocc:].copy()
        self.eris.oovv = int2e[:nocc,:nocc,nocc:,nocc:].copy()
        self.eris.ovov = int2e[:nocc,nocc:,:nocc,nocc:].copy()
        self.eris.ovvv = int2e[:nocc,nocc:,nocc:,nocc:].copy()
        self.eris.vvvv = int2e[nocc:,nocc:,nocc:,nocc:].copy()

    def d1(self):
        nocc = self.nocc
        nvir = self.nvir
        mo_e = self.eris.fdiag
        eia  = mo_e[:nocc,None] - mo_e[None,nocc:]
	de   = eia.reshape((nocc,nvir))
        return de

    def d2(self):
        nocc = self.nocc
        nvir = self.nvir
        mo_e = self.eris.fdiag
        eia  = mo_e[:nocc,None] - mo_e[None,nocc:]
	de = eia.reshape(-1,1) + eia.reshape(-1)
	de = de.reshape((nocc,nvir,nocc,nvir)).transpose(0,2,1,3)
        return de

    def d3(self):
        nocc = self.nocc
        nvir = self.nvir
        mo_e = self.eris.fdiag
        eia  = mo_e[:nocc,None] - mo_e[None,nocc:]
	eijab= eia.reshape(-1,1)+eia.reshape(-1)
	de = eijab.reshape(-1,1)+eia.reshape(-1)
	de = de.reshape((nocc,nvir,nocc,nvir,nocc,nvir)).transpose(0,2,4,1,3,5)
	return de

    def init_amps(self):
        nocc = self.nocc
        nvir = self.nvir
	# ovov
	de = self.d2()
        # t1[0],t2[MP2]
        self.t1 = numpy.zeros((nocc,nvir))
	self.t2 = self.eris.oovv/de
	self.emp2 = 0.25*numpy.einsum('ijab,ijab',self.t2,self.eris.oovv)
        print 'EMP2=',self.emp2
        return self.emp2, self.t1, self.t2

    def ccsd(self):
        self.init_amps()
        eold = self.emp2
        for istep in range(self.max_cycle):
            Foo, Fov, Fvv = cc_Fpq(self, self.t1, self.t2)
            t1new = ccsd_update_t1(self, self.t1, self.t2, Foo, Fov, Fvv)
            t2new = ccsd_update_t2(self, self.t1, self.t2, Foo, Fov, Fvv)
            self.t1 = self.mixing*self.t1 + (1.0-self.mixing)*t1new 
	    self.t2 = self.mixing*self.t2 + (1.0-self.mixing)*t2new
            eccsd = cc_energy(self, self.t1, self.t2)
	    print 'istep = %d, Ecorr(CCSD) = %.15g, dE = %.9g'\
                  %(istep, eccsd, eccsd - eold)
            if abs(eccsd-eold) < self.conv_tol:
                break
            eold = eccsd
	# update    
	self.eccsd = eccsd  
        return self.eccsd, self.t1, self.t2   

    def cisd(self):
        self.init_amps()
        eold = self.emp2
        for istep in range(self.max_cycle):
            Foo, Fov, Fvv = ci_Fpq(self, self.t1, self.t2)
            t1new = cisd_update_t1(self, self.t1, self.t2, Foo, Fov, Fvv, eold)
            t2new = cisd_update_t2(self, self.t1, self.t2, Foo, Fov, Fvv, eold)
            self.t1 = self.mixing*self.t1 + (1.0-self.mixing)*t1new 
	    self.t2 = self.mixing*self.t2 + (1.0-self.mixing)*t2new
            ecisd = ci_energy(self, self.t1, self.t2)
	    print 'istep = %d, Ecorr(CISD) = %.15g, dE = %.9g'\
                  %(istep, ecisd, ecisd - eold)
            if abs(ecisd-eold) < self.conv_tol:
                break
            eold = ecisd
	# update    
	self.ecisd = ecisd  
        return self.ecisd, self.t1, self.t2   
  
    def pt(self):
	print '\nPerturbative triple:'
	print 'Nocc = ',self.nocc
	print 'Nvir = ',self.nvir
	print 'Memory Usage for (T)=',1.0*self.nocc**3*self.nvir**3*8/1024**3,'G'
        # t3[ijkabc]
	tmp = -numpy.einsum('ijae,kebc->ijkabc',self.t2,self.eris.ovvv)\
   	      -numpy.einsum('imab,jkmc->ijkabc',self.t2,self.eris.ooov)
	# cyclic permutation      
	d3 = self.d3()
	t3 = cc_cyclic3(tmp)/d3
        # t3bar
	tmp = numpy.einsum('ia,jkbc->ijkabc',self.t1,self.eris.oovv)
	t3bar = cc_cyclic3(tmp)/d3
	# dEt
	t3bar = (t3bar+t3)*t3
	self.ept = 1.0/36*numpy.einsum('ijkabc,ijkabc',t3bar,d3) 
        print 'E(T)=',self.ept
	return self.ept

    def cisdEigen(self,n,ehf=0.0,enuc=0.0):
        print '\n[cisdEigen]'
	self.ehf = ehf
	#
	# E(det) = fii-<i<j||i<j>
	#
	e = 0.0
	for i in range(self.nocc):
	   e += self.eris.foo[i,i] 
	   for j in range(i):
	      e -= self.eris.oooo[i,j,i,j]
	eref = e+enuc     
	print 'E(ref)=',eref,ehf,eref-ehf
	#
	# Solve CISD equation
	#
        info = [self]
        ndim = 1+self.nocc*self.nvir\
		+self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4
        print "ndim=",ndim
	Diag = cisdHDiag(*info)
        import dvdson
        masker = dvdson.mask(info,cisdHVec)
        # Solve      
        solver = dvdson.eigenSolver()
        solver.iprt = 0
        solver.crit_e = 1.e-14
        solver.crit_vec = 1.e-8
        solver.ndim = ndim
        solver.diag = Diag
        solver.neig = n
        solver.matvec = masker.matvec
        solver.noise = True
        eigs,civec,nmvp = solver.solve_iter(v0=None,iop=4)
        eigs = eigs-ehf
	# test
        return eigs,civec


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [['H',(0.,0.,0.)],
		['F',(1.,0.,0.)]]
    mol.basis={'H':'6-31g',
	       'F':'6-31g'}
    #mol.output = 'out_h2o'
    #mol.atom = [
    #    [8 , (0. , 0.     , 0.)],
    #    [1 , (0. , -0.757 , 0.587)],
    #    [1 , (0. , 0.757  , 0.587)]]

    #mol.basis = {'H': 'cc-pvdz',
    #             'O': 'cc-pvdz'}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol=1.e-14
    rhf.scf() # -76.0267656731

    import antisymeri
    nb,ne,h1e,int2e=antisymeri.rhf_spinorb(rhf,storage=False)

    print '\nSpin-orbital CCSD'
    mcc = CC(nb,ne,h1e,int2e)
    mcc.ccsd()
    #H2O: print(mcc.eccsd - -0.21334318254)
    #pyscf:-0.0007633242657/wf:-0.000763324282
    
    pcc=cc.ccsd.CC(rhf)
    pcc.ccsd()

    
    # UHF
    uhf = scf.UHF(mol)
    uhf.conv_tol=1.e-14
    uhf.scf() # -76.0267656731
    import antisymeri
    nb,ne,h1e,int2e=antisymeri.uhf_spinorb(uhf,storage=False)
    print '\nSpin-orbital U-CCSD'
    mcc = CC(nb,ne,h1e,int2e)
    mcc.ccsd()
