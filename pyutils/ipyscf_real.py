#
# Optimized version of interface for dumping integrals
# The cutted mo_coeff must be inputed directly (maybe with nfrozen).
#
import h5py
import numpy
import scipy.linalg
from pyscf import ao2mo
from pyscf.scf import hf
import functools
import time

# Provide the basic interface
class iface:
   def __init__(self):
      self.nfrozen = 0
      self.unrestricted = False
      self.mol = None
      self.mf = None
      self.ecore = None
      self.hmo = None
      self.eri = None
      self.nact = None

   # This is the central part
   def get_integral(self,mo_coeff):
       shape = mo_coeff.shape
       self.unrestricted = len(shape) == 3
       self.nact = mo_coeff.shape[1]-self.nfrozen
       print('\n[iface.get_integral] unrestricted=',self.unrestricted,
              ' nmo=',mo_coeff.shape[1],\
              ' nfrozen=',self.nfrozen,\
              ' nact=',self.nact)
       if not self.unrestricted:
          result = self.get_integral_r(mo_coeff)
       else:
          result = self.get_integral_u(mo_coeff)
       return result

   def get_integral_u(self,mo_coeff):
      t0 = time.time()
      ecore = self.mol.energy_nuc()
      mo_coeff_a = mo_coeff[0]
      mo_coeff_b = mo_coeff[1]
      # Intergrals
      mcoeffC_a = mo_coeff_a[:,:self.nfrozen].copy()
      mcoeffA_a = mo_coeff_a[:,self.nfrozen:].copy()
      mcoeffC_b = mo_coeff_b[:,:self.nfrozen].copy()
      mcoeffA_b = mo_coeff_b[:,self.nfrozen:].copy()
      hcore = self.mf.get_hcore()
      if self.nfrozen>0:
         # Core part
         pCore_a = mcoeffC_a.dot(mcoeffC_a.T)
         pCore_b = mcoeffC_b.dot(mcoeffC_b.T)
         vj,vk = hf.get_jk(self.mol,(pCore_a,pCore_b))
         fock_a = hcore + vj[0] + vj[1] - vk[0]
         fock_b = hcore + vj[0] + vj[1] - vk[1]
         hmo_a = functools.reduce(numpy.dot,(mcoeffA_a.T,fock_a,mcoeffA_a))
         hmo_b = functools.reduce(numpy.dot,(mcoeffA_b.T,fock_b,mcoeffA_b))
         ecore += 0.5*numpy.trace(pCore_a.dot(hcore+fock_a)) \
                + 0.5*numpy.trace(pCore_b.dot(hcore+fock_b))
      else:
         hmo_a = functools.reduce(numpy.dot,(mcoeffA_a.T,hcore,mcoeffA_a))
         hmo_b = functools.reduce(numpy.dot,(mcoeffA_b.T,hcore,mcoeffA_b))
      t1 = time.time()
      print(' time for heff=',t1-t0,'S')
      # Active part
      nact = mcoeffA_a.shape[1]
      eri_aaaa,eri_bbbb,eri_aabb = self.get_eri_u(mcoeffA_a,mcoeffA_b)
      t2 = time.time()
      print(' time for h2e=',t2-t0,'S')
      # save
      self.ecore = ecore
      self.hmo = (hmo_a,hmo_b)
      self.eri = (eri_aaaa,eri_bbbb,eri_aabb)
      print('finished')
      return ecore,hmo_a,hmo_b,eri_aaaa,eri_bbbb,eri_aabb

   def get_integral_r(self,mo_coeff):
      t0 = time.time()
      ecore = self.mol.energy_nuc()
      # Intergrals
      mcoeffC = mo_coeff[:,:self.nfrozen].copy()
      mcoeffA = mo_coeff[:,self.nfrozen:].copy()
      hcore = self.mf.get_hcore()
      if self.nfrozen>0:
         # Core part
         pCore = 2.0*mcoeffC.dot(mcoeffC.T)
         vj,vk = hf.get_jk(self.mol,pCore)
         fock = hcore + vj - 0.5*vk  
         hmo = functools.reduce(numpy.dot,(mcoeffA.T,fock,mcoeffA))
         ecore += 0.5*numpy.trace(pCore.dot(hcore+fock))
      else:
         hmo = functools.reduce(numpy.dot,(mcoeffA.T,hcore,mcoeffA))
      t1 = time.time()
      print(' time for heff=',t1-t0,'S')
      # Active part
      nact = mcoeffA.shape[1]
      eri = self.get_eri_r(mcoeffA)
      t2 = time.time()
      print(' time for h2e=',t2-t0,'S')
      # save
      self.ecore = ecore
      self.hmo = hmo
      self.eri = eri
      print('finished')
      return ecore,hmo,eri

   def get_eri_u(self,mcoeffA_a,mcoeffA_b):
        nact = mcoeffA_a.shape[1]
        try:
            print(' try Molecular Hamiltonian')
            eri_aaaa = ao2mo.general(self.mol,(mcoeffA_a,mcoeffA_a,mcoeffA_a,mcoeffA_a),compact=0).reshape(nact,nact,nact,nact)
            eri_bbbb = ao2mo.general(self.mol,(mcoeffA_b,mcoeffA_b,mcoeffA_b,mcoeffA_b),compact=0).reshape(nact,nact,nact,nact)
            eri_aabb = ao2mo.general(self.mol,(mcoeffA_a,mcoeffA_a,mcoeffA_b,mcoeffA_b),compact=0).reshape(nact,nact,nact,nact)
        except:
            print(' found Model Hamiltonian')
            h2 =  ao2mo.restore(1, self.mf._eri, nact)
            eri_aaaa =  numpy.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,mcoeffA_a,mcoeffA_a,mcoeffA_a,mcoeffA_a,optimize=True)
            eri_bbbb =  numpy.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,mcoeffA_b,mcoeffA_b,mcoeffA_b,mcoeffA_b,optimize=True)
            eri_aabb =  numpy.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,mcoeffA_a,mcoeffA_a,mcoeffA_b,mcoeffA_b,optimize=True)
        return eri_aaaa,eri_bbbb,eri_aabb

   def get_eri_r(self,mcoeffA):
        nact = mcoeffA.shape[1]
        try:
            print(' try Molecular Hamiltonian')
            eri = ao2mo.general(self.mol,(mcoeffA,mcoeffA,mcoeffA,mcoeffA),compact=0).reshape(nact,nact,nact,nact)
        except:
            print(' found Model Hamiltonian')
            h2 =  ao2mo.restore(1, self.mf._eri, nact)
            eri =  numpy.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,mcoeffA,mcoeffA,mcoeffA,mcoeffA,optimize=True)
        return eri

   def get_integral_FCIDUMP(self,fname='FCIDUMP'):
      print('\n[iface.get_integral_FCIDUMP] fname=',fname)
      with open(fname,'r') as f:
         line = f.readline().split(',')[0].split(' ')[-1]
         print('Num of orb: ', int(line))
         f.readline()
         f.readline()
         f.readline()
         n = int(line)
         e = 0.0
         int1e = numpy.zeros((n,n))
         int2e = numpy.zeros((n,n,n,n))
         for line in f.readlines():
           data = line.split()
           ind = [int(x)-1 for x in data[1:]]
           if ind[2] == -1 and ind[3]== -1:
             if ind[0] == -1 and ind[1] ==-1:
               e = float(data[0])
             else :
               int1e[ind[0],ind[1]] = float(data[0])
               int1e[ind[1],ind[0]] = float(data[0])
           else:
             int2e[ind[0],ind[1], ind[2], ind[3]] = float(data[0])
             int2e[ind[1],ind[0], ind[2], ind[3]] = float(data[0])
             int2e[ind[0],ind[1], ind[3], ind[2]] = float(data[0])
             int2e[ind[1],ind[0], ind[3], ind[2]] = float(data[0])
             int2e[ind[2],ind[3], ind[0], ind[1]] = float(data[0])
             int2e[ind[3],ind[2], ind[0], ind[1]] = float(data[0])
             int2e[ind[2],ind[3], ind[1], ind[0]] = float(data[0])
             int2e[ind[3],ind[2], ind[1], ind[0]] = float(data[0])
      print('finished')
      return e,int1e,int2e

   def dump(self,info,fname='mole.info'):
      print('\n[iface.dump] fname=',fname)
      if len(info) == 3:
         ecore,int1e,int2e = info
         # Spin orbital integrals
         sbas = 2*int1e.shape[0]
         h1e = numpy.zeros((sbas,sbas)) 
         h1e[0::2,0::2] = int1e # AA
         h1e[1::2,1::2] = int1e # BB
         h2e = numpy.zeros((sbas,sbas,sbas,sbas))
         h2e[0::2,0::2,0::2,0::2] = int2e # AAAA
         h2e[1::2,1::2,1::2,1::2] = int2e # BBBB
         h2e[0::2,0::2,1::2,1::2] = int2e # AABB
         h2e[1::2,1::2,0::2,0::2] = int2e # BBAA
      else:
         ecore,int1e_aa,int1e_bb,int2e_aaaa,int2e_bbbb,int2e_aabb = info
         # Spin orbital integrals
         sbas = 2*int1e_aa.shape[0]
         h1e = numpy.zeros((sbas,sbas)) 
         h1e[0::2,0::2] = int1e_aa # AA
         h1e[1::2,1::2] = int1e_bb # BB
         h2e = numpy.zeros((sbas,sbas,sbas,sbas))
         h2e[0::2,0::2,0::2,0::2] = int2e_aaaa # AAAA
         h2e[1::2,1::2,1::2,1::2] = int2e_bbbb # BBBB
         h2e[0::2,0::2,1::2,1::2] = int2e_aabb # AABB
         h2e[1::2,1::2,0::2,0::2] = int2e_aabb.transpose(2,3,0,1) # BBAA
      # antisymmetrize 
      h2e = h2e.transpose(0,2,1,3) # <ij|kl> = [ik|jl]
      h2e = h2e-h2e.transpose(0,1,3,2) # Antisymmetrize V[pqrs]=<pq||rs>=<pq|rs>-<pq|sr>
      
      # ZL@2025/12/04
      thresh = 1.e-8
      print('check symmetry of h2e with thresh=',thresh)
      diff1 = numpy.linalg.norm(h2e + h2e.transpose(1,0,2,3))
      diff2 = numpy.linalg.norm(h2e + h2e.transpose(0,1,3,2))
      diff3 = numpy.linalg.norm(h2e - h2e.transpose(2,3,0,1).conj())
      print('   diff(1,0,2,3)=',diff1)
      print('   diff(0,1,3,2)=',diff2)
      print('   diff(2,3,0,1)=',diff3)
      assert diff1 < thresh 
      assert diff2 < thresh
      assert diff3 < thresh

      self.dumpAERI(ecore,h1e,h2e,fname)
      return 0

   def dumpAERI(self,ecore,h1e,h2e,fname,thresh=1.e-16):
      sbas = h1e.shape[0]
      with open(fname,'w') as f:
         n = sbas
         line = str(n)+'\n'
         f.writelines(line)
         # int2e
         nblk = 0
         np = n*(n-1)/2
         nq = np*(np+1)/2
         for i in range(n):
            for j in range(i):
               for k in range(i+1):
                  if k == i: 	
                     lmax = j+1
                  else:
             	     lmax = k
                  for l in range(lmax):
                     nblk += 1
                     if abs(h2e[i,j,k,l])<thresh: continue
                     line = str(i+1) + ' ' \
                          + str(j+1) + ' ' \
                          + str(k+1) + ' ' \
                          + str(l+1) + ' ' \
                          + str(h2e[i,j,k,l]) \
                          + '\n'
                     f.writelines(line)
         assert nq == nblk 
         # int1e
         for i in range(n):
            for j in range(n):
               if abs(h1e[i,j])<thresh: continue
               line = str(i+1) + ' ' \
                    + str(j+1) + ' ' \
                    + '0 0 ' \
                    + str(h1e[i,j]) \
                    + '\n'
               f.writelines(line)
         # ecore 
         line = '0 0 0 0 ' + str(ecore)+'\n'
         f.writelines(line)
      print('finished')
      return 0

   # <SA*SB> = 0.5*(<SA*SB>+<SB*SA>) [symmetrized form]
   def get_integral_SiSj(self,nact,orblsti,orblstj):
       ecore = 0.0
       # 3/4*Epq*delta[pq]
       hij = numpy.zeros((nact,nact))
       for p in orblsti:
           for q in orblstj:
               if p == q: hij[p,q] = 0.75
       # -1/2*e[pqpq] - 1/4*e[pqqp]
       eri = numpy.zeros((nact,nact,nact,nact))
       # <SA*SB>
       for p in orblsti:
           for q in orblstj:
              # must use accumulation as p can equal to q (consider one orbital case!)
              eri[p,q,q,p] += -1
              eri[p,p,q,q] += -0.5
       # <SB*SA>
       for p in orblstj:
           for q in orblsti:
              # must use accumulation as p can equal to q (consider one orbital case!)
              eri[p,q,q,p] += -1
              eri[p,p,q,q] += -0.5
       eri = 0.5*eri 
       return ecore,hij,eri
   
   def dumpSiSj(self,nact,orblsti,orblstj,fname='sisj.info'):
       print('\n[iface.dumpSiSj] fname=',fname)
       print(' nact=',nact)
       print(' orblsti=',orblsti)
       print(' orblstj=',orblstj)
       info = self.get_integral_SiSj(nact,orblsti,orblstj)
       self.dump(info,fname)
       return 0
