import h5py
import numpy
import scipy.linalg
import os 
import zreleri

#
# Provide the basic interface
#
class iface:
   def __init__(self):
      self.nfrozen = 0
      self.mol = None
      self.mf = None
      #--- special for 4C ---
      self.ifgaunt = False

   # 2016.10.12: 
   # The very first version of dump4C - no localization, no reorder!
   def get_integral4C(self,mo_coeff):
      print '\n[iface.get_integral4C]'	 
      ecore = self.mol.energy_nuc()
      # The first N2C orbitals are negative energy states
      mcoeffC = mo_coeff[:,:self.nfrozen].copy()
      mcoeffA = mo_coeff[:,self.nfrozen:].copy()
      mo_coeff = numpy.hstack((mcoeffC,mcoeffA))
      # Integrals 
      h1e = self.mf.get_hcore()
      hmo = reduce(numpy.dot,(mo_coeff.T.conj(),h1e,mo_coeff))
      # Transform integrals with core+act: H2e
      erifile = 'tmperi'
      zreleri.ao2mo(self.mf, mo_coeff, erifile)
      if self.ifgaunt: zreleri.ao2mo_gaunt(self.mf, mo_coeff, erifile)
      ncore = self.nfrozen
      norb = mo_coeff.shape[1]
      with h5py.File(erifile) as f1:
	 eri = f1['ericas'].value # [ij|kl]
	 eri = eri.reshape(norb,norb,norb,norb)
      os.remove(erifile)
      # Core contribution
      if ncore > 0:
	 eriC = eri[:ncore,:ncore,:ncore,:ncore]
         # ecore = hii + 1/2<ij||ij> = hij + 1/2*([ii|jj]-[ij|ji])
         ecore1 = numpy.einsum('ii',hmo[:ncore,:ncore])\
               + 0.5*(numpy.einsum('iijj',eriC)-numpy.einsum('ijji',eriC))
         ecore += ecore1.real
         # fock_ij = hij + <ik||jk> = hij + [ij|kk]-[ik|kj]; k in core
         fock = hmo[ncore:,ncore:].copy()
         nact = norb - ncore
         for i in range(nact):
            for j in range(nact):
               for k in range(ncore):
                  fock[i,j] += eri[ncore+i,ncore+j,k,k]-eri[ncore+i,k,k,ncore+j]
         hmo = fock.copy()
         # Get antisymmetrized integrals
         eri = eri[ncore:,ncore:,ncore:,ncore:].copy()
      return ecore,hmo,eri

   def dump(self,info,fname='mole.info'):
      print '\n[iface.dump] fname=',fname
      ecore,int1e,int2e = info
      # Spin orbital integrals
      sbas = int1e.shape[0]
      h1e = int1e
      h2e = int2e
      h2e = h2e.transpose(0,2,1,3) # <ij|kl> = [ik|jl]
      h2e = h2e-h2e.transpose(0,1,3,2) # Antisymmetrize V[pqrs]=<pq||rs>
      thresh = 1.e-16
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
                          + '('+str(h2e[i,j,k,l].real)+','+str(h2e[i,j,k,l].imag)+')' \
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
                    + '('+str(h1e[i,j].real)+','+str(h1e[i,j].imag)+')' \
                    + '\n'
               f.writelines(line)
         # ecore 
         line = '0 0 0 0 ' + str(ecore)+'\n'
         f.writelines(line)
      print 'finished'
      return 0
