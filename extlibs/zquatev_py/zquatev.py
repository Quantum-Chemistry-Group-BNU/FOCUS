import libzquatev
import numpy

# HC=CE
def eigh(mat,iop=0):
   n = mat.shape[0]//2
   fmat = mat.T.copy() # Fortran order
   eigs = numpy.zeros(2*n,dtype=numpy.float_)
   err = libzquatev.eigh(fmat,eigs)
   fmat = fmat.T.copy() # convert back to C order
   e = eigs.copy() 
   v = fmat.copy()
   if iop == 1:
      e[0::2] = eigs[:n] 
      e[1::2] = eigs[n:]
      v[:,0::2] = fmat[:,:n]
      v[:,1::2] = fmat[:,n:]
   return e,v

# FC=SCE
def geigh(tfock,tova,debug=False):
   # canonicalization in a kramers symmetry adapted basis
   e,v = eigh(tova)
   if debug: print('eig(S)=',e)
   vcanon = v/numpy.sqrt(e)
   # transform fock matrix to orthonormal basis
   tfockmo = vcanon.T.conj().dot(tfock.dot(vcanon))
   e,v = eigh(tfockmo,iop=1)
   if debug: print('eig(Fmo)=',e)
   # back to AO basis
   v = vcanon.dot(v)
   return e,v

def check_kramers_structure(mat,thresh=1.e-8):
   print('check_kramers_structure for matrix = [A,B;C,D]')
   n = mat.shape[0]//2;
   a = mat[:n,:n]
   b = mat[:n,n:]
   c = mat[n:,:n]
   d = mat[n:,n:]
   #print(a)
   #print(b)
   #print(c)
   #print(d)
   diff1 = numpy.linalg.norm(a-a.T.conj())
   diff2 = numpy.linalg.norm(b+b.T)
   diff3 = numpy.linalg.norm(c+b.conj())
   diff4 = numpy.linalg.norm(d-a.conj())
   print('A-Ah=',diff1)
   print('B+Bt=',diff2)
   print('C+B*=',diff3)
   print('D-A*=',diff4)
   assert(diff1 < thresh and 
          diff2 < thresh and 
          diff3 < thresh and 
          diff4 < thresh)
   return 0

def solve_KR_FCSCE(mol,fock,ova,debug=False):
   n = mol.nao_2c()
   nelec = mol.nelectron
   assert(nelec%2 == 0)
   # index for kramers basis from
   # https://github.com/sunqm/pyscf/blob/master/pyscf/gto/mole.py
   trmaps = mol.time_reversal_map()
   idxA = numpy.where(trmaps > 0)[0]
   idxB = trmaps[idxA] - 1
   if fock.shape[0] == trmaps.size:
      idx = numpy.hstack((idxA,idxB))
   else:   
      idx2 = numpy.hstack((idxA,idxA+n,idxB,idxB+n))
   # {|chi>,T|chi>}
   tova = ova[numpy.ix_(idx2,idx2)].copy()
   tfock = fock[numpy.ix_(idx2,idx2)].copy()
   if debug:
      labels = mol.spinor_labels()
      print labels
      for i in range(n):
         print i,labels[i],trmaps[i]
      print('idxA=',idxA)
      print('idxB=',idxB)
      print('idx2=',idx2)
      check_kramers_structure(tova)
      check_kramers_structure(tfock)
   e,v = geigh(tfock,tova,debug)
   kmo_coeff = numpy.zeros_like(v)
   kmo_coeff[idx2,:] = v
   return e,kmo_coeff

