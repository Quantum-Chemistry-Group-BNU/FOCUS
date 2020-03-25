import numpy
import scipy.linalg
import h5py
import diis_class
import tools_io

def get_SpinMat(k):
   assert k%2 == 0
   sx = numpy.zeros((k,k))
   sy = numpy.zeros((k,k)) # (-iSy)
   sz = numpy.zeros((k,k))
   for i in range(k/2):
      i0 = 2*i
      i1 = 2*i+1
      sx[i0,i1] = 0.5
      sx[i1,i0] = 0.5
      sy[i0,i1] = -0.5
      sy[i1,i0] = 0.5
      sz[i0,i0] = 0.5
      sz[i1,i1] = -0.5
   return sx,sy,sz

def str2lst(s):
   lst = s.split(' ')
   lst = [int(x) for x in lst if x != '']
   lst = numpy.sort(lst)
   return lst

def ghf_diis_get_evec(data):
   f,p = data
   err = f.dot(p)
   err = err-err.T
   return err

def ghf_diis_dump(f,name,data):
   if name in f: del f[name]
   f[name] = data
   return 0

def ghf_diis_load(f,name):
   data = f[name].value
   return data

def ghf_diis_trace(a,b):
   return numpy.trace(a.dot(b))

def ghf_diis_scale(a,fac):
   return a*fac

def ghf_diis_add(a,b,alpha=1.0,beta=1.0):
   return alpha*a+beta*b

class ghf:
   def __init__(self):
      self.nelec = None
      self.ints = None
      # control parameters
      self.iprt = 0
      self.maxcycle = 1000
      self.ifUHF = True
      self.ifMOM = True
      self.thresh_e = 1.e-10
      self.thresh_d = 1.e-5
      self.vshift = 0.5
      # save
      self.occ = None
      self.mocoeff = None
      self.eorb = None
      self.det = None

   def energy_det(self,det):
      print '\nghf.energy_det'
      e0,h1e,h2e = self.ints
      guess = str2lst(det)
      e = e0
      for i in guess:
         e += h1e[i,i]
         for j in guess:
            if j>=i : continue
            e += h2e[i,i,j,j]-h2e[i,j,j,i]  # <ij||ij>=[ii|jj]-[ij|ji] 
      print 'det=',det
      print 'e=',e
      return e

   # generalized hartree-fock
   def solve(self,det=None):
      print '\nghf.solve'
      print 'nelec=',self.nelec
      e0,h1e,h2e = self.ints
      print 'e0=',e0
      # set up initial guess
      k = h1e.shape[0]
      print 'k=',k
      dm = numpy.zeros((k,k))
      if det == None:
         # Hcore 
         e,v = scipy.linalg.eigh(h1e)
         mo_occ = v[:,:self.nelec]
         dm = mo_occ.dot(mo_occ.T)
      else:
         guess = str2lst(det)
         print 'guess=',guess
         assert len(guess) == self.nelec
         dm[guess,guess] = 1.0
      # noise
      if not self.ifUHF:
         numpy.random.seed(0)
         dm += numpy.random.uniform(-1,1,size=(k,k))
         dm = dm/numpy.trace(dm)*self.nelec
      etot0 = 1.e10
      # analysis
      sx,sy,sz = get_SpinMat(k)
      # DIIS settings
      diis = diis_class.diis()
      diis.get_evec = ghf_diis_get_evec
      diis.dump = ghf_diis_dump
      diis.load = ghf_diis_load
      diis.trace = ghf_diis_trace
      diis.scale = ghf_diis_scale
      diis.add = ghf_diis_add
      diis.init()
      # start scf
      for i in range(self.maxcycle):
         # Fpq = hpq + ([pq|sr]-[pr|sq])Drs
         gpq = numpy.einsum('pqsr,rs->pq',h2e,dm)\
             - numpy.einsum('prsq,rs->pq',h2e,dm)
         fpq = h1e + gpq
         # etot
         e1 = numpy.trace(h1e.dot(dm))
         e2 = numpy.trace(gpq.dot(dm))*0.5
         etot = e0 + e1 + e2
         deltaE = etot-etot0
         etot0 = etot
         # DIIS
         diis_result = diis.kernel(i,[fpq,dm])
         if diis_result[0]: fpq = diis_result[1]
         # level_shift
         fpq += self.vshift*(numpy.identity(k)-dm)
         # solve FC=CE in orthonormal basis
         if not self.ifUHF:
            e,v = scipy.linalg.eigh(fpq)
         else:
            fpqA = fpq[0::2,0::2]
            fpqB = fpq[1::2,1::2]
            eA,vA = scipy.linalg.eigh(fpqA)
            eB,vB = scipy.linalg.eigh(fpqB)
            e = numpy.zeros(k)
            v = numpy.zeros((k,k))
            e[0::2] = eA
            e[1::2] = eB
            v[0::2,0::2] = vA
            v[1::2,1::2] = vB
         # update dm by maximum overlap criteria
         if det != None and self.ifMOM:
            pop = numpy.einsum('ij->j',v[guess,:]**2)
            index = numpy.argsort(-pop)
            mo_occ = v[:,index[:self.nelec]]
            occ = numpy.zeros(k)
            occ[index[:self.nelec]] = 1.0
         else:
            mo_occ = v[:,:self.nelec]
            occ = numpy.zeros(k)
            occ[:self.nelec] = 1.0
         dm_new = mo_occ.dot(mo_occ.T)
         deltaD = numpy.linalg.norm(dm_new-dm)
         dm = dm_new
         print('i=%5i diis=%i etot=%20.10f de=%10.2e dp=%10.2e')%\
              (i,diis.dim,etot,deltaE,deltaD)
         ifconv = abs(deltaE) < self.thresh_e and deltaD < self.thresh_d
         if ifconv: break
         
        # # population analysis for occupied mo
        # pop_occA = numpy.einsum('ij->j',mo_occ[0::2,:]**2)
        # pop_occB = numpy.einsum('ij->j',mo_occ[1::2,:]**2)
        # print e
        # print pop_occA
        # print pop_occB
        # sx_av = numpy.diag(mo_occ.T.dot(sx.dot(mo_occ)))
        # sy_av = numpy.diag(mo_occ.T.dot(sy.dot(mo_occ)))
        # sz_av = numpy.diag(mo_occ.T.dot(sz.dot(mo_occ)))
        # print 'sx',sx_av
        # print 'sy',sy_av
        # print 'sz',sz_av

      diis.final()
      # check convergence
      if i == self.maxcycle-1 and not ifconv:
         print "GHF is NOT converged!"
      else:
         print "GHF is converged!"
      # save for later usage
      self.occ = occ.copy()
      self.mocoeff = v.copy()
      self.eorb = numpy.diag(reduce(numpy.dot,(self.mocoeff.T,fpq,self.mocoeff)))
      print 'eorb=',self.eorb
      print 'occ=',self.occ
      det = []
      for i in range(k):
         if abs(self.occ[i]-1) < 1.e-5:
            det.append(str(i))
      self.det = " ".join(det)
      print 'det=',self.det
      return 0

   def trans(self,fname):
      print "\nghf.trans fname=",fname
      f = h5py.File(fname+".h5","w")
      f["occ"] = self.occ
      f["eorb"] = self.eorb
      f["mocoeff"] = self.mocoeff
      e0,h1e,h2e = self.ints
      h1e = reduce(numpy.dot,(self.mocoeff.T,h1e,self.mocoeff))
      f["h1e"] = h1e
      h2e = numpy.einsum("pqrs,pi->iqrs",h2e,self.mocoeff)
      h2e = numpy.einsum("iqrs,qj->ijrs",h2e,self.mocoeff)
      h2e = numpy.einsum("ijrs,rk->ijks",h2e,self.mocoeff)
      h2e = numpy.einsum("ijks,sl->ijkl",h2e,self.mocoeff)
      f["h2e"] = h2e
      f["e0"] = e0
      f.close()
      # FCIDUMP
      fname_tmp = "FCIDUMP_"+fname
      tools_io.save_FCIDUMP(e0,h1e,h2e,fname_tmp)
      return 0

if __name__ == '__main__':

   integrals = "../../database/benchmark/fes/fe2s2/FCIDUMP"
   e,h1,h2 = tools_io.load_FCIDUMP(integrals)
   h1e,h2e = tools_io.to_SpinOrbitalERI(h1,h2)

   mf = ghf()
   mf.nelec = 30
   mf.ints = (e,h1e,h2e)
   det = "0 2 4 6 8 10 12 14 16 18 20 22 24 36 38   1 3 15 17 19 21 23 25 27 29 31 33 35 37 39"
   mf.energy_det(det)
   mf.solve(det)
   mf.trans("fe2s2")

   e,h1,h2 = tools_io.load_FCIDUMP("FCIDUMP_fe2s2")
   mf.ints = (e,h1,h2)
   mf.energy_det(mf.det)
