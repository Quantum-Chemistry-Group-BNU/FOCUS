import os
import h5py
import numpy
import scipy.linalg

class diis():
   def __init__(self):
      # Control parameters
      self.ifdiis      = True
      self.mindiis     = 2
      self.maxdiis     = 8
      # Internal
      self.istore = 0 # Current storage positioin
      self.dim = 0 # Current diis dimension
      # Storage
      self.tdir = './' 
      self.tname = 'diisTmp'
      self.f = None

   def init(self):   
      self.f = h5py.File(self.tdir+self.tname,'w')
      self.prt()
      return 0
      
   def final(self):
      self.f.close()
      os.system('rm '+self.tdir+self.tname)
      return 0

   def prt(self):
      print '\nDIIS control parameters:'
      print 'ifdiis      =',self.ifdiis 
      print 'mindiis     =',self.mindiis 
      print 'maxdiis     =',self.maxdiis
      print 'tmpfile     =',self.tdir+self.tname
      return 0

   # Subroutines to be defined.
   def get_evec(self,data):
      print 'diis.get_evec to be defined by users!'
      exit(1)
   
   def dump(self,name,quantity):
      print 'diis.dump to be defined by users!'
      exit(1)
       
   def load(self,name):
      print 'diis.load to be defined by users!'
      exit(1)

   def trace(self,a,b):
      print 'diis.trace to be defined by users!'
      exit(1)

   def scale(self,fac):
      print 'diis.scale to be defined by users!'
      exit(1)

   def add(self,dat1,dat2,fac1=1.0,fac2=1.0):
      print 'diis.scale to be defined by users!'
      exit(1)

   # Record position
   def posrec(self,i):
      # circularr data storage
      #    0  1  2  3  4  5  6  7  8
      #    9 10 11 12  4  5  6  7  8
      # if given istore = 3 [it=12]
      # loop over i will produce
      # i   = 0 1 2 3 4 5 6 7 8 
      # pos = 0 1 2 3 8 7 6 5 4
      assert i < self.maxdiis
      ipos = (-i + self.istore + self.maxdiis)%self.maxdiis
      return ipos

   # Main   
   def kernel(self,it,data):
      # 0. Save data[0] and error vector,
      # by default data[0] is to be linearly combined.
      evec = self.get_evec(data)
      # save circularly: 0,1,...,m-1
      self.istore = it%self.maxdiis
      self.dump(self.f,'dat'+str(self.istore),data[0])
      self.dump(self.f,'err'+str(self.istore),evec)

      # 1. check if perform diis step
      if (not self.ifdiis) or (it < self.mindiis): return False,0
      self.dim = self.dim+1
      if self.dim == 1: return False,0
      # periodically reset diis dimension to avoid too close error
      if self.dim > self.maxdiis: self.dim = self.mindiis

      # 2. solve Ax=B
      d = self.dim
      a = numpy.zeros((d+1,d+1))
      a[d,:d] = -1.0
      a[:d,d] = -1.0
      b = numpy.zeros(d+1)
      b[d] = -1.0
      for i in range(d):
	 ipos = self.posrec(i) # fetch data
	 ei = self.load(self.f,'err'+str(ipos)) 
	 for j in range(i+1):
	    jpos = self.posrec(j)
	    ej = self.load(self.f,'err'+str(jpos)) 
	    a[i,j] = self.trace(ei,ej)
	    a[j,i] = a[i,j]
      if numpy.trace(a) < 1.e-10: a[:d,:d] += numpy.identity(d)*1.e-10
      c = scipy.linalg.solve(a,b)
      if numpy.max(abs(c[:d])) > 100.0:
	 print 'Warning: too large coeff, DIIS may fail! Use current data.'
	 return False,0
      
      # 3. result = data[0](n)*c(n), leave it for user
      for i in range(d):
	 ipos = self.posrec(i)
	 tmp = self.load(self.f,'dat'+str(ipos))
	 if i == 0:
	    dat = self.scale(tmp,c[i])
         else:
	    dat = self.add(dat,tmp,beta=c[i])
      return True,dat

if __name__ == '__main__':
   ds = diis()
   ds.istore = 5
   ds.maxdiis = 9
   for i in range(10):
      print i,ds.posrec(i)
   # result: reverse fetching
   # 0 5
   # 1 4
   # 2 3
   # 3 2
   # 4 1
   # 5 0
   # 6 8
   # 7 7
   # 8 6
