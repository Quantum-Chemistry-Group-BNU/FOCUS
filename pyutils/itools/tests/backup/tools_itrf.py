import h5py
import numpy


def dumpERIs(ecore,int1e,int2e):
   print '\n[tools_itrf.dumpERIs] to FCIDUMPnew'
   with open('FCIDUMPnew','w') as f:
     n = int1e.shape[0]
     # int2e
     nblk = 0
     np = n*(n+1)/2
     nq = np*(np+1)/2
     for i in range(n):
        for j in range(i+1):
           for k in range(i+1):
	      if k == i: 	
 		 lmax = j+1
	      else:
	 	 lmax = k+1
              for l in range(lmax):
                 nblk += 1
		 line = str(int2e[i,j,k,l])\
	              + ' ' + str(i+1) \
	              + ' ' + str(j+1) \
	              + ' ' + str(k+1) \
	              + ' ' + str(l+1)+'\n'
		 f.writelines(line)
     assert nq == nblk 
     # int1e
     nblk = 0 
     for i in range(n):
        for j in range(i+1):
            nblk += 1
	    line = str(int1e[i,j])\
		 + ' ' + str(i+1) \
		 + ' ' + str(j+1) \
		 + ' 0 0\n'
	    f.writelines(line)
     assert np == nblk
     # ecore 
     line = str(ecore) + ' 0 0 0 0\n'
     f.writelines(line)
   print 'finished'
   return 0


def dump(info,ordering=None,fname='mole.h5',fcidump=False):
   ecore,int1e,int2e = info
   if ordering != None:
      int1e = int1e[numpy.ix_(ordering,ordering)].copy()
      int2e = int2e[numpy.ix_(ordering,ordering,ordering,ordering)].copy()
      dumpERIs(ecore,int1e,int2e)
      if fcidump: return 0  
   # dump information
   nbas = int1e.shape[0]
   sbas = nbas*2
   print '\n[tools_itrf.dump] interface from FCIDUMP with nbas=',nbas
   f = h5py.File(fname, "w")
   cal = f.create_dataset("cal",(1,),dtype='i')
   cal.attrs["nelec"] = 0.
   cal.attrs["sbas"]  = sbas
   cal.attrs["enuc"]  = 0.
   cal.attrs["ecor"]  = ecore
   cal.attrs["escf"]  = 0. # Not useful at all
   # Intergrals
   flter = 'lzf'
   # INT1e:
   h1e = numpy.zeros((sbas,sbas))
   h1e[0::2,0::2] = int1e # AA
   h1e[1::2,1::2] = int1e # BB
   # INT2e:
   h2e = numpy.zeros((sbas,sbas,sbas,sbas))
   h2e[0::2,0::2,0::2,0::2] = int2e # AAAA
   h2e[1::2,1::2,1::2,1::2] = int2e # BBBB
   h2e[0::2,0::2,1::2,1::2] = int2e # AABB
   h2e[1::2,1::2,0::2,0::2] = int2e # BBAA
   # <ij|kl> = [ik|jl]
   h2e = h2e.transpose(0,2,1,3)
   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
   h2e = -0.5*(h2e-h2e.transpose(0,1,3,2))
   int1e = f.create_dataset("int1e", data=h1e, compression=flter)
   int2e = f.create_dataset("int2e", data=h2e, compression=flter)
   # Occupation
   occun = numpy.zeros(sbas)
   orbsym = numpy.array([0]*sbas)
   spinsym = numpy.array([[0,1] for i in range(nbas)]).flatten()
   f.create_dataset("occun",data=occun)
   f.create_dataset("orbsym",data=orbsym)
   f.create_dataset("spinsym",data=spinsym)
   f.close()
   print ' Successfully dump information for MPO-DMRG calculations! fname=',fname
   print ' with ordering',ordering
   return 0

if __name__ == '__main__':
   import tools_io
   info = tools_io.loadERIs()
   ordering = [12, 4, 2, 60, 67, 55, 58, 35, 3, 32, 22, 26, 28, 14, 23, 54, 68, 71, 43, 38, 30, 25, 33, 37, 39, 70, 49, 65, 57, 34, 45, 56, 63, 19, 17, 10, 7, 13, 11, 6, 66, 40, 72, 50, 41, 64, 36, 69, 44, 51, 27, 29, 31, 16, 18, 24, 15, 21, 20, 42, 47, 73, 46, 62, 52, 53, 48, 61, 59, 5, 9, 1, 8]
   ordering = [i-1 for i in ordering]
   print len(ordering)
   print 'ordering=',ordering
   dump(info,ordering,fcidump=False) 
