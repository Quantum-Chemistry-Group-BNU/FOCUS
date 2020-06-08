import numpy

def load_FCIDUMP(fname='FCIDUMP'):
   print '\ntools_io.load_FCIDUMP fname=',fname
   with open(fname,'r') as f:
     line = f.readline().split(',')[0].split(' ')[-1]
     print  'no. of spatial orbitals: ', int(line)
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
   #numpy.save('int2e',int1e)
   #numpy.save('int1e',int2e)
   print 'finished'
   return e,int1e,int2e

def save_FCIDUMP(ecore,int1e,int2e,fname='FCIDUMP'):
   print '\ntools_io.save_FCIDUMP fname=',fname
   with open(fname,'w') as f:
     n = int1e.shape[0]
     line = "&FCI NORB= "+str(n)+",\n"
     f.writelines(line)
     line = " ORBSYM=\n"
     f.writelines(line)
     line = " ISYM=1\n"
     f.writelines(line)
     line = "&END\n"
     f.writelines(line)
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

def to_SpinOrbitalERI(int1e,int2e):
   print '\ntools_io.to_SpinOrbitalERI'
   sbas = int1e.shape[0]*2
   h1e = numpy.zeros((sbas,sbas))
   h1e[0::2,0::2] = int1e # AA
   h1e[1::2,1::2] = int1e # BB
   # INT2e:
   h2e = numpy.zeros((sbas,sbas,sbas,sbas))
   h2e[0::2,0::2,0::2,0::2] = int2e # AAAA
   h2e[1::2,1::2,1::2,1::2] = int2e # BBBB
   h2e[0::2,0::2,1::2,1::2] = int2e # AABB
   h2e[1::2,1::2,0::2,0::2] = int2e # BBAA
   return h1e,h2e
