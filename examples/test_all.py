import os
import time
import numpy

def parse_ctns(fname="ctns.out"):
   debug = False
   f = open(fname,"r")
   lines = f.readlines()
   # get nsweep
   pattern = "results: iter, dwt, energies (delta_e)"
   nsweep = 0
   for line in lines:
      if pattern in line:
         nsweep += 1
   if(debug): print("nsweep=",nsweep)
   # process
   isweep = 0
   iread = 0
   ene = []
   for line in lines:
      if pattern in line:
         isweep += 1
         if isweep == nsweep:
            iread = 1
      elif iread == 1 and isweep > 0:
         isweep -= 1
         ene.append(line)
   f.close()
   # parse
   elst = []
   nstate = 0 
   for line in ene:
      dat = line.split()
      nstate = (len(dat)-2)//2
      es = []
      for istate in range(nstate):
         ei = float(dat[2+2*istate].split('=')[-1])
         es.append(ei)
      elst.append(es)
   elst = numpy.array(elst)
   return elst

def testAll(dirs):
   print("\ntestAll")
   print("dirs=",dirs)
   tmpdir = "./tmp"
   for fdir in dirs:
      print('\n###',fdir,'###')
      os.chdir(fdir)
      os.system("pwd")
      if(not os.path.exists(tmpdir)): os.mkdir(tmpdir)
      for prefix in ['','r','c']:
         finput = "results/"+prefix+"input.dat"
         exist = os.path.exists(finput)
         if(not exist): continue
         print('finput=',finput)
	 # SCI
         cmd = SCI +" results/"+prefix+"input.dat > "+tmpdir+"/"+prefix+"sci.out"
         print(cmd)
         t0 = time.time()
         os.system(cmd)
         t1 = time.time()
         print('timing =',t1-t0)
	 # CTNS
         cmd = CTNS+" results/"+prefix+"input.dat > "+tmpdir+"/"+prefix+"ctns.out"
         print(cmd)
         t0 = time.time()
         os.system(cmd)
         t1 = time.time()
         print('timing =',t1-t0)
      os.chdir("..")
   print("\nRun successfully!")
   return 0

def compareAll(dirs,thresh=1.e-8): 
   print("\ncompareAll with thresh=",thresh)
   print("dirs=",dirs)
   tmpdir = "./tmp"
   nfail = 0
   for fdir in dirs:
      print('\n###',fdir,'###')
      os.chdir(fdir)
      os.system("pwd")
      for prefix in ['','r','c']:
         finput = "results/"+prefix+"input.dat"
         exist = os.path.exists(finput)
         if(not exist): continue
         print('finput=',finput)
	 # COMPARISON
         elst0 = parse_ctns("results/"+prefix+"ctns.out")[-1]
         elst1 = parse_ctns(tmpdir+"/"+prefix+"ctns.out")[-1]
         ediff = numpy.linalg.norm(elst0 - elst1)
         print('eref=',elst0,' ecal=',elst1,' ediff=',ediff)
         if ediff < thresh:
            print("pass")
         else:
            print("fail")
            nfail += 1
      os.chdir("..")
   print("\nSummary: nfail =",nfail) 
   return 0


if __name__ == '__main__':

   HOME = os.path.dirname(os.getcwd())
   print('HOME=',HOME)
   SCI = HOME+"/bin/sci.x"
   CTNS = HOME+"/bin/ctns.x"

   #cdir = os.getcwd()
   #dirs = [tdir for tdir in os.listdir(cdir) if os.path.isdir(tdir)]
   dirs = ['0_h6_tns', '1_lih3_dcg', '2_lih3+_dcg', '3_h6+_kr', '4_h5_twodot']
   print(dirs)
   testAll(dirs)
   compareAll(dirs)

