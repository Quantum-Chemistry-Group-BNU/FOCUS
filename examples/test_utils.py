import os
import time
import numpy

def parse_ctns(fname="ctns.out"):
   debug = False
   f = open(fname,"r")
   lines = f.readlines()
   # get nsweep
   pattern = "results:"
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
      nstate = (len(dat)-3)//2
      es = []
      for istate in range(nstate):
         ei = float(dat[3+2*istate].split('=')[-1])
         es.append(ei)
      if(debug): print('es=',es)
      elst.append(es)
   elst = numpy.array(elst)
   return elst

def testAll(dirs):
   parrent = os.getcwd()
   tmpdir = "./tmp"
   for fdir in dirs:
      print("")
      print('-'*60) 
      print("### run:",fdir,"###")
      print('-'*60)
      os.chdir(fdir)
      os.system("pwd")
      if(not os.path.exists(tmpdir)): os.mkdir(tmpdir)
      for prefix in ['','r','c']:
         finput = "results/"+prefix+"input.dat"
         exist = os.path.exists(finput)
         if(not exist): continue
	      # SCI
         print('=== SCI ===')
         SCI = os.environ['SCI']
         cmd = SCI +" results/"+prefix+"input.dat > "+tmpdir+"/"+prefix+"sci.out"
         print('cmd = ', cmd)
         t0 = time.time()
         os.system(cmd)
         t1 = time.time()
         print('timing =',t1-t0)
	      # CTNS
         print('=== CTNS ===')
         CTNS = os.environ['CTNS']
         cmd = CTNS+" results/"+prefix+"input.dat > "+tmpdir+"/"+prefix+"ctns.out"
         print('cmd = ', cmd)
         t0 = time.time()
         os.system(cmd)
         t1 = time.time()
         print('timing =',t1-t0)
      os.chdir(parrent)
   print("\nRun successfully!")
   return 0

def compareAll(dirs,nfail,thresh=1.e-8): 
   parrent = os.getcwd()
   tmpdir = "./tmp"
   for fdir in dirs:
      print("\n### check:",fdir,"###")
      os.chdir(fdir)
      for prefix in ['','r','c']:
         finput = "results/"+prefix+"input.dat"
         exist = os.path.exists(finput)
         if(not exist): continue
         print('finput=',finput)
	     # COMPARISON
         # reference
         fname = "results/"+prefix+"ctns.out"
         print('fname[ref]=',fname)
         elst0 = parse_ctns(fname)[-1]
         # computed
         fname = tmpdir+"/"+prefix+"ctns.out"
         print('fname[cal]=',fname)
         if(os.path.exists(fname)):
             try:
                elst1 = parse_ctns(fname)[-1]
                ediff = numpy.linalg.norm(elst0 - elst1)
                print('eref=',elst0)
                print('ecal=',elst1)
                print('ediff=',ediff)
                if ediff < thresh:
                   print("pass")
                else:
                   print("fail: different results")
                   nfail += 1
             except:
                 print("fail: output is problematic")
                 nfail += 1
                 break
         else:
               print("fail: no output is found")
               nfail += 1
      os.chdir(parrent)
   return nfail

def test_run(dirs):
   nfail = 0
   failed = []
   t0 = time.time()
   for fname in dirs:
      fdir = [fname]
      testAll(fdir)
      nfail_out = compareAll(fdir,nfail)
      if nfail_out != nfail:
          failed.append(fname)
          nfail = nfail_out
      print("\nSummary: nfail =",nfail) 
      print('failed tests:',failed)
   t1 = time.time()
   print('\ntotol time =',t1-t0)
   return nfail

