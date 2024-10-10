import numpy as np

def parse_sci(fname="sci.out"):
    f = open(fname,'r')
    lines = f.readlines()
    elst = []
    for line in lines:
        if 'state' in line and 'energy' in line:
            e = eval(line.split()[-1])
            elst.append(e)
    return elst

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
   return elst

def parse_ham(fname="ham.out"):
    f = open(fname,"r")
    lines = f.readlines()
    iHij = 0
    iSij = 0
    for line in lines:
        # parse Hij
        if iHij > 0:
            hrow = line.split()
            for k in range(dim):
                Hij[iHij-1,k] = eval(hrow[k])
            if iHij == dim:
                iHij = 0
                continue
            iHij += 1
        if 'matrix: Hij' in line:
            iHij = 1
            dim = eval(line.split()[2].split(',')[-1].split(')')[0])
            Hij = np.zeros((dim,dim))
        # parse Sij
        if iSij > 0:
            srow = line.split()
            for k in range(dim):
                Sij[iSij-1,k] = eval(srow[k])
            if iSij == dim:
                iSij = 0
                continue
            iSij += 1
        if 'matrix: Sij' in line:
            iSij = 1
            dim = eval(line.split()[2].split(',')[-1].split(')')[0])
            Sij = np.zeros((dim,dim))
    return Hij,Sij

def parse_oodmrg(fname="ctns.out",iprt=0):
    f = open(fname,"r")
    lines = f.readlines()
    # get nsweep
    pattern = "OO-DMRG results:"
    nsweep = 0
    for line in lines:
      if pattern in line:
         nsweep += 1
    if iprt > 0: print("nsweep=",nsweep)
    # process
    isweep = 0
    iread = 0
    nline = 0
    elines = []
    for line in lines:
      if pattern in line:
         isweep += 1
         if isweep == nsweep:
            iread = 1
      elif iread == 1 and nline < nsweep+3:
         nline += 1
         elines.append(line)
         if iprt > 0: print(line)
    f.close()
    # parse
    result = []
    elst = elines[1].split()
    eSCI = eval(elst[5])
    Sd = eval(elst[7])
    Sr = eval(elst[9])
    result.append([1,eSCI,Sd,Sr])
    for i in range(len(elines)):
        if i < 3 or i > len(elines)-1: continue
        elst = elines[i].split()
        accept = eval(elst[1])
        em = eval(elst[3])
        sd = eval(elst[7])
        sr = eval(elst[9])
        result.append([accept,em,sd,sr])
    result = np.array(result).T
    result = {'acceptance':result[0],
              'emin':result[1],
              'Sdiag':result[2],
              'Srenyi':result[3]}
    return result

def parse_Sdiag(output):
    f = open(output,'r')
    lines = f.readlines()
    iread = 0
    sdiag = -1
    cmax = -1
    conf = None
    for line in lines:
        if 'ctns::rcanon_Sdiag_sample: ifab=' in line:
            iread = 1
            nsample = eval(line.split()[3].split('=')[-1])
        elif "TIMING FOR ctns::rcanon_Sdiag_sample" in line:
            iread = 0
            break
        elif iread >= 1:
            iread += 1
            # we simply assume 10 lines
            if iread == 12:
                sdiag = eval(line.split()[3])
            elif iread == 15:
                res = line.split('=')
                conf = res[2].split()[0]
                cmax = eval(res[3].split()[0])
    f.close()
    return (sdiag,conf,cmax)

if __name__ == '__main__':

    fname = './tmp/ctns.out'
    elst = parse_ctns(fname)
    print(elst)

