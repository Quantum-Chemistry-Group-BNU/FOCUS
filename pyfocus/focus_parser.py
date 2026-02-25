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

def parse_ctns_full(fname="ctns.out"):
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
   dcutlst = []
   dwtlst = []
   elst = []
   nstate = 0
   for line in ene:
      dat = line.split()
      dcut = eval(dat[1])
      dcutlst.append(dcut)
      dwt = eval(dat[2])
      dwtlst.append(dwt)
      nstate = (len(dat)-3)//2
      es = []
      for istate in range(nstate):
         ei = float(dat[3+2*istate].split('=')[-1])
         es.append(ei)
      if(debug): print('es=',es)
      elst.append(es)
   dic = {'dcut':dcutlst,
          'dwt':dwtlst,
          'elst':elst}
   return dic

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

def parse_oodmrg(fname="ctns.out",iprt=0,ifnew=True):
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
    
    # initial MPS
    elst = elines[1].split()
    eSCI = eval(elst[5])
    Sr = eval(elst[7])
    Sd = eval(elst[9])
    #result.append([1,eSCI,Sr,Sd])
    
    # oo-dmrg result
    for i in range(len(elines)):
        if i < 3 or i > len(elines)-1: continue
        elst = elines[i].split()
        accept = eval(elst[1])
        if ifnew:
            em = eval(elst[2])
            sr = eval(elst[6])
            sd = eval(elst[8])
        else:
            em = eval(elst[3])
            sr = eval(elst[7])
            sd = eval(elst[9])
        result.append([accept,em,sr,sd])
    result = np.array(result).T
    result = {'acceptance':result[0],
              'emin':result[1],
              'Srenyi':result[2],
              'Sdiag':result[3]}
    return result

def parse_Sdiag(output,iprt=0):
    f = open(output,'r')
    lines = f.readlines()
    iread = 0
    sdiag = -1
    cmax = -1
    conf = None
    data = []
    nsample_lst = []
    Sdiag_lst = []
    IPR_lst = []
    leadconf_lst = []
    cmax_lst = []
    for line in lines:
        if 'ctns::rcanon_Sdiag_sample: ifab=' in line:
            iread = 1
            nsample = eval(line.split()[3].split('=')[-1])
            nsample_lst.append(nsample)
            if iprt>0: print(line)
        elif "TIMING FOR ctns::rcanon_Sdiag_sample" in line:
            iread = 0
            Sdiag_lst.append(Sdiag)
            IPR_lst.append(IPR)
            leadconf_lst.append(leadconf)
            cmax_lst.append(cmax)
        elif iread >= 1:
            if iprt>0: print(line)
            iread += 1
            # we simply assume 10 lines
            if iread == 12:
                #print(line.split())
                Sdiag = eval(line.split()[3])
                IPR = eval(line.split()[5].split('=')[-1])
            elif iread == 15:
                res = line.split('=')
                leadconf = res[2].split()[0]
                cmax = eval(res[3].split()[0])
    f.close()
    dic = {'nsample':nsample_lst,
           'Sdiag':Sdiag_lst,
           'IPR':IPR_lst,
           'leadconf':leadconf_lst,
           'cmax':cmax_lst}
    return dic

def parse_entropy(output,iprt=0):
    f = open(output,'r')
    lines = f.readlines()
    sumSvN_lst = []
    sumSr_lst = []
    maxSvN_lst = []
    maxSr_lst = []
    for line in lines:
        if 'SvN[sum' in line:
            sumSvN = eval(line.split()[0].split('=')[-1])
            sumSr  = eval(line.split()[1].split('=')[-1])
            sumSvN_lst.append(sumSvN)
            sumSr_lst.append(sumSr)
        if 'SvN[max' in line:
            maxSvN = eval(line.split()[0].split('=')[-1])
            maxSr  = eval(line.split()[1].split('=')[-1])
            maxSvN_lst.append(maxSvN)
            maxSr_lst.append(maxSr)
    f.close()
    dic = {'sumSvN':sumSvN_lst,
           'sumSr':sumSr_lst,
           'maxSvN':maxSvN_lst,
           'maxSr':maxSr_lst}
    return dic

if __name__ == '__main__':

    fname = './tmp/ctns.out'
    elst = parse_ctns(fname)
    print(elst)

