import numpy

def random_topology(nsite,topo="comb"):
   if topo == "comb":
      tp = random_comb(nsite)
   elif topo == "chain":
      tp =random_chain(nsite)
   dump(tp)
   return 0

def random_chain(nsite):
   arr = numpy.arange(nsite)
   numpy.random.shuffle(arr)
   chain = [[i] for i in arr]
   return chain

#numpy.random.seed(0)
def random_comb(nsite):
   arr = numpy.arange(nsite)
   numpy.random.shuffle(arr)

   length = numpy.random.randint(3,nsite)
   lens = [1]*length
   print 'length=',length
   
   nres = nsite-length 
   for i in range(nres):
      ip = numpy.random.randint(length-2)+1
      lens[ip] += 1
   print 'lens=',lens
   
   comb = []
   ioff = 0
   for i in range(length):
      pos = ioff+numpy.array(range(lens[i]))
      ioff += lens[i]
      comb.append(list(arr[pos]))
   return comb

def dump(comb,fname="random_topo"):
   print 'dump comb='
   f = open(fname,"w")
   length = len(comb)
   for i in range(length):
      lst = [str(k) for k in comb[i]]
      s = ",".join(lst)
      print s
      f.writelines(s+'\n')
   f.close()
   return 0

if __name__ == '__main__':
   
    nsite = 6
    #random_topology(nsite,"comb")
    random_topology(nsite,"chain")
