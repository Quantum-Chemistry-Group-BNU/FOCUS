import copy

def perm_parity(lst0):
    '''\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    lst = copy.deepcopy(lst0)
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity

def dicForm(op,partition):
   npart = len(partition)
   dic = {}
   for i in range(npart):
      dic[i] = []
   for opi in op:
      for j in range(npart):
         if partition[j] == opi[1]:
            dic[j].append(opi)
   return dic 

# Rule-based transformation
def transform(op_dic,partition,index4):
   op_trans = {};
   for key in op_dic:
      dat = op_dic[key]
      # 4-index object
      if(len(dat) == 4):
         op_trans[key] = ('H^'+partition[key],0)
      # 2-index object
      elif(len(dat) == 2):
         optypes = (dat[0][-1],dat[1][-1])
         if optypes == (1,1):
            op_trans[key] = ('A^'+partition[key]+'_'\
                             +dat[0][0]+dat[0][1]\
                             +dat[1][0]+dat[1][1],0)
         elif optypes == (0,0):
            op_trans[key] = ('A^'+partition[key]+'_'\
                             +dat[1][0]+dat[1][1]\
                             +dat[0][0]+dat[0][1],1)
         elif optypes == (1,0):
            op_trans[key] = ('B^'+partition[key]+'_'\
                             +dat[0][0]+dat[0][1]\
                             +dat[1][0]+dat[1][1],0)
         else:
            print 'error in 2-index object'
            exit(1)
      # 1-index object
      elif(len(dat) == 1):
         op_trans[key] = ('C^'+partition[key]+'_'\
                          +dat[0][0]+dat[0][1],dat[0][2]) 
      # 3-index object
      elif(len(dat) == 3):
         optypes = (dat[0][-1],dat[1][-1],dat[2][-1])
         index3 = [dat[0][0]+dat[0][1],
                   dat[1][0]+dat[1][1],
                   dat[2][0]+dat[2][1]]
         for index in index4:
            if index not in index3:
               index1 = index
         if(optypes == (1,1,0)):
            op_trans[key] = ('S^'+partition[key]+'_'\
                             +index1,1)
         elif(optypes == (1,0,0)):
            op_trans[key] = ('S^'+partition[key]+'_'\
                             +index1,0)
         else:
            print 'error in 3-index object'
            exit(1)
      elif(len(dat) == 0):
          op_trans[key] = ('I^'+partition[key],0)
      else:
          print 'error: no such case'
          exit(1)
   # convert to final form
   oplst = []
   for key in op_trans:
      dat = op_trans[key]
      oplst.append(dat)
   return oplst

def classification(oplst):
   lst = []
   for op in oplst:
      lst.append(op[0][0:1])
   lst = sorted(lst)   
   key = ''.join(lst)   
   return key
