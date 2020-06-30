import numpy
import tns

#twoindex  = [0, 0, 0] # =0, AB; =1, PQ
partition = ['L','C','R'] 
partition = ['L','R'] 
partition = ['L','C1','C2','R']
npart = len(partition)

terms = {}
idx = 0
for kp in range(npart):
 for kq in range(npart):
  if kp>kq: continue
  for kr in range(npart):
   for ks in range(npart):
      if kr<ks: continue
      idx += 1
      irep = (kp,kq,kr,ks)
      # eri
      symp = 'p'+partition[kp]
      symq = 'q'+partition[kq]
      symr = 'r'+partition[kr]
      syms = 's'+partition[ks]
      index4 = [symp,symq,symr,syms]
      eri = '<'+symp+symq+'||'+syms+symr+'>'
      # op
      op = (('p',partition[kp],1),
            ('q',partition[kq],1),
            ('r',partition[kr],0),
            ('s',partition[ks],0))

      # canonical form
      #print (eri,op)
      perm = numpy.argsort(irep)
      op_ordered = [op[i] for i in perm]
      sgn = tns.perm_parity(perm)
      #print op_ordered,sgn

      op_dic = tns.dicForm(op_ordered,partition)
      op_trans = tns.transform(op_dic,partition,index4)
      key = tns.classification(op_trans)
      if key not in terms:
         terms[key] = [op_trans]
      else:
         terms[key].append(op_trans) 

for key in terms:
   oplst = sorted(terms[key])
   print 'type =',key,' len =',len(oplst)
   for op in oplst:
      print " ",op 

