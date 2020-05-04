import numpy

#def get_eri(kp,kq,kr,ks):
#   return '<'+symp+symq+'||'+syms+symr+'>'

# 2-electron term: <pq||sr> p^+ q^+ r s (p<q,r>s)

keys = ["C","R"] #["L","C","R"]
npart = len(keys)
print keys,npart
idx = 0
for kp in range(npart):
 for kq in range(npart):
  if kp>kq: continue
  for kr in range(npart):
   for ks in range(npart):
      if kr<ks: continue
      idx += 1
      symp = 'p'+keys[kp]
      symq = 'q'+keys[kq]
      symr = 'r'+keys[kr]
      syms = 's'+keys[ks]
      eri = '<'+symp+symq+'||'+syms+symr+'>'
      op = symp+'^+'+symq+'^+'+symr+syms
      irep = (kp,kq,kr,ks)
      # sort
      oplst = [symp+'^+',symq+'^+',symr,syms]
      perm = numpy.argsort(irep)
      op_sort = ''.join([oplst[i] for i in perm])
      print idx,eri+' '+op,irep,perm,op_sort
exit(1)

print
keys = ["C","R"] #["L","C","R"]
npart = len(keys)
print keys,npart
idx = 0
for kq in range(npart):
  for kr in range(npart):
   for ks in range(npart):
      if kr<ks: continue
      idx += 1
      symq = 'q'+keys[kq]
      symr = 'r'+keys[kr]
      syms = 's'+keys[ks]
      eri = '<pL'+symq+'||'+syms+symr+'>'
      op = symq+'^+'+symr+syms
      irep = (kq,kr,ks)
      # sort
      oplst = [symq+'^+',symr,syms]
      perm = numpy.argsort(irep)
      op_sort = ''.join([oplst[i] for i in perm])
      print idx,eri+' '+op,irep,perm,op_sort

