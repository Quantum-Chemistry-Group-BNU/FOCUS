import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPSTools

mpsfile = './scratch2/rcanon_isweep3.bin'
dmax = 100
fcidumpfile = './moleinfo/FCIDUMP'
spin = 3
topofile = 'topology/topoA'

topo = []
f = open(topofile)
for line in f.readlines():
   topo.append(eval(line))
f.close()
topo = np.array(topo[::-1])
print('topo=',topo)

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SAnySZ, \
      n_threads=28,
      stack_mem=1024*1024*1024*400)
driver.set_symmetry_groups('U1Fermi', 'U1', 'AbelianPG', hints=['ALT'])

driver.read_fcidump(filename=fcidumpfile, pg='nopg')

ncas = driver.n_sites
n_elec = driver.n_elec
spin = driver.spin
orb_sym = driver.orb_sym
h1e = driver.h1e
g2e = driver.g2e
ecore = driver.ecore

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)
#mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder=topo)
#impo = driver.get_identity_mpo()

#=== import MPS from file ===
pymps = MPSTools.from_focus_mps_file(fname=mpsfile, is_su2=False)
mps = MPSTools.to_block2(pymps, driver.basis, center=driver.n_sites - 1)

#expt = driver.expectation(mps, mpo, mps) / driver.expectation(mps, impo, mps)
#print('Energy from expectation = %20.15f' % expt)

# pdms & expectations
for fac in [1]:
   dcut = fac*dmax
   print('\nGenerate n-pdm with dcut=',dcut)

   t0 = time.time()

   pdm1 = driver.get_1pdm(mps, max_bond_dim=dcut) #, iprint=2)
   pdm1spatial = pdm1[0]+pdm1[1]
   np.save('pdm1',pdm1spatial)
   print('|dm1-dm1.T|=',np.linalg.norm(pdm1spatial-pdm1spatial.T))
   print('tr(dm1)=',np.trace(pdm1spatial))
   print('diag(dm1)=',np.diag(pdm1spatial))

   ifpdm2 = False
   if ifpdm2:
      pdm2 = driver.get_2pdm(mps, max_bond_dim=dcut)
      pdm2spatial = pdm2[0]+pdm2[1]+pdm2[1].transpose(1,0,3,2)+pdm2[2]
      np.save('pdm2',pdm2spatial)
       
      # spin-free rdms
      print(pdm1spatial.shape)
      print(pdm2spatial.shape)
      pdm1tmp = np.einsum('ijjk->ik',pdm2spatial)/(n_elec-1)
      print('diff=',np.linalg.norm(pdm1tmp-pdm1spatial))
   
   t1 = time.time()
   print('\ndt=',t1-t0)

   # Classifier
   orbs = [None]*13
   orbs[0]  = [0,1]				# _end
   orbs[1]  = [2,3,4,5,6]                       # _fe
   orbs[2]  = [7,8,9,10,11,12,13,14,15]	        # _s
   orbs[3]  = [16,17,18,19,20]                  # _fe
   orbs[4]  = [21,22,23,24,25]                  # _fe
   orbs[5]  = [26,27,28,29,30]                  # _fe
   orbs[6]  = [31,32,33,34,35,36,37,38,39,40,41,42,43] # _s
   orbs[7]  = [44,45,46,47,48]                  # _fe
   orbs[8]  = [49,50,51,52,53]                  # _fe
   orbs[9]  = [54,55,56,57,58]                  # _fe
   orbs[10] = [59,60,61,62,63,64,65,66,67]	# _s
   orbs[11] = [68,69,70,71,72]                  # _mo [1,1,1,0,0]
   orbs[12] = [73,74,75]                        # _end
   dlst = [0,2,6,10,12]
   femolst = [1,3,4,5,7,8,9,11]
   groups = [orbs[x] for x in femolst]
   
   pdm1spin = pdm1[0] - pdm1[1]
   k = 76
   rdx = range(k)[-1::-1]
   sdm1r = pdm1spin[np.ix_(rdx,rdx)]
   pdm1r = pdm1spatial[np.ix_(rdx,rdx)]
   ng = len(groups)
   print('\ntr(Sz)=',np.trace(pdm1spin))
   ne_sum = 0.0
   sz_sum = 0.0
   for i in range(ng):
      pdm1blk = pdm1r[np.ix_(groups[i],groups[i])]
      sdm1blk = sdm1r[np.ix_(groups[i],groups[i])]
      ne = np.trace(pdm1blk)
      sz = np.trace(sdm1blk)
      print('i=',i,' ne=',ne,' sz=',sz)
      print(' ',np.diag(pdm1blk))
      print(' ',np.diag(sdm1blk))
      ne_sum += ne
      sz_sum += sz
   print('total ne=',ne_sum,' sz=',sz_sum)

