import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPSTools

#dmax = 500
#mpsfile = './rcanon_dcompress'+str(dmax)+'.bin'
#fcidumpfile = './FCIDUMP'
#spin = 0
#
#norb = 36
#topo = range(norb)
#topo = np.array(topo[::-1]) # need to reverse in rcanon case
#print('topo=',topo)

dmax = 100
mpsfile = './rcanon_dcompress'+str(dmax)+'.bin'
fcidumpfile = './fmole.info.FCIDUMP'
spin = 3
topofile = './topoA'
ifEnergyExpect = False #True 
ifpdm2 = True #False

topo = []
f = open(topofile)
for line in f.readlines():
   topo.append(eval(line))
f.close()
topo = np.array(topo[::-1]) # need to reverse in rcanon case
print('topo=',topo)


driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SAnySZ, \
      n_threads=28,
      stack_mem=1024*1024*1024*10)
driver.set_symmetry_groups('U1Fermi', 'U1', 'AbelianPG', hints=['ALT'])

driver.read_fcidump(filename=fcidumpfile, pg='nopg')

ncas = driver.n_sites
n_elec = driver.n_elec
spin = driver.spin
orb_sym = driver.orb_sym
h1e = driver.h1e
g2e = driver.g2e
ecore = driver.ecore
print('ncas=',ncas,'n_elec=',n_elec)

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)

#=== import MPS from file ===
pymps = MPSTools.from_focus_mps_file(fname=mpsfile, is_su2=False)
mps = MPSTools.to_block2(pymps, driver.basis, center=driver.n_sites - 1)

if ifEnergyExpect:
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder=topo)
    impo = driver.get_identity_mpo()
    expt = driver.expectation(mps, mpo, mps) / driver.expectation(mps, impo, mps)
    print('Energy from expectation = %20.15f' % expt)

# pdms & expectations
for fac in [1]:
   dcut = fac*dmax
   print('\nGenerate n-pdm with dcut=',dcut)

   t0 = time.time()

   if ifEnergyExpect:
      rdx = range(len(topo)) # reorder is automatically handled when building mpo
   else:
      rdx = np.argsort(topo)

   pdm1 = driver.get_1pdm(mps, max_bond_dim=dcut) #, iprint=2)
   pdm1spatial = pdm1[0]+pdm1[1]
   # save 
   pdm1spatial = pdm1spatial[np.ix_(rdx,rdx)]
   pdm1aa = pdm1[0][np.ix_(rdx,rdx)]
   pdm1bb = pdm1[1][np.ix_(rdx,rdx)]
   np.save('pdm1aa_d'+str(dcut),pdm1aa)
   np.save('pdm1bb_d'+str(dcut),pdm1bb)
   np.save('pdm1_d'+str(dcut),pdm1spatial)
   # check
   print('|dm1-dm1.T|=',np.linalg.norm(pdm1spatial-pdm1spatial.T))
   print('tr(dm1)=',np.trace(pdm1spatial))
   print('diag(dm1)=',np.diag(pdm1spatial))

   if ifpdm2:
      pdm2 = driver.get_2pdm(mps, max_bond_dim=dcut)
      pdm2spatial = pdm2[0]+pdm2[1]+pdm2[1].transpose(1,0,3,2)+pdm2[2]
      # save
      pdm2spatial = pdm2spatial[np.ix_(rdx,rdx,rdx,rdx)]
      # G[i,j,k,l] = <ia+ ja+ ka la>
      #              <ia+ jb+ kb la>
      #              <ib+ jb+ kb lb>
      pdm2aaaa = pdm2[0][np.ix_(rdx,rdx,rdx,rdx)]
      pdm2abba = pdm2[1][np.ix_(rdx,rdx,rdx,rdx)]
      pdm2bbbb = pdm2[2][np.ix_(rdx,rdx,rdx,rdx)]
      np.save('pdm2aaaa_d'+str(dcut),pdm2aaaa)
      np.save('pdm2abba_d'+str(dcut),pdm2abba)
      np.save('pdm2bbbb_d'+str(dcut),pdm2bbbb)
      np.save('pdm2_d'+str(dcut),pdm2spatial)
      # check
      pdm1tmp = np.einsum('ijjk->ik',pdm2spatial)/(n_elec-1)
      print('diff=',np.linalg.norm(pdm1tmp-pdm1spatial))
      pdm2b = pdm2spatial.transpose(0, 3, 1, 2)
      print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1spatial, h1e)
         + 0.5 * np.einsum('ijkl,ijkl->', pdm2b, driver.unpack_g2e(g2e)) + ecore))

   t1 = time.time()
   print('\ndt=',t1-t0)

